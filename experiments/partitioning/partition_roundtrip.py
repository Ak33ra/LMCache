# SPDX-License-Identifier: Apache-2.0
"""Verify live private/shared resizing through public MP and Store APIs."""

# Standard
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

# Third Party
import httpx
import msgspec
import torch

# First Party
from examples.multi_process.cpu_inline_roundtrip import (
    CHUNK_COUNT,
    CHUNK_SIZE,
    MODEL,
    RPC_TIMEOUT,
    SHAPE,
)
from lmcache import torch_device_type
from lmcache.v1.distributed.api import (
    EncodedObjectKey,
    MemoryLayoutDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.inline import EngineDrivenContextInline
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
import lmcache
import lmcache.lmcache_mooncake as native_mooncake

from process_host import Process, ProcessHost

POOL_BYTES = 64 * 1024 * 1024
SDK_BYTES = 8 * 1024 * 1024
STAGING_BYTES = 16 * 1024 * 1024
TOTAL_BYTES = POOL_BYTES + SDK_BYTES + STAGING_BYTES
ALIGN_BYTES = 4096


class L1State(msgspec.Struct):
    """Allocator and lifetime counters returned by MP's public status API."""

    total_object_count: int
    memory_used_bytes: int
    memory_total_bytes: int
    staging_bytes: int
    read_locked_count: int
    write_locked_count: int
    temporary_count: int


class StoreState(msgspec.Struct):
    """Outstanding store work returned by MP's public status API."""

    pending_keys_count: int
    in_flight_task_count: int
    deferred_l1_deletion_count: int


class StorageState(msgspec.Struct):
    """The one private allocator and one shared adapter used in this check."""

    l1_manager: L1State
    store_controller: StoreState
    num_l2_adapters: int


class ServerState(msgspec.Struct):
    """Public MP server health and storage state."""

    is_healthy: bool
    storage_manager: StorageState


class ReplicaState(msgspec.Struct):
    """A read-only shared-copy descriptor returned by the Store master."""

    size_: int
    buffer_address_: int
    protocol_: str
    transport_endpoint_: str


class ReplicaEntry(msgspec.Struct):
    """The shared copies currently published for one object key."""

    ok: bool
    values: list[ReplicaState] = msgspec.field(default_factory=list)


class ReplicaQuery(msgspec.Struct):
    """A batch metadata lookup that does not grant a lease or update recency."""

    success: bool
    data: dict[str, ReplicaEntry]


def _key(request_id: str, token_offset: int = 0) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        MODEL,
        1,
        0,
        list(range(token_offset, token_offset + CHUNK_COUNT * CHUNK_SIZE)),
        end=CHUNK_COUNT * CHUNK_SIZE,
        request_id=request_id,
    )


def _wait(check: Callable[[], bool], description: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise TimeoutError(description)


def _ports(count: int) -> list[int]:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]


def _ready(http: httpx.Client, url: str, process: Process) -> None:
    def check() -> bool:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with {process.returncode}: {url}")
        try:
            return http.get(url, timeout=1).status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout):
            return False

    _wait(check, f"HTTP readiness timed out: {url}")


@dataclass
class Node:
    """One registered inline client and the HTTP view of its private cache."""

    name: str
    rpc: RequestClient
    transfer: EngineDrivenContextInline
    http: httpx.Client
    http_url: str
    output: Path
    instance_id: int
    shared_endpoint: str
    process: Process

    def state(self) -> ServerState:
        """Return validated public status; raise on an HTTP or schema failure."""
        response = self.http.get(f"{self.http_url}/status")
        response.raise_for_status()
        return msgspec.json.decode(response.content, type=ServerState)

    def settle(self, objects: int, label: str) -> None:
        """Check quiescent allocation ownership and save the observed counters.

        Args:
            objects: Expected resident object count after background work ends.
            label: Filename suffix for the recorded status snapshot.

        Raises:
            TimeoutError: Counters do not converge within thirty seconds.
        """

        def ready() -> bool:
            status = self.state()
            l1 = status.storage_manager.l1_manager
            store = status.storage_manager.store_controller
            return (
                status.is_healthy
                and status.storage_manager.num_l2_adapters == 1
                and l1.total_object_count == objects
                and l1.memory_used_bytes == objects * ALIGN_BYTES
                and l1.staging_bytes
                == l1.read_locked_count
                == l1.write_locked_count
                == 0
                and l1.temporary_count == 0
                and store.pending_keys_count == store.in_flight_task_count == 0
                and store.deferred_l1_deletion_count == 0
            )

        try:
            _wait(ready, f"{self.name}: expected {objects} quiescent private objects")
        finally:
            (self.output / f"{self.name}-{label}.json").write_bytes(
                msgspec.json.encode(self.state())
            )

    def read(
        self, label: str, expected: list[torch.Tensor], token_offset: int = 0
    ) -> int:
        """Read exact bytes or check a miss, then release the request session.

        Args:
            label: Unique request name within this server.
            expected: Expected chunks; an empty list requires a cache miss.
            token_offset: First token value identifying a distinct cache prefix.

        Returns:
            The observed completed lookup's hit count.

        Raises:
            RuntimeError: Lookup counts or inline transfer violate the contract.
            AssertionError: Returned bytes differ from the source snapshot.
        """
        key = _key(f"{self.name}-{label}", token_offset)
        self.rpc.lookup(key, 1).result(timeout=RPC_TIMEOUT)
        count = self.rpc.wait_prefetch_status(key.request_id, RPC_TIMEOUT).result(
            timeout=RPC_TIMEOUT + 1
        )
        if count != len(expected):
            raise RuntimeError(
                f"{key.request_id}: expected {len(expected)}, got {count}"
            )
        if expected:
            actual = self.transfer.prepare_retrieve(key, self.instance_id, CHUNK_COUNT)
            if actual is None:
                raise RuntimeError(f"{key.request_id}: lookup hit returned no bytes")
            for received, original in zip(actual, expected, strict=True):
                torch.testing.assert_close(received, original, rtol=0, atol=0)
            self.rpc.commit_retrieve(key, self.instance_id).result(timeout=RPC_TIMEOUT)
        self.rpc.end_session(key.request_id).result(timeout=RPC_TIMEOUT)
        return count


def _node(
    stack: ExitStack,
    name: str,
    ports: list[int],
    adapter: dict[str, str | int],
    http: httpx.Client,
    output: Path,
    host: ProcessHost,
) -> Node:
    rpc_port, http_port, te_port = ports
    config = dict(adapter, local_hostname=f"{host.address}:{te_port}")
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host",
        host.address,
        "--port",
        str(rpc_port),
        "--http-host",
        host.address,
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--hash-algorithm",
        "blake3",
        "--supported-transfer-mode",
        "engine_driven",
        "--kv-partition",
        msgspec.json.encode(
            {
                "total_bytes": TOTAL_BYTES,
                "staging_bytes": STAGING_BYTES,
                "sdk_scratch_bytes": SDK_BYTES,
                "alpha": 0.0,
            }
        ).decode(),
        "--l1-align-bytes",
        str(ALIGN_BYTES),
        "--shm-name",
        "",
        "--eviction-policy",
        "LRU",
        "--l2-adapter",
        msgspec.json.encode(config).decode(),
        "--disable-observability",
        "--disable-metrics",
    ]
    process = stack.enter_context(host.service(name, command, output))
    http_url = f"http://{host.address}:{http_port}"
    _ready(http, f"{http_url}/healthcheck", process)
    rpc = RequestClientFactory.create(f"tcp://{host.address}:{rpc_port}")
    stack.callback(rpc.close)
    instance_id = os.getpid()
    registration = rpc.register_kv_cache_engine_driven_context(
        RegisterEngineDrivenContextPayload(
            instance_id, MODEL, 1, 4, 2, 4, "float32", False
        )
    ).result(timeout=RPC_TIMEOUT)
    if registration.shm_name:
        raise RuntimeError("Expected typed inline transfer without SHM")
    transfer = EngineDrivenContextInline(
        EngineDrivenContextMetadata(
            MemoryLayoutDesc([SHAPE], [torch.float32]), 4, False
        ),
        rpc,
        RPC_TIMEOUT,
    )
    stack.callback(transfer.close)
    return Node(
        name,
        rpc,
        transfer,
        http,
        http_url,
        output,
        instance_id,
        f"{host.address}:{te_port}",
        process,
    )


def _shared_keys(http: httpx.Client, master: str) -> list[str]:
    response = http.get(f"{master}/get_all_keys")
    response.raise_for_status()
    return sorted(response.text.splitlines())


def _replicas(http: httpx.Client, master: str) -> dict[str, ReplicaEntry]:
    """Snapshot shared copies without acquiring leases or updating cache recency."""
    keys = _shared_keys(http, master)
    response = http.get(f"{master}/batch_query_keys", params={"keys": ",".join(keys)})
    response.raise_for_status()
    snapshot = msgspec.json.decode(response.content, type=ReplicaQuery)
    if (
        not snapshot.success
        or set(snapshot.data) != set(keys)
        or any(not entry.ok or not entry.values for entry in snapshot.data.values())
    ):
        raise RuntimeError(f"Shared objects are not readable: {response.text}")
    return snapshot.data


def _delete_shared(node: Node) -> None:
    key = _key("delete")
    hashes = TokenHasher(CHUNK_SIZE, "blake3").compute_packed_chunk_hashes(
        key.token_bytes
    )
    objects = ipc_key_to_object_keys(key, hashes, [0])[0]
    encoded = [
        asdict(EncodedObjectKey(k.chunk_hash.hex(), k.model_name, k.kv_rank))
        for k in objects
    ]
    response = node.http.request(
        "DELETE", f"{node.http_url}/cache/objects", json={"tier": "l2", "keys": encoded}
    )
    response.raise_for_status()
    (node.output / "delete-shared.json").write_bytes(response.content)


def _snapshot(node: Node, label: str) -> dict:
    response = node.http.get(f"{node.http_url}/partition")
    response.raise_for_status()
    status = response.json()
    if status["charged_bytes"] > TOTAL_BYTES:
        raise RuntimeError(f"Managed backing exceeded budget: {status}")
    maps = node.process.read_maps()
    (node.output / f"{node.name}-{label}.maps").write_text(maps)
    mapped = 0
    for line in maps.splitlines():
        if "/memfd:lmcache_dram_" in line:
            begin, end = (int(value, 16) for value in line.split()[0].split("-"))
            mapped += end - begin
    if not status["quiesced"] and mapped != status["mapped_bytes"]:
        raise RuntimeError(f"Private maps {mapped} != accounting {status}")
    shared = status["shared"]
    computed = (
        status["mapped_bytes"]
        + shared["scratch_bytes"]
        + shared["shared_allocated_bytes"]
        + shared["retained_failed_bytes"]
        + status["pending_growth_bytes"]
    )
    if computed != status["charged_bytes"]:
        raise RuntimeError(f"Charge components do not sum: {status}")
    status["observed_memfd_bytes"] = mapped
    status["pid"] = node.process.pid
    (node.output / f"{node.name}-partition-{label}.json").write_bytes(
        msgspec.json.encode(status)
    )
    return status


def _resize(node: Node, alpha: float, request_id: str) -> dict:
    url = f"{node.http_url}/partition/resize"
    payload = {"request_id": request_id, "alpha": alpha}
    before = _snapshot(node, f"{request_id}-before")
    started = time.perf_counter_ns()
    response = node.http.post(url, json=payload)
    response.raise_for_status()
    if response.status_code != 202:
        raise RuntimeError(response.text)
    sample = 0

    def complete() -> bool:
        nonlocal sample
        status = _snapshot(node, f"{request_id}-{sample}")
        sample += 1
        if status["resize"]["request_id"] != request_id:
            return False
        phase = status["resize"]["phase"]
        if phase == "failed":
            raise RuntimeError(f"Resize failed: {status}")
        return phase == "complete"

    _wait(complete, f"resize {request_id}")
    elapsed_ns = time.perf_counter_ns() - started
    (node.output / f"{node.name}-{request_id}-resize-timing.json").write_bytes(
        msgspec.json.encode(
            {
                "request_id": request_id,
                "alpha": alpha,
                "elapsed_ns": elapsed_ns,
                "poll_interval_ms": 50,
                "scope": "HTTP submit through observed completion, including map sampling",
            }
        )
    )
    result = node.http.get(f"{url}/{request_id}")
    result.raise_for_status()
    replay = node.http.post(url, json=payload)
    if replay.status_code != 202 or replay.json() != result.json():
        raise RuntimeError("Request replay changed completed resize result")
    conflict = node.http.post(
        url,
        json={"request_id": request_id, "alpha": 1.0 - alpha if alpha != 0.5 else 0.0},
    )
    (node.output / f"{node.name}-{request_id}-http-replay.json").write_bytes(
        msgspec.json.encode(
            {
                "replay_status": replay.status_code,
                "replay_body": replay.json(),
                "conflict_status": conflict.status_code,
                "conflict_body": conflict.json(),
            }
        )
    )
    if conflict.status_code != 409:
        raise RuntimeError(f"Conflicting request accepted: {conflict.text}")
    status = _snapshot(node, f"{request_id}-complete")
    capacity = status["current"]
    if capacity["private_effective_bytes"] != int(POOL_BYTES * alpha):
        raise RuntimeError(f"Unexpected effective split: {status}")
    active = {item["segment_id"] for item in status["shared"]["shared_segments"]}
    released = [
        item
        for item in before["shared"]["shared_segments"]
        if item["segment_id"] not in active
    ]
    ranges = []
    for line in node.process.read_maps().splitlines():
        start, end = [int(value, 16) for value in line.split()[0].split("-")]
        ranges.append((start, end, line))
    observations = []
    for item in released:
        overlaps = [
            line
            for start, end, line in ranges
            if start < item["base"] + item["size"] and end > item["base"]
        ]
        # Address reuse is allowed. Anonymous overlap cannot prove munmap.
        observations.append(
            {
                "released": item,
                "current_overlaps": overlaps,
                "range_absent": not overlaps,
            }
        )
    (node.output / f"{node.name}-{request_id}-released-ranges.json").write_bytes(
        msgspec.json.encode(observations)
    )
    return status


def _store(node: Node, label: str, token_offset: int = 0) -> list[torch.Tensor]:
    chunks = [
        torch.arange(SHAPE.numel(), dtype=torch.float32).reshape(SHAPE) + i * 1000
        for i in range(CHUNK_COUNT)
    ]
    expected = [chunk.clone() for chunk in chunks]
    key = _key(label, token_offset)
    node.transfer.prepare_store(key, node.instance_id)
    if not node.transfer.commit_store(key, node.instance_id, chunks):
        raise RuntimeError(f"{label}: inline store failed")
    for chunk in chunks:
        chunk.fill_(-1)
    node.rpc.end_session(key.request_id).result(timeout=RPC_TIMEOUT)
    return expected


def _verify(a: Node, b: Node, master: str, expected: list[torch.Tensor]) -> dict:
    b.read("remote-alpha-zero", expected)
    b.settle(0, "remote-alpha-zero")
    _snapshot(b, "remote-alpha-zero")
    invalid = b.http.post(
        f"{b.http_url}/partition/resize",
        json={"request_id": "invalid", "alpha": 1.1},
    )
    (b.output / "invalid-alpha.json").write_bytes(
        msgspec.json.encode({"status": invalid.status_code, "body": invalid.json()})
    )
    if invalid.status_code != 422:
        raise RuntimeError("Out-of-range alpha was accepted")
    _resize(b, 0.5, "promote")
    b.read("promote", expected)
    b.settle(CHUNK_COUNT, "promoted")
    b.rpc.clear().result(timeout=RPC_TIMEOUT)
    b.settle(0, "private-deleted")
    b.read("shared-survives-private-delete", expected)
    b.settle(CHUNK_COUNT, "promoted-again")
    _delete_shared(a)
    _wait(lambda: not _shared_keys(a.http, master), "shared deletion")
    b.read("private-survives-shared-delete", expected)
    b.settle(CHUNK_COUNT, "private-survives")
    _resize(b, 1.0, "all-private")
    before = _shared_keys(a.http, master)
    private_expected = _store(b, "private-only", 1000)
    b.read("private-only", private_expected, 1000)
    if _shared_keys(a.http, master) != before:
        raise RuntimeError("All-private store exported shared objects")
    a.read("peer-cannot-see-private", [], 1000)
    shared_expected = _store(a, "source-for-disabled-fetch", 2000)
    a.settle(0, "source-for-disabled-fetch")
    b.read("all-private-no-peer-fetch", [], 2000)
    original_pid = b.process.pid
    for cycle in range(2):
        _resize(b, 0.0, f"zero-{cycle}")
        b.read(f"zero-hit-{cycle}", shared_expected, 2000)
        b.settle(0, f"zero-hit-{cycle}")
        _resize(b, 0.5, f"half-{cycle}")
        b.read(f"half-hit-{cycle}", shared_expected, 2000)
        b.settle(CHUNK_COUNT, f"half-hit-{cycle}")
        _resize(b, 1.0, f"one-{cycle}")
        if b.process.poll() is not None or b.process.pid != original_pid:
            raise RuntimeError("Resize replaced the MP process")
    _resize(b, 0.0, "final-zero")
    return {
        "success": True,
        "same_pid": original_pid,
        "byte_verified_chunks_per_hit": CHUNK_COUNT,
        "budget_bytes_per_server": TOTAL_BYTES,
    }


def _workload(a: Node, b: Node, master: str) -> dict:
    """Measure serial byte-checked requests with identical shape and key order.

    Five interleaved rounds use alpha 0, 0.5, and 1. Latency includes lookup,
    completed prefetch, typed inline copy, exact tensor comparison, and session
    release. Alpha 1 seeds local data because peer fetch is disabled. These
    tiny requests characterize correctness and orchestration overhead, not
    bandwidth, production TTFT, or a ranking of partition configurations.
    """
    _resize(b, 1.0, "workload-publish-on-a")
    expected = _store(a, "workload-source", 3000)
    a.settle(0, "workload-source")
    replicas = _replicas(a.http, master)
    for entry in replicas.values():
        if any(
            replica.transport_endpoint_ != a.shared_endpoint for replica in entry.values
        ):
            raise RuntimeError("Workload source is not exclusively on fixed A")
    (b.output / "workload-source-replicas.json").write_bytes(
        msgspec.json.encode(replicas)
    )
    rows = []
    for repeat in range(5):
        for alpha in (0.0, 0.5, 1.0):
            label = f"workload-{repeat}-{alpha}"
            _resize(b, alpha, label)
            b.rpc.clear().result(timeout=RPC_TIMEOUT)
            b.settle(0, f"{label}-empty")
            if alpha == 1.0:
                keys = _shared_keys(a.http, master)
                _store(b, f"{label}-seed", 3000)
                b.settle(CHUNK_COUNT, f"{label}-seeded")
                if _shared_keys(a.http, master) != keys:
                    raise RuntimeError("Private workload seed exported shared keys")
            for request in range(3):
                resident = b.state().storage_manager.l1_manager.total_object_count
                started = time.perf_counter_ns()
                hits = b.read(f"{label}-{request}", expected, 3000)
                elapsed = time.perf_counter_ns() - started
                rows.append(
                    {
                        "round": repeat,
                        "alpha": alpha,
                        "request": request,
                        "elapsed_ns": elapsed,
                        "hit_chunks": hits,
                        "private_hit_chunks": hits if resident else 0,
                        "shared_hit_chunks": 0 if resident else hits,
                        "verified_bytes": CHUNK_COUNT * SHAPE.numel() * 4,
                    }
                )
                b.settle(CHUNK_COUNT if alpha else 0, f"{label}-{request}")
                _snapshot(b, f"{label}-{request}")
    latency = []
    for alpha, route in (
        (0.0, "shared"),
        (0.5, "shared"),
        (0.5, "private"),
        (1.0, "private"),
    ):
        samples = [
            row["elapsed_ns"] / 1_000_000
            for row in rows
            if row["alpha"] == alpha and row[f"{route}_hit_chunks"]
        ]
        latency.append(
            {
                "alpha": alpha,
                "route": route,
                "samples": len(samples),
                "median_ms": median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }
        )
    result = {
        "rows": rows,
        "latency": latency,
        "errors": 0,
        "claim": "Serial byte-checked functional workload; no speedup claim",
        "timing_scope": (
            "lookup, prefetch, inline copy, exact comparison, session release"
        ),
        "limiter": "Not profiled; small-payload orchestration hypothesis only",
        "concurrency": 1,
        "rounds": 5,
    }
    (b.output / "workload.json").write_bytes(msgspec.json.encode(result))
    return result


def main() -> None:
    """Run actual CPU services and preserve commands, hashes, logs, and snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload", action="store_true")
    parser.add_argument(
        "--lmcache-root",
        type=Path,
        default=Path(lmcache.__file__).resolve().parent.parent,
    )
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--peer-output", type=Path)
    args = parser.parse_args()
    if torch_device_type != "cpu":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for this CPU check")
    source = args.lmcache_root.resolve()
    imported_source = Path(lmcache.__file__).resolve().parent.parent
    if source != imported_source:
        parser.error(
            f"Set PYTHONPATH={source}; imported LMCache from {imported_source}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "native_binding": str(Path(native_mooncake.__file__).resolve()),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(
            Path(native_mooncake.__file__).read_bytes()
        ).hexdigest(),
        "master_sha256": hashlib.sha256(args.master_bin.read_bytes()).hexdigest(),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=source
        ).strip(),
        "managed_budget_bytes": TOTAL_BYTES,
        "request_payload_bytes": CHUNK_COUNT * SHAPE.numel() * 4,
        "concurrency": 1,
        "workspace_bound_bytes": 32 * 1024 * 1024,
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=source)
        ).hexdigest(),
        "load_average": list(os.getloadavg()),
        "cpu_count": os.cpu_count(),
    }
    (args.output_dir / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    deployment = json.loads(args.deployment.read_text()) if args.deployment else None
    local_address = deployment["node0"]["experiment_ip"] if deployment else "127.0.0.1"
    local = ProcessHost(source, args.output_dir, local_address)
    if deployment:
        if args.peer_output is None:
            parser.error("--deployment requires --peer-output")
        peer = ProcessHost(
            source,
            args.peer_output,
            deployment["node1"]["experiment_ip"],
            ssh=deployment["node1"]["ssh"],
        )
        ports = list(range(deployment["port_base"], deployment["port_base"] + 9))
    else:
        peer = local
        ports = _ports(9)
    rpc_port, metadata_port, admin_port = ports[:3]
    master_url = f"http://{local_address}:{admin_port}"
    command = [
        str(args.master_bin.resolve()),
        f"--rpc_address={local_address}",
        f"--port={rpc_port}",
        "--rpc_thread_num=2",
        "--handler_worker_thread_num=2",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={local_address}",
        f"--http_metadata_server_port={metadata_port}",
        "--enable_metric_reporting=true",
        f"--metrics_host={local_address}",
        f"--metrics_port={admin_port}",
        "--enable_offload=false",
        "--enable_disk_eviction=false",
        "--enable_cxl=false",
        "--allocation_strategy=local_first",
        "--dynamic_replication_mode=off",
    ]
    adapter = {
        "type": "mooncake_store",
        "num_workers": 2,
        "protocol": "tcp",
        "metadata_server": f"http://{local_address}:{metadata_port}/metadata",
        "master_server_addr": f"{local_address}:{rpc_port}",
        "global_segment_size": "0",
        "local_buffer_size": str(SDK_BYTES),
    }
    print(f"Evidence directory: {args.output_dir}", flush=True)
    with ExitStack() as stack:
        http = stack.enter_context(httpx.Client(timeout=RPC_TIMEOUT, trust_env=False))
        master = stack.enter_context(local.service("master", command, args.output_dir))
        _ready(http, f"{master_url}/health", master)
        a = _node(stack, "a", ports[3:6], adapter, http, args.output_dir, local)
        _snapshot(a, "initial")
        expected = _store(a, "initial-source")
        a.settle(0, "published")
        _wait(lambda: len(_shared_keys(http, master_url)) == CHUNK_COUNT, "publication")
        replicas = _replicas(http, master_url)
        for entry in replicas.values():
            if (
                len(entry.values) != 1
                or entry.values[0].transport_endpoint_ != a.shared_endpoint
            ):
                raise RuntimeError("Source replica does not belong exclusively to A")
        (args.output_dir / "initial-replicas.json").write_bytes(
            msgspec.json.encode(replicas)
        )
        b = _node(stack, "b", ports[6:9], adapter, http, args.output_dir, peer)
        summary = _verify(a, b, master_url, expected)
        summary["verified_chunk_sha256"] = [
            hashlib.sha256(chunk.numpy().tobytes()).hexdigest() for chunk in expected
        ]
        (args.output_dir / "correctness.json").write_bytes(msgspec.json.encode(summary))
        if args.workload:
            summary["workload"] = _workload(a, b, master_url)
        for node in (a, b):
            node.rpc.unregister_kv_cache_engine_driven_context(node.instance_id).result(
                timeout=RPC_TIMEOUT
            )
    summary["service_exits"] = {
        name: int((args.output_dir / f"{name}-exit.txt").read_text())
        for name in ("master", "a", "b")
    }
    (args.output_dir / "summary.json").write_bytes(msgspec.json.encode(summary))
    print(
        "PASS: live private/shared copies, endpoint limits, and repeated resizing",
        flush=True,
    )


if __name__ == "__main__":
    main()
