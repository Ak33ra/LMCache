# SPDX-License-Identifier: Apache-2.0
"""Verify bounded private/shared copies with two MP servers and Mooncake Store."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import os
import signal
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

POOL_BYTES = 64 * 1024 * 1024
SDK_BYTES = 16 * 1024 * 1024
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


class SegmentState(msgspec.Struct):
    """A mounted shared allocation reported by the Store master."""

    segment_name: str
    size_bytes: int
    allocator_capacity_bytes: int
    allocator_used_bytes: int
    te_endpoint: str


class SegmentsState(msgspec.Struct):
    """The master's complete mounted-segment inventory."""

    total_segments: int
    segments: list[SegmentState]


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


def _key(request_id: str) -> IPCCacheServerKey:
    """Use identical token content across clients with distinct sessions."""
    return IPCCacheServerKey.from_token_ids(
        MODEL,
        1,
        0,
        list(range(CHUNK_COUNT * CHUNK_SIZE)),
        end=CHUNK_COUNT * CHUNK_SIZE,
        request_id=request_id,
    )


def _wait(check: Callable[[], bool], description: str) -> None:
    """Wait up to thirty seconds for an observable asynchronous result."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise TimeoutError(description)


def _ports(count: int) -> list[int]:
    """Reserve distinct loopback ports together, then release them for startup."""
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]


@contextmanager
def _process(
    name: str,
    command: list[str],
    output: Path,
) -> Iterator[subprocess.Popen[bytes]]:
    """Own one service process and preserve its command, log, and exit status."""
    (output / f"{name}-command.json").write_bytes(msgspec.json.encode(command))
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LMCACHE_TRACK_USAGE="false",
        MC_TCP_BIND_ADDRESS="127.0.0.1",
        MC_NUM_TCP_IO_THREADS="2",
        LMCACHE_COORDINATOR_URL="",
    )
    environment.pop("MC_STORE_LOCAL_HOT_CACHE_SIZE", None)
    with (output / f"{name}.log").open("wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        try:
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError(f"{name} required SIGKILL; see {output}") from None
            (output / f"{name}-exit.txt").write_text(f"{process.returncode}\n")
            if process.returncode not in (0, -signal.SIGTERM):
                raise RuntimeError(
                    f"{name} exited with {process.returncode}; see {output}"
                )


def _ready(http: httpx.Client, url: str, process: subprocess.Popen[bytes]) -> None:
    """Wait for a service's HTTP readiness, failing promptly if it exits."""

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
                and l1.memory_total_bytes == POOL_BYTES
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

    def read(self, label: str, expected: list[torch.Tensor]) -> int:
        """Read exact bytes or check a miss, then release the request session.

        Args:
            label: Unique request name within this server.
            expected: Expected chunks; an empty list requires a cache miss.

        Returns:
            The observed completed lookup's hit count.

        Raises:
            RuntimeError: Lookup counts or inline transfer violate the contract.
            AssertionError: Returned bytes differ from the source snapshot.
        """
        key = _key(f"{self.name}-{label}")
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
    policy: str,
    http: httpx.Client,
    output: Path,
) -> Node:
    """Launch an MP HTTP/TCP server and register its typed inline client."""
    rpc_port, http_port, te_port = ports
    config = dict(adapter, local_hostname=f"127.0.0.1:{te_port}")
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(rpc_port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--hash-algorithm",
        "blake3",
        "--supported-transfer-mode",
        "engine_driven",
        "--l1-size-gb",
        str(POOL_BYTES / (1 << 30)),
        "--no-l1-use-lazy",
        "--l1-align-bytes",
        str(ALIGN_BYTES),
        "--shm-name",
        "",
        "--eviction-policy",
        "LRU",
        "--l2-store-policy",
        policy,
        "--l2-prefetch-policy",
        "retain" if policy == "default" else "default",
        "--l2-adapter",
        msgspec.json.encode(config).decode(),
        "--disable-observability",
        "--disable-metrics",
    ]
    process = stack.enter_context(_process(name, command, output))
    http_url = f"http://127.0.0.1:{http_port}"
    _ready(http, f"{http_url}/healthcheck", process)
    rpc = RequestClientFactory.create(f"tcp://127.0.0.1:{rpc_port}")
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
        f"127.0.0.1:{te_port}",
    )


def _shared_keys(http: httpx.Client, master: str) -> list[str]:
    """Read the isolated master's actual published object namespace."""
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


def _segments(
    http: httpx.Client,
    master: str,
    output: Path,
    label: str,
    expected_count: int = 2,
) -> SegmentsState:
    """Check actual shared backing capacity and preserve allocator occupancy."""
    response = http.get(f"{master}/get_segments_detail")
    response.raise_for_status()
    (output / f"shared-segments-{label}.json").write_bytes(response.content)
    snapshot = msgspec.json.decode(response.content, type=SegmentsState)
    if (
        snapshot.total_segments != expected_count
        or len(snapshot.segments) != expected_count
        or any(
            s.size_bytes != POOL_BYTES or s.allocator_capacity_bytes != POOL_BYTES
            for s in snapshot.segments
        )
    ):
        raise RuntimeError(
            f"Expected {expected_count} shared segments of {POOL_BYTES} bytes"
        )
    return snapshot


def _delete_shared(node: Node) -> None:
    """Delete the exact shared keys through the existing tier-specific HTTP API."""
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


def _publish(
    a: Node, master: str, policy: str
) -> tuple[list[torch.Tensor], dict[str, ReplicaEntry]]:
    """Publish on A before mounting B so every shared replica must belong to A."""
    http, output = a.http, a.output
    a.settle(0, "empty")
    if _shared_keys(http, master):
        raise RuntimeError("Expected an empty isolated Store master")
    a.read("initial-miss", [])
    chunks = [
        torch.arange(SHAPE.numel(), dtype=torch.float32).reshape(SHAPE) + i * 1000
        for i in range(CHUNK_COUNT)
    ]
    expected = [chunk.clone() for chunk in chunks]
    key = _key("store-a")
    a.transfer.prepare_store(key, a.instance_id)
    if not a.transfer.commit_store(key, a.instance_id, chunks):
        raise RuntimeError("A's inline store failed")
    for chunk in chunks:
        chunk.fill_(-1)
    a.rpc.end_session(key.request_id).result(timeout=RPC_TIMEOUT)
    _wait(lambda: len(_shared_keys(http, master)) == CHUNK_COUNT, "shared publication")
    retained = CHUNK_COUNT if policy == "default" else 0
    a.settle(retained, "published")
    shared = _replicas(http, master)
    (output / "shared-replicas.json").write_bytes(msgspec.json.encode(shared))
    for entry in shared.values():
        if len(entry.values) != 1 or (
            entry.values[0].transport_endpoint_ != a.shared_endpoint
            or entry.values[0].size_ != SHAPE.numel() * 4
            or entry.values[0].protocol_ != "tcp"
        ):
            raise RuntimeError(
                f"Expected one TCP shared replica at {a.shared_endpoint}: {entry}"
            )
    _segments(http, master, output, "published-before-b", expected_count=1)
    return expected, shared


def _scenario(
    a: Node,
    b: Node,
    master: str,
    policy: str,
    expected: list[torch.Tensor],
    shared: dict[str, ReplicaEntry],
) -> dict[str, object]:
    """Prove exact B bytes, policy retention, and independent partition deletion."""
    http, output = a.http, a.output
    retained = CHUNK_COUNT if policy == "default" else 0
    b.settle(0, "before-shared-fetch")
    shared_hits = b.read("from-shared", expected)
    b.settle(retained, "after-shared-fetch")
    b_private = b.state().storage_manager.l1_manager
    after_fetch = _replicas(http, master)
    (output / "shared-replicas-after-fetch.json").write_bytes(
        msgspec.json.encode(after_fetch)
    )
    if after_fetch != shared:
        raise RuntimeError("B's fetch changed the shared replica set")
    a.rpc.clear().result(timeout=RPC_TIMEOUT)
    a.settle(0, "private-cleared")
    a.read("shared-survives-private-clear", expected)
    a.settle(retained, "after-shared-refetch")
    if _replicas(http, master) != shared:
        raise RuntimeError("Private removal/refetch changed shared replicas")
    _delete_shared(a)
    _wait(lambda: not _shared_keys(http, master), "shared deletion")
    (output / "shared-keys-after-delete.json").write_bytes(
        msgspec.json.encode(_shared_keys(http, master))
    )
    private_hits = b.read("after-shared-delete", expected if retained else [])
    b.settle(retained, "after-shared-delete")
    if _shared_keys(http, master):
        raise RuntimeError("Private read re-exported shared objects")
    for node in (a, b):
        node.rpc.clear().result(timeout=RPC_TIMEOUT)
        node.settle(0, "final")
        node.read("final-miss", [])
        node.rpc.unregister_kv_cache_engine_driven_context(node.instance_id).result(
            timeout=RPC_TIMEOUT
        )
    final_shared = _segments(http, master, output, "final")
    if any(segment.allocator_used_bytes for segment in final_shared.segments):
        raise RuntimeError("Shared allocator still charges deleted objects")
    print(
        f"PASS: {policy}: published shared bytes survive "
        "source mutation and private clear"
    )
    print(
        f"PASS: {policy}: retained private objects per server={retained}; "
        "shared deletion verified"
    )
    return {
        "expected": {
            "shared_hit_chunks": CHUNK_COUNT,
            "b_private_objects_after_fetch": retained,
            "b_private_bytes_after_fetch": retained * ALIGN_BYTES,
            "b_hit_chunks_after_shared_delete": retained,
            "final_private_objects": [0, 0],
            "final_shared_used_bytes": [0, 0],
        },
        "observed": {
            "shared_hit_chunks": shared_hits,
            "b_private_objects_after_fetch": b_private.total_object_count,
            "b_private_bytes_after_fetch": b_private.memory_used_bytes,
            "b_hit_chunks_after_shared_delete": private_hits,
            "final_private_objects": [
                node.state().storage_manager.l1_manager.total_object_count
                for node in (a, b)
            ],
            "final_shared_used_bytes": [
                segment.allocator_used_bytes for segment in final_shared.segments
            ],
        },
    }


def main() -> None:
    """Run one isolated live policy check and preserve commands, logs, and counters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-bin", type=Path, required=True)
    parser.add_argument("--store-policy", choices=("default", "skip_l1"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if torch_device_type != "cpu":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for the CPU baseline")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "native_binding": str(Path(native_mooncake.__file__).resolve()),
        "master_binary": str(args.master_bin.resolve()),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "store_policy": args.store_policy,
        "prefetch_policy": "retain" if args.store_policy == "default" else "default",
        "private_bytes_per_server": POOL_BYTES,
        "shared_bytes_per_server": POOL_BYTES,
        "sdk_buffer_bytes_per_server": SDK_BYTES,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(lmcache.__file__).resolve().parent.parent,
        ).strip(),
    }
    (args.output_dir / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    ports = _ports(9)
    rpc_port, metadata_port, admin_port = ports[:3]
    master_url = f"http://127.0.0.1:{admin_port}"
    master_command = [
        str(args.master_bin.resolve()),
        "--rpc_address=127.0.0.1",
        f"--port={rpc_port}",
        "--rpc_thread_num=2",
        "--handler_worker_thread_num=2",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=127.0.0.1",
        f"--http_metadata_server_port={metadata_port}",
        "--enable_metric_reporting=true",
        "--metrics_host=127.0.0.1",
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
        "metadata_server": f"http://127.0.0.1:{metadata_port}/metadata",
        "master_server_addr": f"127.0.0.1:{rpc_port}",
        "global_segment_size": str(POOL_BYTES),
        "local_buffer_size": str(SDK_BYTES),
    }
    print(f"Evidence directory: {args.output_dir}", flush=True)
    print(f"Native binding: {native_mooncake.__file__}", flush=True)
    with ExitStack() as stack:
        http = stack.enter_context(httpx.Client(timeout=RPC_TIMEOUT, trust_env=False))
        master = stack.enter_context(
            _process("master", master_command, args.output_dir)
        )
        _ready(http, f"{master_url}/health", master)
        a = _node(
            stack, "a", ports[3:6], adapter, args.store_policy, http, args.output_dir
        )
        _segments(http, master_url, args.output_dir, "initial", expected_count=1)
        expected, shared = _publish(a, master_url, args.store_policy)
        b = _node(
            stack, "b", ports[6:9], adapter, args.store_policy, http, args.output_dir
        )
        version = http.get(f"{master_url}/version")
        version.raise_for_status()
        (args.output_dir / "master-version.json").write_bytes(version.content)
        _segments(http, master_url, args.output_dir, "both-mounted")
        summary = _scenario(a, b, master_url, args.store_policy, expected, shared)
    summary["success"] = True
    summary["store_policy"] = args.store_policy
    summary["service_exits"] = {
        name: int((args.output_dir / f"{name}-exit.txt").read_text())
        for name in ("master", "a", "b")
    }
    (args.output_dir / "summary.json").write_bytes(msgspec.json.encode(summary))
    print(
        "PASS: all owned service processes exited; logs and allocation counters saved"
    )


if __name__ == "__main__":
    main()
