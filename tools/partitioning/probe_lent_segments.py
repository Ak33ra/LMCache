# SPDX-License-Identifier: Apache-2.0
"""Probe Store placement and quiescent lending of RAM-backed arena segments."""

# Standard
import argparse
from collections.abc import Callable
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import mmap
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

# Third Party
import mooncake
import torch
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

# First Party
import lmcache
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat


MIB = 1024 * 1024
SEGMENT_BYTES = 16 * MIB
ARENA_BYTES = 2 * SEGMENT_BYTES
SCRATCH_BYTES = 8 * MIB


def require(condition: bool, message: str) -> None:
    """Fail the probe if an observed contract does not hold."""
    if not condition:
        raise RuntimeError(message)


def free_port() -> int:
    """Select an unused loopback port for this local experiment."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def setup_store(
    master: str, metadata: str, contribution: int
) -> MooncakeDistributedStore:
    """Create one bounded TCP client without optional storage tiers."""
    store = MooncakeDistributedStore()
    status = store.setup(
        f"127.0.0.1:{free_port()}",
        metadata,
        contribution,
        SCRATCH_BYTES,
        "tcp",
        "",
        master,
    )
    require(status == 0, f"Store setup failed with {status}")
    return store


def peer_process(
    connection: Connection, master: str, metadata: str, contribution: int
) -> None:
    """Serve synchronous get requests in a separate Store client process."""
    store = setup_store(master, metadata, contribution)
    connection.send(store.get_hostname())
    try:
        while True:
            command, key = connection.recv()
            if command == "close":
                break
            if command == "get":
                connection.send(store.get(key))
            else:
                connection.send(store.probe_key(key))
    finally:
        store.close()
        connection.close()


def peer_call(connection: Connection, command: str, key: str) -> object:
    """Wait for a completed peer operation before allowing a handoff."""
    connection.send((command, key))
    require(connection.poll(30), f"Peer {command} timed out for {key}")
    return connection.recv()


def mapping_count(name: str) -> int:
    """Count mappings of this memfd in the owner process."""
    mappings = Path("/proc/self/maps").read_text().splitlines()
    return sum(name in line for line in mappings)


def create_memfd(name: str) -> int:
    """Create a Linux RAM-backed file even when Python omits os.memfd_create."""
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(name.encode(), 1)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


def placement_probe(store: MooncakeDistributedStore, peer: Connection) -> dict:
    """Show that a preferred owner and key prefix do not enforce privacy."""
    key = "private/owner-a/placement-probe"
    payload = b"private-placement-probe" * 128
    config = ReplicateConfig()
    config.replica_num = 1
    config.preferred_segment = store.get_hostname()
    require(store.put(key, payload, config) == 0, "Preferred-owner put failed")
    require(peer_call(peer, "get", key) == payload, "Peer did not read the object")
    descriptors = store.get_replica_desc(key)
    require(len(descriptors) == 1, "Expected exactly one shared replica")
    return {
        "owner_contribution_bytes": 0,
        "peer_contribution_bytes": SEGMENT_BYTES,
        "put_preferred_owner_without_capacity": "succeeded_on_peer",
        "prefixed_private_key_peer_read": "visible",
        "store_owned_private_contract": "failed",
    }


def lend_probe(
    store: MooncakeDistributedStore,
    peer: Connection,
    cycles: int,
    grace_seconds: int,
) -> dict:
    """Alternate a disjoint memfd region between LMCache allocation and Store."""
    name = f"kvc-lent-arena-{os.getpid()}"
    fd = create_memfd(name)
    os.ftruncate(fd, ARENA_BYTES)
    arena = mmap.mmap(fd, ARENA_BYTES)
    tensor = torch.frombuffer(arena, dtype=torch.uint8)
    private = TensorMemoryAllocator(tensor[:SEGMENT_BYTES])
    sentinel = private.allocate(torch.Size([4096]), torch.uint8, MemoryFormat.BINARY)
    require(sentinel is not None, "Private sentinel allocation failed")
    sentinel.tensor.fill_(113)
    require(
        peer_call(peer, "probe", "private/owner-a/sentinel") == 0,
        "Private key became visible",
    )
    records = []
    mounted_ids = []
    try:
        for cycle in range(cycles):
            require(mapping_count(name) == 1, "Store mapping survived prior handoff")
            mounted = store.mount_segment(
                f"/proc/self/fd/{fd}", SEGMENT_BYTES, SEGMENT_BYTES, "tcp"
            )
            require(mounted["ret"] == 0, f"Mount failed: {mounted}")
            mounted_ids = mounted["segment_ids"]
            require(len(mounted_ids) == 1, "Expected one coarse Store segment")
            require(mapping_count(name) == 2, "Store did not map the arena region")
            key = f"shared/cycle-{cycle}"
            payload = bytes([cycle + 17]) * 4096
            require(store.put(key, payload) == 0, "Shared put failed")
            require(peer_call(peer, "get", key) == payload, "Peer shared bytes differ")
            require(
                arena.find(payload, SEGMENT_BYTES) >= SEGMENT_BYTES,
                "Store did not write the lent physical backing",
            )
            require(
                bool(torch.all(sentinel.tensor == 113)),
                "Shared write corrupted private bytes",
            )
            require(
                store.unmount_segment(mounted_ids, grace_seconds) == 0,
                "Unmount failed",
            )
            if grace_seconds:
                require(
                    mapping_count(name) == 2,
                    "Expected accepted unmount to retain its mapping",
                )
                deadline = time.monotonic() + 45
                while mapping_count(name) == 2:
                    require(
                        time.monotonic() < deadline,
                        "Graceful cleanup did not complete",
                    )
                    time.sleep(0.05)
            mounted_ids = []
            require(mapping_count(name) == 1, "Store still maps the withdrawn region")
            require(
                peer_call(peer, "probe", key) == 0,
                "Withdrawn key remains advertised",
            )

            borrowed = TensorMemoryAllocator(tensor[SEGMENT_BYTES:])
            obj = borrowed.allocate(
                torch.Size([SEGMENT_BYTES]), torch.uint8, MemoryFormat.BINARY
            )
            require(
                obj is not None,
                "LMCache could not consume the entire withdrawn region",
            )
            obj.tensor.fill_(cycle + 37)
            require(
                borrowed.total_allocated_size == SEGMENT_BYTES,
                "Private charge differs from region size",
            )
            require(
                bool(torch.all(sentinel.tensor == 113)),
                "Private expansion corrupted original region",
            )
            require(
                peer_call(peer, "probe", key) == 0,
                "Private overwrite became shared",
            )
            borrowed.free(obj)
            require(
                borrowed.total_allocated_size == 0,
                "Private release leaked accounting",
            )
            del obj, borrowed
            gc.collect()
            records.append(
                {
                    "cycle": cycle,
                    "shared_peer_bytes": len(payload),
                    "private_consumed_bytes": SEGMENT_BYTES,
                    "arena_backing_bytes": os.fstat(fd).st_size,
                    "store_mappings_after_unmount": mapping_count(name) - 1,
                }
            )
    finally:
        if mounted_ids:
            require(store.unmount_segment(mounted_ids, 0) == 0, "Final unmount failed")
        private.free(sentinel)
        require(private.total_allocated_size == 0, "Private sentinel leaked accounting")
        del sentinel, private, tensor
        gc.collect()
        arena.close()
        os.close(fd)
    return {
        "arena_bytes": ARENA_BYTES,
        "segment_bytes": SEGMENT_BYTES,
        "scratch_bytes_per_client": SCRATCH_BYTES,
        "grace_seconds": grace_seconds,
        "cycles": records,
        "quiescent_lending": "passed",
        "scope": "allocator_and_store_sdk_only",
        "active_transfer_and_unregister_failure_safety": "unverified",
    }


def run_probe(
    mode: str,
    output: Path,
    cycles: int,
    grace_seconds: int,
    observation: (
        Callable[[MooncakeDistributedStore, Connection, str], dict] | None
    ) = None,
) -> dict:
    """Start an isolated master and two clients, retaining the native log."""
    output.mkdir(parents=True, exist_ok=True)
    master_port, metadata_port, metrics_port = free_port(), free_port(), free_port()
    master_addr = f"127.0.0.1:{master_port}"
    metadata_addr = f"http://127.0.0.1:{metadata_port}/metadata"
    master_binary = Path(next(iter(mooncake.__path__))) / "mooncake_master"
    log = (output / f"{mode}-master.log").open("w")
    master = subprocess.Popen(
        [
            str(master_binary),
            "--rpc_address=127.0.0.1",
            f"--rpc_port={master_port}",
            "--rpc_thread_num=2",
            "--handler_worker_thread_num=2",
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=127.0.0.1",
            f"--http_metadata_server_port={metadata_port}",
            "--metrics_host=127.0.0.1",
            f"--metrics_port={metrics_port}",
            "--enable_metric_reporting=false",
            "--enable_offload=false",
            "--logtostderr=true",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    owner = None
    peer = None
    parent_connection = None
    try:
        deadline = time.monotonic() + 15
        while True:
            require(master.poll() is None, f"Master exited; inspect {log.name}")
            try:
                with socket.create_connection(("127.0.0.1", master_port), timeout=0.2):
                    break
            except OSError:
                require(time.monotonic() < deadline, "Master did not start")
                time.sleep(0.05)
        owner = setup_store(master_addr, metadata_addr, 0)
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        peer = context.Process(
            target=peer_process,
            args=(
                child_connection,
                master_addr,
                metadata_addr,
                SEGMENT_BYTES if mode == "placement" else 0,
            ),
        )
        peer.start()
        child_connection.close()
        require(parent_connection.poll(30), "Peer setup timed out")
        peer_name = parent_connection.recv()
        if observation is not None:
            result = observation(owner, parent_connection, metadata_addr)
        elif mode == "placement":
            result = placement_probe(owner, parent_connection)
        else:
            result = lend_probe(owner, parent_connection, cycles, grace_seconds)
        result["peer"] = peer_name
        return result
    finally:
        original_error = sys.exception()
        cleanup_errors = []

        def close_peer() -> None:
            if peer is not None:
                if peer.is_alive() and parent_connection is not None:
                    try:
                        parent_connection.send(("close", ""))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                peer.join(15)
                if peer.is_alive():
                    peer.terminate()
                    peer.join(5)
                if peer.is_alive():
                    peer.kill()
                    peer.join(5)
                require(peer.exitcode == 0, f"Peer exit status: {peer.exitcode}")

        def close_master() -> None:
            master.terminate()
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait(timeout=5)
            require(
                master.returncode in (0, -signal.SIGTERM),
                f"Master exit status: {master.returncode}",
            )

        cleanup = [close_peer]
        if parent_connection is not None:
            cleanup.append(parent_connection.close)
        if owner is not None:
            cleanup.append(owner.close)
        cleanup.extend([close_master, log.close])
        for close in cleanup:
            try:
                close()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if original_error is not None:
                original_error.add_note(f"Cleanup also failed: {cleanup_errors!r}")
            else:
                raise ExceptionGroup("Probe cleanup failed", cleanup_errors)


def main() -> None:
    """Run the bounded prototype and write machine-readable observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["placement", "lend", "all"], default="all")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--grace-seconds", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/kvc-lent-segments"))
    args = parser.parse_args()
    require(1 <= args.cycles <= 32, "Cycles must be between 1 and 32")
    torch.set_num_threads(1)
    os.environ["MC_TCP_BIND_ADDRESS"] = "127.0.0.1"
    os.environ["MC_MIN_RPC_PORT"] = "26000"
    os.environ["MC_MAX_RPC_PORT"] = "26999"
    modes = ["placement", "lend"] if args.mode == "all" else [args.mode]
    results = {
        "mooncake_version": importlib.metadata.version("mooncake-transfer-engine"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "lmcache_import": str(Path(lmcache.__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observations": {},
    }
    for mode in modes:
        results["observations"][mode] = run_probe(
            mode, args.output, args.cycles, args.grace_seconds
        )
        text = json.dumps(results, indent=2)
        (args.output / "observations.json").write_text(text + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
