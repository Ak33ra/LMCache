# SPDX-License-Identifier: Apache-2.0
"""Observe SDK-owned capacity release and the lifetime of accepted TCP reads."""

# Standard
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import struct
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

# Third Party
import torch
from mooncake.store import MooncakeDistributedStore

# First Party
from lmcache import lmcache_native
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager

# Local
from probe_lent_segments import MIB, peer_call, require, run_probe


def metadata_for(store: MooncakeDistributedStore, metadata_url: str) -> dict:
    """Read the real transport descriptor from this experiment's master."""
    query = urlencode({"key": "mooncake/ram/" + store.get_hostname()})
    with urlopen(metadata_url + "?" + query, timeout=10) as response:
        return json.load(response)


def mapped_bytes(address: int, size: int) -> int:
    """Count mapped virtual bytes intersecting a known backing allocation."""
    total = 0
    for line in Path("/proc/self/maps").read_text().splitlines():
        interval = line.split()[0]
        begin, end = [int(value, 16) for value in interval.split("-")]
        total += max(0, min(end, address + size) - max(begin, address))
    return total


def sdk_handoff(
    store: MooncakeDistributedStore,
    peer: Connection,
    metadata_url: str,
    cycles: int,
) -> dict:
    """Transfer a coarse capacity grant only after observed mapping release."""
    requested = 16 * MIB + 1
    records = []
    gate = "passed_quiescent_backing_lifecycle"
    for cycle in range(cycles):
        mounted = store.allocate_and_mount_segment(requested, "tcp")
        require(mounted["ret"] == 0, f"SDK allocation failed: {mounted}")
        effective = mounted["allocated_size"]
        require(
            requested <= effective <= 64 * MIB,
            f"Unexpected allocation size: {mounted}",
        )
        ids = mounted["segment_ids"]
        key = f"sdk-owned/{cycle}"
        payload = bytes([cycle + 61]) * 4096
        require(store.put(key, payload) == 0, "SDK-owned shared put failed")
        require(peer_call(peer, "get", key) == payload, "SDK-owned peer bytes differ")
        descriptor = store.get_replica_desc(key)[0].get_memory_descriptor()
        pointer = descriptor.buffer_descriptor.buffer_address
        buffers = metadata_for(store, metadata_url)["buffers"]
        region = next(
            item
            for item in buffers
            if item["addr"] <= pointer < item["addr"] + item["length"]
        )
        require(
            region["length"] == effective,
            "Registered capacity differs from allocation",
        )
        require(
            mapped_bytes(region["addr"], effective) == effective,
            "Store backing is not mapped",
        )
        require(store.unmount_and_free_segment(ids, 0) == 0, "SDK unmount failed")
        remaining = mapped_bytes(region["addr"], effective)
        record = {
            "cycle": cycle,
            "requested_shared_bytes": requested,
            "effective_shared_bytes": effective,
            "shared_mapped_bytes_after_free": remaining,
            "private_granted_bytes": 0,
        }
        records.append(record)
        if remaining:
            gate = "blocked_shared_physical_release_unestablished"
            break

        private = L1Manager(
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    size_in_bytes=effective,
                    use_lazy=False,
                    shm_name="",
                )
            )
        )
        desc = private.get_l1_memory_desc()
        record["private_granted_bytes"] = effective
        private_key = ObjectKey(ObjectKey.IntHash2Bytes(cycle + 1), "sdk-handoff", 0)
        layout = MemoryLayoutDesc(
            [torch.Size([2, 1, 1, effective // 2])], [torch.uint8]
        )
        try:
            error, obj = private.reserve_write([private_key], [False], layout)[
                private_key
            ]
            require(
                error == L1Error.SUCCESS and obj is not None,
                "Private full-region allocation failed",
            )
            tensor = obj.get_tensor(0)
            tensor.fill_(cycle + 71)
            del tensor, obj
            require(
                private.finish_write([private_key])[private_key] == L1Error.SUCCESS,
                "Private publication failed",
            )
            error, obj = private.reserve_read([private_key])[private_key]
            require(error == L1Error.SUCCESS and obj is not None, "Private read failed")
            require(
                bool(torch.all(obj.get_tensor(0) == cycle + 71)),
                "Private bytes differ",
            )
            del obj
            require(
                private.delete([private_key])[private_key] == L1Error.KEY_IS_LOCKED,
                "Native read reservation did not block deletion",
            )
            require(
                private.finish_read([private_key])[private_key] == L1Error.SUCCESS,
                "Private read release failed",
            )
            require(
                private.delete([private_key])[private_key] == L1Error.SUCCESS,
                "Private deletion failed",
            )
            require(
                private.get_memory_usage() == (0, effective),
                "Private object cleanup leaked",
            )
        finally:
            private.close()
        del private
        gc.collect()
        remaining = mapped_bytes(desc.ptr, desc.size)
        record["private_mapped_bytes_after_close"] = remaining
        record["shared_granted_before_private_release"] = 0
        if remaining:
            gate = "blocked_private_physical_release_unestablished"
            break
    return {
        "gate": gate,
        "private_backing": "anonymous",
        "cycles": records,
        "private_pool_lifecycle": "empty_L1Manager_created_and_closed_each_cycle",
        "existing_L1Manager_resize": "not_implemented",
        "mapping_observation": "virtual_mapping_presence_not_RSS_or_malloc_ownership",
    }


def receive_exact(sock: socket.socket, count: int) -> bytes:
    """Read one bounded protocol frame."""
    result = bytearray()
    while len(result) < count:
        chunk = sock.recv(count - len(result))
        require(bool(chunk), "TCP transfer ended before the expected frame")
        result.extend(chunk)
    return bytes(result)


def accepted_tcp_read(
    store: MooncakeDistributedStore,
    peer: Connection,
    metadata_url: str,
    policy: str,
) -> dict:
    """Keep a source alive while proving unregister does not drain its reader."""
    del peer
    size = 64 * MIB
    source = torch.full((size,), 23, dtype=torch.uint8)
    pointer = source.data_ptr()
    require(store.register_buffer(pointer, size) == 0, "Source registration failed")
    metadata = metadata_for(store, metadata_url)
    require(metadata["tcp_proto_version"] == 2, "Probe requires TCP framing version 2")
    registered = True
    sock = socket.socket()
    sock.settimeout(30)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    try:
        sock.connect((metadata["tcp_data_host"], metadata["tcp_data_port"]))
        sock.sendall(struct.pack("<QQB7x", size, pointer, 0x80))
        status = struct.unpack("<Q", receive_exact(sock, 8))[0]
        require(status == 0x4D43545600000000, f"Read was not accepted: {status}")
        initial = receive_exact(sock, 4096)
        require(initial == bytes([23]) * 4096, "Accepted read starts with wrong bytes")
        if policy == "repro":
            require(store.unregister_buffer(pointer) == 0, "Source unregister failed")
            registered = False
            after = metadata_for(store, metadata_url)
            require(
                all(item["addr"] != pointer for item in after["buffers"]),
                "Source remains registered",
            )
            source[-4096:].fill_(89)
        received = len(initial)
        tail = b""
        while received < size:
            chunk = sock.recv(min(MIB, size - received))
            require(bool(chunk), "Accepted read stopped after unregister")
            received += len(chunk)
            tail = (tail + chunk)[-4096:]
        expected = 89 if policy == "repro" else 23
        require(tail == bytes([expected]) * 4096, "Delayed read tail differs")
        if policy == "drain-first":
            require(store.unregister_buffer(pointer) == 0, "Drained unregister failed")
            registered = False
        return {
            "accepted_source_bytes": size,
            "bytes_consumed_before_unregister": (
                len(initial) if policy == "repro" else received
            ),
            "received_bytes": received,
            "policy": policy,
            "post_unregister_source_access": (
                "observed_changed_tail" if policy == "repro" else "not_requested"
            ),
            "source_backing_retained_through_completion": True,
            "capacity_granted_while_read_incomplete": 0,
            "gate": (
                "blocked_unregister_is_not_a_TCP_drain"
                if policy == "repro"
                else "passed_harness_drains_its_only_known_read"
            ),
            "topology": "raw_loopback_TCP_socket_in_owner_process",
        }
    finally:
        sock.close()
        if registered:
            store.unregister_buffer(pointer)
        store.close()
        del source


def stale_tcp_header(
    store: MooncakeDistributedStore, peer: Connection, metadata_url: str
) -> dict:
    """Replay a real v2 request after a new registration at the same address."""
    del peer
    size = 4096
    source = torch.full((size,), 23, dtype=torch.uint8)
    pointer = source.data_ptr()
    require(store.register_buffer(pointer, size) == 0, "Initial registration failed")
    metadata = metadata_for(store, metadata_url)
    require(metadata["tcp_proto_version"] == 2, "Probe requires TCP framing version 2")
    header = struct.pack("<QQB7x", size, pointer, 0x80)
    registered = True
    sock = socket.socket()
    sock.settimeout(10)
    try:
        sock.connect((metadata["tcp_data_host"], metadata["tcp_data_port"]))
        sock.sendall(header)
        require(
            receive_exact(sock, 8) == struct.pack("<Q", 0x4D43545600000000),
            "Original header rejected",
        )
        require(
            receive_exact(sock, size) == bytes([23]) * size, "Original bytes differ"
        )
        require(
            store.unregister_buffer(pointer) == 0,
            "Old registration removal failed",
        )
        registered = False
        source.fill_(89)
        require(store.register_buffer(pointer, size) == 0, "New registration failed")
        registered = True
        sock.sendall(header)
        require(
            receive_exact(sock, 8) == struct.pack("<Q", 0x4D43545600000000),
            "Replayed header rejected",
        )
        require(
            receive_exact(sock, size) == bytes([89]) * size,
            "Replayed header did not access new bytes",
        )
        return {
            "source_bytes": size,
            "identical_v2_header_replayed": True,
            "new_registration_bytes_returned_to_old_header": True,
            "gate": "blocked_v2_header_has_no_registration_generation",
            "topology": "raw_loopback_TCP_socket_in_owner_process",
        }
    finally:
        sock.close()
        if registered:
            store.unregister_buffer(pointer)
        store.close()
        del source


def main() -> None:
    """Write reproducible quiescent and delayed-read observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["handoff", "tcp-read", "stale-header"], required=True
    )
    parser.add_argument(
        "--tcp-policy", choices=["repro", "drain-first"], default="repro"
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(1 <= args.cycles <= 16, "Cycles must be between 1 and 16")
    torch.set_num_threads(1)
    os.environ["MC_TCP_BIND_ADDRESS"] = "127.0.0.1"
    os.environ["MC_MIN_RPC_PORT"] = "26000"
    os.environ["MC_MAX_RPC_PORT"] = "26999"

    def observation(
        store: MooncakeDistributedStore, peer: Connection, metadata_url: str
    ) -> dict:
        if args.mode == "handoff":
            return sdk_handoff(store, peer, metadata_url, args.cycles)
        if args.mode == "tcp-read":
            return accepted_tcp_read(store, peer, metadata_url, args.tcp_policy)
        return stale_tcp_header(store, peer, metadata_url)

    result = {
        "mooncake_version": importlib.metadata.version("mooncake-transfer-engine"),
        "python": sys.version.split()[0],
        "native_extension": lmcache_native.__file__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observations": run_probe(args.mode, args.output, args.cycles, 0, observation),
    }
    text = json.dumps(result, indent=2)
    (args.output / "observations.json").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
