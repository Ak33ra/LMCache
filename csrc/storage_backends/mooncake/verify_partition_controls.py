# SPDX-License-Identifier: Apache-2.0
"""Exercise native partition controls against an isolated real TCP master."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import ctypes
import json
import mmap
import os
import signal
import socket
import subprocess
import time

# First Party
import lmcache.lmcache_mooncake as native

MIB = 1024 * 1024
PRIVATE_BYTES = 64 * 1024
REQUESTED_SHARED_BYTES = 64 * MIB + 1


def free_port() -> int:
    """Select a loopback listener port for the isolated probe."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def snapshot(status: native.PartitionStatus) -> dict[str, object]:
    """Serialize the native accounting snapshot for manual inspection."""
    return {
        "enabled": status.enabled,
        "closing": status.closing,
        "closed": status.closed,
        "scratch_bytes": status.scratch_bytes,
        "allocation_granularity_bytes": status.allocation_granularity_bytes,
        "pending_shared_bytes": status.pending_shared_bytes,
        "shared_allocated_bytes": status.shared_allocated_bytes,
        "retained_failed_bytes": status.retained_failed_bytes,
        "local_buffers": [
            {"id": b.id, "base": b.base, "bytes": b.size, "state": str(b.state)}
            for b in status.local_buffers
        ],
        "shared_segments": [
            {"id": s.id, "base": s.base, "bytes": s.size}
            for s in status.shared_segments
        ],
    }


def success(result: native.PartitionControlResult) -> native.PartitionControlResult:
    """Require confirmed SDK completion rather than an accepted request."""
    if result.code != native.PartitionControlCode.SUCCESS:
        raise RuntimeError(
            f"{result.code}: SDK {result.sdk_error}, {snapshot(result.snapshot)}"
        )
    return result


def completed(
    client: native.LMCacheMooncakeClient, future: int
) -> tuple[int, bool, str, list[bool] | None]:
    """Wait for the native connector's terminal completion of one request."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for result in client.drain_completions():
            assert result[0] == future
            return result
        time.sleep(0.01)
    raise TimeoutError(f"Native future {future} did not complete")


def mapped(base: int, size: int) -> bool:
    """Check whether a released allocation still overlaps a process mapping."""
    for line in Path("/proc/self/maps").read_text().splitlines():
        lower, upper = (int(value, 16) for value in line.split()[0].split("-"))
        if lower < base + size and base < upper:
            return True
    return False


def run(master_binary: Path, output: Path, fail_mount: bool) -> None:
    """Run real ownership transitions and retain logs and observed accounting."""
    output.mkdir(parents=True, exist_ok=False)
    rpc, metadata, metrics = free_port(), free_port(), free_port()
    log = (output / "master.log").open("w")
    master = subprocess.Popen(
        [
            str(master_binary),
            "--rpc_address=127.0.0.1",
            f"--rpc_port={rpc}",
            "--rpc_thread_num=2",
            "--handler_worker_thread_num=2",
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=127.0.0.1",
            f"--http_metadata_server_port={metadata}",
            "--metrics_host=127.0.0.1",
            f"--metrics_port={metrics}",
            "--enable_metric_reporting=false",
            "--enable_offload=false",
            "--logtostderr=true",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    client = None
    source = mmap.mmap(-1, PRIVATE_BYTES)
    destination = mmap.mmap(-1, PRIVATE_BYTES)
    source_view, destination_view = memoryview(source), memoryview(destination)
    records = {
        "native_extension": native.__file__,
        "requested_shared_bytes": REQUESTED_SHARED_BYTES,
    }
    try:
        deadline = time.monotonic() + 15
        while True:
            assert master.poll() is None, "Master exited; inspect master.log"
            try:
                with socket.create_connection(("127.0.0.1", rpc), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Master did not start") from None
                time.sleep(0.05)
        config = {
            "local_hostname": f"127.0.0.1:{free_port()}",
            "metadata_server": f"http://127.0.0.1:{metadata}/metadata",
            "master_server_addr": f"127.0.0.1:{rpc}",
            "protocol": "tcp",
            "global_segment_size": "0",
            "local_buffer_size": str(8 * MIB),
        }
        assert native.partition_control_api_version == 1
        for invalid in (
            {**config, "global_segment_size": str(8 * MIB)},
            {**config, "protocol": "rdma"},
        ):
            try:
                native.LMCacheMooncakeClient(invalid, 1, partition_controls=True)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("Unsafe partition configuration accepted")
        client = native.LMCacheMooncakeClient(config, 1, partition_controls=True)
        source_base = ctypes.addressof(ctypes.c_char.from_buffer(source))
        destination_base = ctypes.addressof(ctypes.c_char.from_buffer(destination))
        source_id = success(
            client.register_l1_buffer(source_base, PRIVATE_BYTES)
        ).local_buffer_id
        destination_id = success(
            client.register_l1_buffer(destination_base, PRIVATE_BYTES)
        ).local_buffer_id
        assert (
            success(
                client.register_l1_buffer(source_base, PRIVATE_BYTES)
            ).local_buffer_id
            == source_id
        )
        assert (
            client.register_l1_buffer(source_base + 1, 16).code
            == native.PartitionControlCode.INVALID
        )
        records["private_only"] = snapshot(client.partition_status())
        assert records["private_only"]["shared_allocated_bytes"] == 0
        expected = bytes(range(256)) * (PRIVATE_BYTES // 256)
        source[:] = expected
        result = completed(
            client, client.submit_batch_set(["no-shared-capacity"], [source_view])
        )
        assert not result[1], "Private registration became Store allocation capacity"
        cycles = []
        for cycle in range(2):
            allocation = success(client.allocate_shared_segment(REQUESTED_SHARED_BYTES))
            assert allocation.snapshot.shared_allocated_bytes >= REQUESTED_SHARED_BYTES
            assert allocation.snapshot.retained_failed_bytes == 0
            key = f"roundtrip-{cycle}"
            source[:] = expected
            assert completed(client, client.submit_batch_set([key], [source_view]))[1]
            source[:] = b"x" * PRIVATE_BYTES
            result = completed(
                client, client.submit_batch_get([key], [destination_view])
            )
            assert result[1] and result[3] == [True]
            assert destination[:] == expected
            deleted = completed(client, client.submit_batch_delete([key]))
            assert deleted[1] and deleted[3] == [True]
            retired = list(allocation.snapshot.shared_segments)
            for segment in retired:
                release = success(client.release_shared_segment(segment.id))
                assert release.released_shared_bytes == segment.size
                assert (
                    client.release_shared_segment(segment.id).code
                    == native.PartitionControlCode.INVALID
                )
            assert client.partition_status().shared_allocated_bytes == 0
            assert all(not mapped(s.base, s.size) for s in retired)
            cycles.append(
                {
                    "allocated": snapshot(allocation.snapshot),
                    "released": snapshot(client.partition_status()),
                    "mapping_removed": all(not mapped(s.base, s.size) for s in retired),
                }
            )
        records["cycles"] = cycles
        os.kill(master.pid, signal.SIGSTOP)
        with ThreadPoolExecutor(1) as pool:
            growth = pool.submit(client.allocate_shared_segment, REQUESTED_SHARED_BYTES)
            try:
                deadline = time.monotonic() + 3
                while not client.partition_status().pending_shared_bytes:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Growth did not expose its pending charge")
                    time.sleep(0.01)
                pending = client.partition_status()
                alignment = pending.allocation_granularity_bytes
                assert pending.pending_shared_bytes == (
                    (REQUESTED_SHARED_BYTES + alignment - 1) // alignment * alignment
                )
                assert client.close_partition().code == native.PartitionControlCode.BUSY
                records["pending_growth"] = snapshot(pending)
            finally:
                os.kill(master.pid, signal.SIGCONT)
            grown = success(growth.result(timeout=30))
            assert grown.snapshot.pending_shared_bytes == 0
            for segment in grown.segment_ids:
                success(client.release_shared_segment(segment))
        os.kill(master.pid, signal.SIGSTOP)
        try:
            future = client.submit_batch_exists(["blocked-master"])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                result = client.register_l1_buffer(0, 0)
                if result.code == native.PartitionControlCode.BUSY:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Did not observe worker SDK access")
            assert (
                client.unregister_l1_buffer(source_id).code
                == native.PartitionControlCode.BUSY
            )
            assert client.close_partition().code == native.PartitionControlCode.BUSY
            assert len(client.partition_status().local_buffers) == 2
            assert not client.partition_status().closing
            records["busy_preserved"] = snapshot(client.partition_status())
        finally:
            os.kill(master.pid, signal.SIGCONT)
        completed(client, future)
        deregistered = success(client.unregister_l1_buffer(source_id))
        assert deregistered.released_shared_bytes == 0
        assert source[:] == b"x" * PRIVATE_BYTES
        new_id = success(
            client.register_l1_buffer(source_base, PRIVATE_BYTES)
        ).local_buffer_id
        assert new_id != source_id
        assert (
            client.unregister_l1_buffer(source_id).code
            == native.PartitionControlCode.INVALID
        )
        success(client.unregister_l1_buffer(new_id))
        success(client.unregister_l1_buffer(destination_id))
        if fail_mount:
            master.terminate()
            master.wait(timeout=10)
            failed = client.allocate_shared_segment(REQUESTED_SHARED_BYTES)
            assert failed.code == native.PartitionControlCode.SDK_ERROR
            assert failed.snapshot.retained_failed_bytes >= REQUESTED_SHARED_BYTES
            assert failed.snapshot.shared_allocated_bytes == 0
            records["failed_mount"] = snapshot(failed.snapshot)
            retained_close = client.close_partition()
            assert retained_close.code == native.PartitionControlCode.SDK_ERROR
            assert not retained_close.snapshot.closed
            assert retained_close.released_shared_bytes == 0
            assert (
                retained_close.snapshot.retained_failed_bytes
                == failed.snapshot.retained_failed_bytes
            )
            records["retained_close"] = snapshot(retained_close.snapshot)
            master = subprocess.Popen(master.args, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 15
            while True:
                assert master.poll() is None, "Recovery master exited"
                try:
                    with socket.create_connection(("127.0.0.1", rpc), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Recovery master did not start") from None
                    time.sleep(0.05)
        closed = success(client.close_partition())
        assert closed.snapshot.closed and closed.snapshot.scratch_bytes == 0
        assert not closed.snapshot.local_buffers
        assert success(client.close_partition()).snapshot.closed
        records["closed"] = snapshot(closed.snapshot)
        records["result"] = "PASS"
        (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            "PASS: private registration, shared grow/free, bytes, "
            "BUSY retry, stale IDs, close"
        )
        print(
            json.dumps(
                {"mapping_removed_each_cycle": [c["mapping_removed"] for c in cycles]}
            )
        )
    finally:
        if master.poll() is None:
            os.kill(master.pid, signal.SIGCONT)
        if client is not None:
            result = client.close_partition()
            if result.code != native.PartitionControlCode.SUCCESS:
                print(
                    "Unresolved ownership retained until process exit:",
                    snapshot(result.snapshot),
                    flush=True,
                )
                master.terminate()
                master.wait(timeout=10)
                os._exit(2)
        source_view.release()
        destination_view.release()
        source.close()
        destination.close()
        if master.poll() is None:
            master.terminate()
        master.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-bin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fail-mount", action="store_true")
    args = parser.parse_args()
    run(args.master_bin.resolve(), args.output, args.fail_mount)
