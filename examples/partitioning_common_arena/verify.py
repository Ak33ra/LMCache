# SPDX-License-Identifier: Apache-2.0
"""Run the bounded native common-arena experiment and print evidence."""

# Standard
from dataclasses import asdict
import argparse
import json
import os
import platform

# Third Party
import torch

# First Party
from examples.partitioning_common_arena.arena import CommonArena, Partition, ResizePhase
from lmcache.v1.distributed.api import ObjectKey
import lmcache.lmcache_native as native


def check(condition: bool, message: str) -> None:
    """Fail the executable check even when Python assertions are disabled."""
    if not condition:
        raise RuntimeError(message)


def report(arena: CommonArena, stage: str) -> None:
    """Print a labeled snapshot with native and logical byte counts."""
    print(json.dumps({"stage": stage, **asdict(arena.snapshot())}, sort_keys=True))


def run(arena_mib: int) -> None:
    """Exercise real copies and blocked resize in a bounded local arena.

    Args:
        arena_mib: Total backing size in MiB, from 1 through 64.

    Raises:
        ValueError: The arena size is outside the prototype bound.
        RuntimeError: Any correctness check fails.
    """
    if not 1 <= arena_mib <= 64:
        raise ValueError("arena-mib must be between 1 and 64")
    total = arena_mib << 20
    arena = CommonArena(total, total // 2)
    private = Partition.PRIVATE
    shared = Partition.SHARED
    object_key = ObjectKey(b"a" * 32, "common-arena-demo", 0)
    payload = b"0123456789abcdef" * (total // 128)
    writer = None
    reader = None
    try:
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "native_extension": native.__file__,
                    "scope": "local native DRAM only; no network or MP integration",
                },
                sort_keys=True,
            )
        )
        baseline = arena.snapshot()
        backing_path = f"/dev/shm/{baseline.shm_name}"
        check(os.stat(backing_path).st_size == total, "backing mapping size mismatch")
        report(arena, "empty")
        writer = arena.reserve_write(shared, object_key, len(payload))
        arena.write(writer, payload)
        check(arena.reserve_read(shared, object_key) is None, "staging is visible")
        report(arena, "shared_staged")
        check(arena.publish(writer), "first shared publication failed")
        check(arena.copy(shared, private, object_key), "private copy failed")
        report(arena, "two_independent_copies")

        reader = arena.reserve_read(private, object_key)
        if reader is None:
            raise RuntimeError("private read missed")
        check(arena.read(reader) == payload, "copy byte mismatch")
        ticket = arena.request_resize(0)
        check(arena.advance_resize(ticket) == ResizePhase.DRAINING, "pin did not block")
        check(arena.snapshot().shared.effective == total // 2, "premature growth")
        report(arena, "resize_blocked_by_private_read")
        arena.finish_read(reader)
        check(arena.advance_resize(ticket) == ResizePhase.COMMITTED, "resize blocked")
        check(
            arena.advance_resize(ticket) == ResizePhase.COMMITTED, "retry changed state"
        )
        check(not arena.finish_read(reader), "old read completion was reused")
        check(not arena.publish(writer), "old write completion was reused")
        report(arena, "all_shared_committed")
        reader = arena.reserve_read(shared, object_key)
        if reader is None:
            raise RuntimeError("shared source lost during private eviction")
        check(arena.read(reader) == payload, "shared source changed")
        arena.finish_read(reader)

        reverse = arena.request_resize(total)
        check(arena.advance_resize(reverse) == ResizePhase.COMMITTED, "reverse failed")
        check(
            arena.advance_resize(ticket) == ResizePhase.STALE, "stale resize accepted"
        )
        writer = arena.reserve_write(private, object_key, len(payload))
        arena.write(writer, payload)
        check(arena.publish(writer), "new private allocation failed")
        report(arena, "all_private_committed")
        check(
            arena.snapshot().backing_address == baseline.backing_address, "arena moved"
        )
        check(arena.delete(private, object_key), "final allocation was not released")
        report(arena, "all_object_charges_released")
        check(arena.snapshot().native_used == 0, "native allocation leaked")
    finally:
        if reader is not None:
            arena.finish_read(reader)
        if writer is not None:
            arena.discard(writer)
        arena.close()
    check(not os.path.exists(backing_path), "backing segment was not unlinked")
    print("PASS: native bytes, independent copies, drain, retry, stale handles")


def main() -> None:
    """Parse the bounded arena size and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-mib", type=int, default=8)
    args = parser.parse_args()
    run(args.arena_mib)


if __name__ == "__main__":
    main()
