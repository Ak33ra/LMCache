# SPDX-License-Identifier: Apache-2.0
"""Real CPU/native-pool checks; run with pytest --noconftest."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event

# Third Party
import pytest

# First Party
from examples.partitioning_common_arena.arena import (
    ALIGNMENT,
    MAX_OPERATIONS,
    CommonArena,
    Partition,
    ResizePhase,
)
from lmcache.v1.distributed.api import ObjectKey

PRIVATE = Partition.PRIVATE
SHARED = Partition.SHARED
PAGE = ALIGNMENT


def key(value: int = 1) -> ObjectKey:
    """Build a real namespaced LMCache key for a small test object."""
    return ObjectKey(value.to_bytes(32, "big"), "arena-test", 0)


def store(
    arena: CommonArena, partition: Partition, value: bytes, index: int = 1
) -> None:
    """Write and publish a complete object through the public interface."""
    writer = arena.reserve_write(partition, key(index), len(value))
    arena.write(writer, value)
    assert arena.publish(writer)


def load(arena: CommonArena, partition: Partition, index: int = 1) -> bytes:
    """Copy a resident object's bytes and release its read operation."""
    reader = arena.reserve_read(partition, key(index))
    assert reader is not None
    try:
        return arena.read(reader)
    finally:
        arena.finish_read(reader)


@pytest.fixture
def arena() -> Iterator[CommonArena]:
    """Use a 32 KiB native arena with equal private/shared quotas."""
    instance = CommonArena(8 * PAGE, 4 * PAGE)
    yield instance
    instance.close()


def test_staging_alignment_and_publication(arena: CommonArena) -> None:
    writer = arena.reserve_write(PRIVATE, key(), 37)
    snapshot = arena.snapshot()
    assert snapshot.native_used == snapshot.private.used == PAGE
    assert snapshot.private.staged == snapshot.private.pinned == PAGE
    assert arena.reserve_read(PRIVATE, key()) is None
    with pytest.raises(ValueError, match="incomplete"):
        arena.publish(writer)
    with pytest.raises(ValueError, match="length"):
        arena.write(writer, b"short")
    arena.write(writer, b"x" * 37)
    assert arena.reserve_read(PRIVATE, key()) is None
    assert arena.publish(writer)
    assert load(arena, PRIVATE) == b"x" * 37
    snapshot = arena.snapshot()
    assert snapshot.private.staged == snapshot.private.pinned == 0
    assert snapshot.private.residents == 1


@pytest.mark.parametrize("source,destination", [(SHARED, PRIVATE), (PRIVATE, SHARED)])
def test_independent_native_copies(
    arena: CommonArena, source: Partition, destination: Partition
) -> None:
    store(arena, source, b"original" * 257)
    assert arena.copy(source, destination, key())
    assert arena.snapshot().native_used == 2 * PAGE
    assert arena.delete(source, key())
    store(arena, source, b"replaced" * 257)
    assert load(arena, destination) == b"original" * 257
    assert load(arena, source) == b"replaced" * 257
    assert arena.delete(destination, key())
    assert load(arena, source) == b"replaced" * 257


def test_duplicate_staging_loser_releases_exact_charge(arena: CommonArena) -> None:
    first = arena.reserve_write(SHARED, key(), 10)
    second = arena.reserve_write(SHARED, key(), 10)
    arena.write(first, b"first copy")
    arena.write(second, b"other copy")
    assert arena.snapshot().shared.staged == 2 * PAGE
    assert arena.publish(first)
    assert not arena.publish(second)
    assert not arena.discard(second)
    assert arena.snapshot().native_used == PAGE
    assert load(arena, SHARED) == b"first copy"


def test_stale_completions_do_not_touch_recreated_key(arena: CommonArena) -> None:
    old_writer = arena.reserve_write(PRIVATE, key(), 3)
    arena.write(old_writer, b"old")
    assert arena.publish(old_writer)
    old_reader = arena.reserve_read(PRIVATE, key())
    assert old_reader is not None
    assert arena.finish_read(old_reader)
    assert arena.delete(PRIVATE, key())
    new_writer = arena.reserve_write(PRIVATE, key(), 3)
    arena.write(new_writer, b"new")
    assert new_writer.allocation != old_writer.allocation
    assert not arena.publish(old_writer)
    assert not arena.discard(old_writer)
    assert arena.publish(new_writer)
    new_reader = arena.reserve_read(PRIVATE, key())
    assert new_reader is not None
    assert not arena.finish_read(old_reader)
    assert not arena.delete(PRIVATE, key())
    assert arena.read(new_reader) == b"new"
    assert arena.finish_read(new_reader)
    assert arena.snapshot().private.pinned == 0


def test_failed_copy_and_full_destination_release_source(arena: CommonArena) -> None:
    def failing_copy(source: memoryview, destination: memoryview) -> None:
        destination[:1] = source[:1]
        raise OSError("controlled copy failure")

    store(arena, SHARED, b"source")
    with pytest.raises(OSError, match="controlled copy failure"):
        arena.copy(SHARED, PRIVATE, key(), copier=failing_copy)
    assert arena.reserve_read(PRIVATE, key()) is None
    snapshot = arena.snapshot()
    assert snapshot.native_used == PAGE
    assert snapshot.active_operations == snapshot.private.used == 0
    store(arena, PRIVATE, b"p" * (4 * PAGE), index=2)
    with pytest.raises(MemoryError, match="admission"):
        arena.copy(SHARED, PRIVATE, key())
    assert arena.snapshot().shared.pinned == 0
    assert load(arena, SHARED) == b"source"


@pytest.mark.parametrize("cancel_copy", [False, True])
def test_copy_barrier_pins_until_real_completion(
    arena: CommonArena, cancel_copy: bool
) -> None:
    started = Event()
    stopped = Event()
    cancel = Event()

    def held_copy(source: memoryview, destination: memoryview) -> None:
        destination[:3] = source[:3]
        started.set()
        if not stopped.wait(10):
            raise TimeoutError("test did not release copier")
        destination[3:] = source[3:]

    store(arena, SHARED, b"native copy bytes")
    initial = arena.snapshot()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            arena.copy, SHARED, PRIVATE, key(), copier=held_copy, cancel=cancel
        )
        try:
            assert started.wait(10)
            assert arena.reserve_read(PRIVATE, key()) is None
            assert not arena.delete(SHARED, key())
            with pytest.raises(RuntimeError, match="active operations"):
                arena.close()
            ticket = arena.request_resize(8 * PAGE)
            assert arena.advance_resize(ticket) == ResizePhase.DRAINING
            if cancel_copy:
                cancel.set()
            snapshot = arena.snapshot()
            assert snapshot.native_used == 2 * PAGE
            assert snapshot.private.staged == snapshot.private.pinned == PAGE
            assert snapshot.shared.pinned == PAGE
            assert snapshot.private.effective == initial.private.effective
            assert snapshot.shared.effective == initial.shared.effective
            assert arena.advance_resize(ticket) == ResizePhase.DRAINING
        finally:
            stopped.set()
        assert future.result(timeout=10) is not cancel_copy
    assert arena.advance_resize(ticket) == ResizePhase.COMMITTED
    assert arena.snapshot().backing_address == initial.backing_address
    assert arena.snapshot().shared.used == 0
    assert arena.snapshot().private.effective == 8 * PAGE
    if cancel_copy:
        assert arena.reserve_read(PRIVATE, key()) is None
        assert arena.snapshot().native_used == 0
    else:
        assert load(arena, PRIVATE) == b"native copy bytes"


def test_resize_does_not_grant_capacity_until_pinned_bytes_drain(
    arena: CommonArena,
) -> None:
    store(arena, PRIVATE, b"pinned", index=1)
    store(arena, PRIVATE, b"evictable", index=2)
    store(arena, SHARED, b"s" * (4 * PAGE))
    reader = arena.reserve_read(PRIVATE, key())
    assert reader is not None
    ticket = arena.request_resize(0)
    assert arena.request_resize(0) == ticket
    assert arena.advance_resize(ticket) == ResizePhase.DRAINING
    snapshot = arena.snapshot()
    assert snapshot.private.used == snapshot.private.pinned == PAGE
    assert snapshot.private.effective == snapshot.shared.effective == 4 * PAGE
    assert snapshot.private.admission_limit == 0
    with pytest.raises(MemoryError, match="admission"):
        arena.reserve_write(SHARED, key(2), 1)
    with pytest.raises(RuntimeError, match="still draining"):
        arena.request_resize(PAGE)
    assert arena.read(reader) == b"pinned"
    assert arena.finish_read(reader)
    assert arena.advance_resize(ticket) == ResizePhase.COMMITTED
    assert arena.advance_resize(ticket) == ResizePhase.COMMITTED
    store(arena, SHARED, b"new capacity", index=2)
    assert arena.snapshot().shared.effective == 8 * PAGE
    reverse = arena.request_resize(8 * PAGE)
    assert arena.advance_resize(reverse) == ResizePhase.COMMITTED
    assert arena.advance_resize(ticket) == ResizePhase.STALE
    assert arena.snapshot().shared.effective == 0
    store(arena, PRIVATE, b"p" * (8 * PAGE))
    assert arena.snapshot().native_used == 8 * PAGE


def test_staged_writer_blocks_resize_until_discard(arena: CommonArena) -> None:
    writer = arena.reserve_write(PRIVATE, key(), 1)
    ticket = arena.request_resize(0)
    assert arena.advance_resize(ticket) == ResizePhase.DRAINING
    with pytest.raises(MemoryError, match="admission"):
        arena.reserve_write(PRIVATE, key(2), 1)
    assert arena.discard(writer)
    assert not arena.discard(writer)
    assert arena.advance_resize(ticket) == ResizePhase.COMMITTED
    assert arena.snapshot().native_used == 0


def test_resize_lru_and_repeated_cycles(arena: CommonArena) -> None:
    store(arena, PRIVATE, b"one", index=1)
    store(arena, PRIVATE, b"two", index=2)
    assert load(arena, PRIVATE, index=1) == b"one"
    ticket = arena.request_resize(PAGE)
    assert arena.advance_resize(ticket) == ResizePhase.COMMITTED
    assert arena.reserve_read(PRIVATE, key(2)) is None
    assert load(arena, PRIVATE) == b"one"
    address = arena.snapshot().backing_address
    for _ in range(10):
        for private_bytes, destination in [(0, SHARED), (8 * PAGE, PRIVATE)]:
            ticket = arena.request_resize(private_bytes)
            assert arena.advance_resize(ticket) == ResizePhase.COMMITTED
            assert arena.snapshot().native_used == 0
            store(arena, destination, b"x" * (8 * PAGE))
            snapshot = arena.snapshot()
            assert snapshot.native_used == snapshot.backing_bytes == 8 * PAGE
            assert snapshot.backing_address == address


def test_operation_bound_and_close(arena: CommonArena) -> None:
    store(arena, PRIVATE, b"held")
    readers = [arena.reserve_read(PRIVATE, key()) for _ in range(MAX_OPERATIONS)]
    try:
        with pytest.raises(MemoryError, match="operation limit"):
            arena.reserve_read(PRIVATE, key())
        with pytest.raises(MemoryError, match="operation limit"):
            arena.reserve_write(SHARED, key(), 1)
        assert arena.snapshot().private.pinned == PAGE
        with pytest.raises(RuntimeError, match="active operations"):
            arena.close()
    finally:
        for reader in readers:
            assert reader is not None
            arena.finish_read(reader)
    arena.close()
    arena.close()
    with pytest.raises(RuntimeError, match="closed"):
        arena.reserve_read(PRIVATE, key())


def test_foreign_arena_handles_cannot_finish_local_operations(
    arena: CommonArena,
) -> None:
    other = CommonArena(8 * PAGE, 4 * PAGE)
    local_writer = arena.reserve_write(PRIVATE, key(), 3)
    foreign_writer = other.reserve_write(PRIVATE, key(), 3)
    try:
        arena.write(local_writer, b"own")
        assert not arena.publish(foreign_writer)
        assert not arena.discard(foreign_writer)
        assert arena.snapshot().private.staged == PAGE
        assert arena.publish(local_writer)
        assert load(arena, PRIVATE) == b"own"
        local_resize = arena.request_resize(0)
        foreign_resize = other.request_resize(0)
        assert arena.advance_resize(foreign_resize) == ResizePhase.STALE
        assert arena.snapshot().private.effective == 4 * PAGE
        assert arena.advance_resize(local_resize) == ResizePhase.COMMITTED
    finally:
        arena.discard(local_writer)
        other.discard(foreign_writer)
        other.close()


def test_native_fragmentation_failure_does_not_charge_or_pin() -> None:
    instance = CommonArena(8 * PAGE, 8 * PAGE)
    try:
        for index in range(8):
            store(instance, PRIVATE, b"x" * PAGE, index=index)
        for index in range(0, 8, 2):
            assert instance.delete(PRIVATE, key(index))
        before = instance.snapshot()
        with pytest.raises(MemoryError, match="fragmented"):
            instance.reserve_write(PRIVATE, key(10), 3 * PAGE)
        assert instance.snapshot() == before
    finally:
        instance.close()


@pytest.mark.parametrize("private_bytes", [0, 8 * PAGE])
def test_zero_and_full_partition(private_bytes: int) -> None:
    instance = CommonArena(8 * PAGE, private_bytes)
    full = SHARED if private_bytes == 0 else PRIVATE
    empty = PRIVATE if private_bytes == 0 else SHARED
    try:
        with pytest.raises(MemoryError, match="admission"):
            instance.reserve_write(empty, key(), 1)
        store(instance, full, b"x" * (8 * PAGE))
        assert instance.snapshot().native_used == 8 * PAGE
    finally:
        instance.close()


@pytest.mark.parametrize(
    "total,private", [(0, 0), (65 << 20, 0), (PAGE + 1, 0), (PAGE, 1), (PAGE, -1)]
)
def test_invalid_budgets(total: int, private: int) -> None:
    with pytest.raises(ValueError):
        CommonArena(total, private)
