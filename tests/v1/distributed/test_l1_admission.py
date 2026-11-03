# SPDX-License-Identifier: Apache-2.0
"""Bounded L1 admission with real CPU mappings and native reservation locks."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1AdmissionLimits,
    L1AdmissionState,
    L1AdmissionUsage,
    L1WriteOperation,
    L1WriteOperationResult,
    L1WriteRetention,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    SegmentedDRAML1MemoryManager,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.memory_management import MemoryObj

PAGE = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([PAGE])], [torch.uint8])
RETAIN = L1WriteRetention.RETAIN
TEMPORARY = L1WriteRetention.TEMPORARY
PREFER = L1WriteRetention.PREFER_RETAIN
pytestmark = pytest.mark.no_shared_allocator


def key(index: int) -> ObjectKey:
    """Create a distinct object in the admission test namespace."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "admission", 0)


@contextmanager
def manager_with_limits(
    retained_pages: int,
    temporary_pages: int,
    *,
    mapped_pages: int = 16,
    read_ttl_seconds: int = 300,
) -> Iterator[L1Manager]:
    """Own a tiny real segmented pool and its bounded L1 for one test."""
    pool = SegmentedDRAML1MemoryManager(align_bytes=PAGE)
    pool.add_segment(mapped_pages * PAGE)
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                size_in_bytes=mapped_pages * PAGE,
                use_lazy=False,
                shm_name="",
                align_bytes=PAGE,
            ),
            read_ttl_seconds=read_ttl_seconds,
            write_ttl_seconds=1,
        ),
        memory_manager=pool,
        admission_limits=L1AdmissionLimits(
            retained_pages * PAGE, temporary_pages * PAGE
        ),
    )
    try:
        yield manager
    finally:
        manager.close()


def reserve(
    manager: L1Manager,
    index: int,
    retention: L1WriteRetention = RETAIN,
    *,
    tag: str = "",
) -> L1WriteOperation:
    """Require one successful original allocation capability."""
    error, operation = manager.reserve_write_operations(
        [key(index)], [retention], LAYOUT, tag=tag
    )[key(index)]
    assert error is L1Error.SUCCESS and operation is not None
    return operation


def test_prefer_retained_then_bounded_temporary_in_input_order() -> None:
    with manager_with_limits(2, 1) as manager:
        keys = [key(i) for i in range(4)]
        results = manager.reserve_write_operations(keys, [PREFER] * 4, LAYOUT)
        operations = [operation for _, operation in results.values() if operation]
        assert len(operations) == 3
        assert results[keys[-1]] == (L1Error.OUT_OF_MEMORY, None)
        assert manager.get_admission_usage() == L1AdmissionUsage(
            2 * PAGE, PAGE, 2 * PAGE, PAGE
        )
        assert manager.get_memory_usage()[0] == 3 * PAGE
        assert manager.reserve_read(keys)[keys[0]][0] is L1Error.KEY_NOT_EXIST
        for index, operation in enumerate(operations):
            operation.memory_obj.get_tensor(0).fill_(index + 1)
        manager.finish_write_operations(operations)
        for index, object_key in enumerate(keys[:3]):
            obj = manager.unsafe_read([object_key])[object_key][1]
            assert obj is not None
            assert torch.all(obj.get_tensor(0) == index + 1)
        manager.finish_read(keys[:3])
        assert manager.get_admission_usage().retained_allocated_bytes == 2 * PAGE
        assert manager.get_admission_usage().temporary_allocated_bytes == 0
        assert manager.reserve_read([keys[2]])[keys[2]][0] is L1Error.KEY_NOT_EXIST


def test_zero_private_refuses_retain_and_uses_only_temporary_quota() -> None:
    with manager_with_limits(0, 2) as manager:
        keys = [key(i) for i in range(4)]
        results = manager.reserve_write_operations(
            keys, [RETAIN, TEMPORARY, PREFER, PREFER], LAYOUT
        )
        assert results[keys[0]] == results[keys[3]] == (L1Error.OUT_OF_MEMORY, None)
        operations = [operation for _, operation in results.values() if operation]
        assert manager.get_admission_usage() == L1AdmissionUsage(
            0, 2 * PAGE, 0, 2 * PAGE
        )
        assert manager.get_eviction_pressure() == (0, 0)
        manager.finish_write_operations(operations, discard=frozenset(operations))
        assert manager.get_memory_usage()[0] == 0


def test_alignment_is_charged_and_physical_exhaustion_does_not_leak() -> None:
    layout = MemoryLayoutDesc([torch.Size([PAGE + 1])], [torch.uint8])
    with manager_with_limits(3, 2, mapped_pages=3) as manager:
        keys = [key(1), key(2)]
        results = manager.reserve_write_operations(keys, [RETAIN, RETAIN], layout)
        operation = results[keys[0]][1]
        assert operation is not None
        assert operation.memory_obj.get_size() == PAGE + 1
        assert operation.memory_obj.get_physical_size() == 2 * PAGE
        assert results[keys[1]] == (L1Error.OUT_OF_MEMORY, None)
        assert manager.get_admission_usage().retained_allocated_bytes == 2 * PAGE
        assert manager.get_memory_usage()[0] == 2 * PAGE
        manager.finish_write_operations([operation], discard=frozenset([operation]))
        assert manager.get_memory_usage()[0] == 0
        assert manager.memcheck()


def test_aligned_quota_refusal_with_free_physical_space() -> None:
    layout = MemoryLayoutDesc([torch.Size([1])], [torch.uint8])
    with manager_with_limits(1, 0) as manager:
        manager.set_admission_limits(L1AdmissionLimits(PAGE - 1, 0))
        assert manager.reserve_write_operations([key(1)], [RETAIN], layout) == {
            key(1): (L1Error.OUT_OF_MEMORY, None)
        }
        assert manager.get_memory_usage()[0] == 0
        assert manager.get_admission_usage().retained_allocated_bytes == 0
        assert manager.memcheck()


def test_concurrent_writers_cannot_overbook_either_quota() -> None:
    with manager_with_limits(3, 2) as manager:
        gate = threading.Barrier(12)

        def write(index: int) -> L1WriteOperationResult:
            gate.wait(timeout=5)
            return manager.reserve_write_operations([key(index)], [PREFER], LAYOUT)[
                key(index)
            ]

        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(write, range(12)))
        operations = [operation for _, operation in results if operation]
        assert len(operations) == 5
        assert manager.get_admission_usage() == L1AdmissionUsage(
            3 * PAGE, 2 * PAGE, 3 * PAGE, 2 * PAGE
        )
        assert manager.get_memory_usage()[0] == 5 * PAGE
        manager.finish_write_operations(operations, discard=frozenset(operations))
        assert manager.get_memory_usage()[0] == 0


def test_allocator_exception_rolls_back_unreturned_reservations() -> None:
    class FailingPool(SegmentedDRAML1MemoryManager):
        def __init__(self) -> None:
            super().__init__(align_bytes=PAGE)
            self.calls = 0

        def allocate(
            self, layout_desc: MemoryLayoutDesc, count: int
        ) -> tuple[L1Error, list[MemoryObj]]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("allocation interrupted")
            return super().allocate(layout_desc, count)

    pool = FailingPool()
    pool.add_segment(2 * PAGE)
    manager = L1Manager(
        L1ManagerConfig(L1MemoryManagerConfig(2 * PAGE, False, shm_name="")),
        memory_manager=pool,
        admission_limits=L1AdmissionLimits(2 * PAGE, 0),
    )
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            manager.reserve_write_operations([key(1), key(2)], [RETAIN] * 2, LAYOUT)
        assert manager.get_memory_usage()[0] == 0
        assert manager.get_admission_usage().retained_allocated_bytes == 0
        assert manager.report_status()["active_write_operation_count"] == 0
        operation = reserve(manager, 1)
        manager.finish_write_operations([operation], discard=frozenset([operation]))
    finally:
        manager.close()


@pytest.mark.parametrize("winner_retention", [RETAIN, TEMPORARY])
def test_same_key_loser_releases_its_own_class(
    winner_retention: L1WriteRetention,
) -> None:
    with manager_with_limits(1, 1) as manager:
        loser_retention = TEMPORARY if winner_retention is RETAIN else RETAIN
        winner = reserve(manager, 1, winner_retention, tag="winner")
        loser = reserve(manager, 1, loser_retention, tag="loser")
        for operation, value in ((winner, 17), (loser, 91)):
            tensor = operation.memory_obj.get_tensor(0)
            assert tensor is not None
            tensor.fill_(value)
            del tensor
        assert manager.get_memory_usage()[0] == 2 * PAGE
        manager.finish_write_operations([winner])
        assert manager.get_memory_usage()[0] == 2 * PAGE
        result = manager.finish_write_operations([loser])
        assert result[loser][1] is winner.memory_obj
        assert torch.all(winner.memory_obj.get_tensor(0) == 17)
        usage = manager.get_admission_usage()
        assert usage.retained_allocated_bytes == (
            PAGE if winner_retention is RETAIN else 0
        )
        assert usage.temporary_allocated_bytes == (
            PAGE if winner_retention is TEMPORARY else 0
        )
        manager.finish_read([key(1)], read_locks=2)
        manager.clear()
        assert manager.get_memory_usage()[0] == 0


def test_discard_remains_charged_until_exact_and_legacy_readers_finish() -> None:
    with manager_with_limits(1, 1) as manager:
        writer = reserve(manager, 1)
        manager.finish_write_operations([writer])
        first = manager.acquire_read_operations([key(1)])[key(1)][1]
        second = manager.acquire_read_operations([key(1)])[key(1)][1]
        assert first is not None and second is not None
        manager.finish_read_operations([first], discard=frozenset([first]))
        manager.finish_read_operations([second])
        assert manager.reclaim_discarded_read_operations() == 1
        assert manager.get_admission_usage().retained_allocated_bytes == PAGE
        assert manager.reserve_write_operations([key(2)], [RETAIN], LAYOUT)[key(2)] == (
            L1Error.OUT_OF_MEMORY,
            None,
        )
        manager.finish_read([key(1)])
        assert manager.get_admission_usage().retained_allocated_bytes == 0
        replacement = reserve(manager, 1)
        with pytest.raises(ValueError, match="not active"):
            manager.finish_write_operations([writer])
        assert manager.get_admission_usage().retained_allocated_bytes == PAGE
        manager.finish_write_operations([replacement], discard=frozenset([replacement]))


def test_borrowed_refcount_cleanup_cannot_free_owned_allocation() -> None:
    with manager_with_limits(1, 0) as manager:
        writer = reserve(manager, 1)
        tensor = writer.memory_obj.get_tensor(0)
        assert tensor is not None
        tensor.fill_(73)
        del tensor
        writer.memory_obj.ref_count_down()
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == PAGE
        assert torch.all(writer.memory_obj.get_tensor(0) == 73)
        manager.finish_write_operations([writer], read_locks=0)
        reader = manager.acquire_read_operations([key(1)])[key(1)][1]
        assert reader is not None
        manager.clear(force=True)
        assert manager.get_admission_usage().retained_allocated_bytes == PAGE
        assert torch.all(reader.memory_obj.get_tensor(0) == 73)
        manager.finish_read_operations([reader], discard=frozenset([reader]))
        assert manager.get_memory_usage()[0] == 0


def test_quiesce_refuses_new_ownership_but_allows_held_completion() -> None:
    with manager_with_limits(2, 1) as manager:
        first = reserve(manager, 1)
        manager.finish_write_operations([first], read_locks=0)
        reader = manager.acquire_read_operations([key(1)])[key(1)][1]
        assert reader is not None
        writer = reserve(manager, 2)
        manager.set_admission_state(L1AdmissionState.QUIESCING)
        assert manager.get_admission_state() is L1AdmissionState.QUIESCING
        assert manager.reserve_write_operations([key(3)], [PREFER], LAYOUT)[key(3)] == (
            L1Error.KEY_NOT_WRITABLE,
            None,
        )
        assert manager.reserve_read([key(1)])[key(1)] == (
            L1Error.KEY_NOT_READABLE,
            None,
        )
        assert manager.acquire_read_operations([key(1)])[key(1)] == (
            L1Error.KEY_NOT_READABLE,
            None,
        )
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 2 * PAGE
        manager.finish_write_operations([writer])
        manager.finish_read([key(2)])
        manager.finish_read_operations([reader], discard=frozenset([reader]))
        manager.clear()
        manager.set_admission_limits(L1AdmissionLimits(0, PAGE))
        manager.set_admission_state(L1AdmissionState.OPEN)
        temporary = reserve(manager, 3, PREFER)
        assert manager.get_admission_usage().temporary_allocated_bytes == PAGE
        manager.finish_write_operations([temporary], discard=frozenset([temporary]))


def test_lowering_limits_preserves_live_charges_without_reclassification() -> None:
    with manager_with_limits(2, 1) as manager:
        writer = reserve(manager, 1)
        manager.set_admission_limits(L1AdmissionLimits(0, PAGE))
        assert manager.get_admission_usage() == L1AdmissionUsage(PAGE, 0, 0, PAGE)
        fallback = reserve(manager, 2, PREFER)
        assert manager.get_eviction_pressure() == (PAGE, 0)
        manager.finish_write_operations([writer, fallback], read_locks=0)
        manager.clear()
        assert manager.get_memory_usage()[0] == 0


@pytest.mark.parametrize("retained_limit", [PAGE, 0])
def test_lru_reclaims_full_private_below_physical_pool_watermark(
    retained_limit: int,
) -> None:
    with manager_with_limits(1, 1) as manager:
        controller = L1EvictionController(
            manager, EvictionConfig("LRU", trigger_watermark=0.8, eviction_ratio=1.0)
        )
        retained = reserve(manager, 1)
        temporary = reserve(manager, 2, TEMPORARY)
        manager.finish_write_operations([retained], read_locks=0)
        manager.finish_write_operations([temporary])
        assert manager.get_memory_usage() == (2 * PAGE, 16 * PAGE)
        manager.set_admission_limits(L1AdmissionLimits(retained_limit, PAGE))
        assert manager.get_eviction_pressure() == (PAGE, retained_limit)
        assert not manager.is_key_evictable(key(2))
        controller.start()
        try:
            deadline = time.monotonic() + 5
            while manager.get_admission_usage().retained_allocated_bytes:
                assert time.monotonic() < deadline
                time.sleep(0.02)
            assert manager.get_admission_usage().temporary_allocated_bytes == PAGE
        finally:
            controller.stop()
        manager.finish_read([key(2)])


def test_zero_private_sweeps_expired_temporary_but_never_active_operations() -> None:
    with manager_with_limits(0, 3, read_ttl_seconds=1) as manager:
        abandoned = reserve(manager, 1, TEMPORARY)
        active = reserve(manager, 2, TEMPORARY)
        writer = reserve(manager, 3, TEMPORARY)
        manager.finish_write_operations([abandoned, active])
        reader = manager.acquire_read_operations([key(2)])[key(2)][1]
        assert reader is not None
        controller = L1EvictionController(manager, EvictionConfig("noop"))
        controller.start()
        try:
            deadline = time.monotonic() + 5
            while manager.get_admission_usage().temporary_allocated_bytes > 2 * PAGE:
                assert time.monotonic() < deadline
                time.sleep(0.02)
            assert manager.get_memory_usage()[0] == 2 * PAGE
            assert manager.reclaim_temporary_objects() == 0
            manager.clear(force=True)
            assert manager.get_memory_usage()[0] == 2 * PAGE
        finally:
            controller.stop()
        manager.finish_read_operations([reader])
        manager.finish_write_operations([writer], discard=frozenset([writer]))
        assert manager.get_memory_usage()[0] == 0


def test_disabled_batch_atomicity_and_empty_only_first_activation() -> None:
    pool = SegmentedDRAML1MemoryManager(align_bytes=PAGE)
    pool.add_segment(PAGE)
    manager = L1Manager(
        L1ManagerConfig(L1MemoryManagerConfig(PAGE, False, shm_name="")),
        memory_manager=pool,
    )
    try:
        keys = [key(1), key(2)]
        assert manager.get_admission_limits() is None
        assert manager.get_admission_usage() == L1AdmissionUsage(0, 0, None, None)
        assert manager.reserve_write_operations(keys, [PREFER] * 2, LAYOUT) == {
            object_key: (L1Error.OUT_OF_MEMORY, None) for object_key in keys
        }
        assert manager.get_memory_usage()[0] == 0
        operation = reserve(manager, 1, PREFER)
        with pytest.raises(ValueError, match="empty L1"):
            manager.set_admission_limits(L1AdmissionLimits(PAGE, 0))
        manager.finish_write_operations([operation], discard=frozenset([operation]))
        manager.set_admission_limits(L1AdmissionLimits(PAGE, 0))
        assert manager.get_admission_limits() == L1AdmissionLimits(PAGE, 0)
        with pytest.raises(RuntimeError, match="reserve_write_operations"):
            manager.reserve_write([key(1)], [False], LAYOUT)
        with pytest.raises(ValueError, match="nonnegative"):
            L1AdmissionLimits(-1, PAGE)
    finally:
        manager.close()
