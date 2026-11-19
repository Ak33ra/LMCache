# SPDX-License-Identifier: Apache-2.0
"""Real segmented mappings with controlled capability; not live SDK evidence."""

# Standard
from dataclasses import replace
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    SegmentedDRAML1MemoryManager,
)
from lmcache.v1.distributed.partition_config import PartitionConfig
from lmcache.v1.distributed.partition_controls import CapacityControlCode as Code
from lmcache.v1.distributed.partition_controls import CapacityControlResult as Result
from lmcache.v1.distributed.partition_controls import (
    LocalRegistration,
    LocalRegistrationState,
    SharedAllocation,
    SharedCapacitySnapshot,
)
from lmcache.v1.distributed.partition_manager import PartitionManager, ResizePhase

PAGE = 4096
pytestmark = pytest.mark.no_shared_allocator


class Controls:
    """Deterministic failure/blocked call injection with exact identity snapshots."""

    def __init__(self) -> None:
        self.state = SharedCapacitySnapshot(
            True, False, False, PAGE, PAGE, 0, 0, 0, (), ()
        )
        self.sequence = 0
        self.fail_release = False
        self.fail_close = False
        self.fail_register = False
        self.busy_register = False
        self.busy_allocate = False
        self.registration_attempts: list[tuple[int, int]] = []
        self.busy_attempts = 0
        self.repeated_busy = threading.Event()
        self.block: threading.Event | None = None
        self.entered = threading.Event()
        self.unregistered: list[int] = []

    def snapshot(self) -> SharedCapacitySnapshot:
        return self.state

    def register_local(self, base: int, size: int) -> Result:
        self.registration_attempts.append((base, size))
        if self.busy_register:
            self.busy_attempts += 1
            if self.busy_attempts >= 3:
                self.repeated_busy.set()
            return Result(Code.BUSY, 0, self.state)
        assert all(item.base != base for item in self.state.local_buffers)
        self.sequence += 1
        item = LocalRegistration(
            self.sequence, base, size, LocalRegistrationState.REGISTERED
        )
        self.state = replace(
            self.state, local_buffers=(*self.state.local_buffers, item)
        )
        return Result(
            Code.FAILED if self.fail_register else Code.SUCCESS,
            17 if self.fail_register else 0,
            self.state,
            item.registration_id,
        )

    def unregister_local(self, registration_id: int) -> Result:
        assert registration_id not in self.unregistered
        self.unregistered.append(registration_id)
        self.state = replace(
            self.state,
            local_buffers=tuple(
                x
                for x in self.state.local_buffers
                if x.registration_id != registration_id
            ),
        )
        return Result(Code.SUCCESS, 0, self.state)

    def allocate_shared(self, size: int) -> Result:
        if self.busy_allocate:
            self.busy_attempts += 1
            if self.busy_attempts >= 3:
                self.repeated_busy.set()
            return Result(Code.BUSY, 0, self.state)
        if self.block is not None:
            self.entered.set()
            assert self.block.wait(5)
        self.sequence += 1
        item = SharedAllocation(str(self.sequence), self.sequence * PAGE, size)
        self.state = replace(
            self.state,
            shared_segments=(*self.state.shared_segments, item),
            shared_allocated_bytes=self.state.shared_allocated_bytes + size,
        )
        return Result(Code.SUCCESS, 0, self.state, segment_ids=(item.segment_id,))

    def release_shared(self, segment_id: str) -> Result:
        if self.fail_release:
            return Result(Code.FAILED, 31, self.state)
        segment = next(
            x for x in self.state.shared_segments if x.segment_id == segment_id
        )
        self.state = replace(
            self.state,
            shared_segments=tuple(
                x for x in self.state.shared_segments if x.segment_id != segment_id
            ),
            shared_allocated_bytes=self.state.shared_allocated_bytes - segment.size,
        )
        return Result(Code.SUCCESS, 0, self.state, released_shared_bytes=segment.size)

    def close(self) -> Result:
        if self.fail_close:
            return Result(Code.FAILED, 31, self.state)
        self.state = replace(
            self.state,
            closed=True,
            local_buffers=(),
            shared_segments=(),
            shared_allocated_bytes=0,
        )
        return Result(Code.SUCCESS, 0, self.state)


def setup(
    alpha: float = 0.5,
    allocation_unit_bytes: int | None = None,
) -> tuple[PartitionManager, L1Manager, SegmentedDRAML1MemoryManager, Controls]:
    pool = SegmentedDRAML1MemoryManager(align_bytes=PAGE)
    pool.add_segment(PAGE)
    l1 = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                size_in_bytes=PAGE, use_lazy=False, shm_name="", align_bytes=PAGE
            )
        ),
        memory_manager=pool,
    )
    controls = Controls()
    manager = PartitionManager(
        PartitionConfig(6 * PAGE, PAGE, PAGE, alpha, allocation_unit_bytes),
        pool,
        l1,
        controls,
        quiesce=lambda: True,
    )
    return manager, l1, pool, controls


def wait(
    manager: PartitionManager,
    request_id: str,
    phase: ResizePhase = ResizePhase.COMPLETE,
) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = manager.get_status()
        assert status.charged_bytes <= 6 * PAGE
        if manager.resize_status(request_id).phase is phase:
            return
        time.sleep(0.01)
    raise AssertionError(manager.get_status())


def test_cycles_rounding_and_idempotency() -> None:
    manager, l1, pool, _ = setup()
    try:
        state = manager.initialize()
        assert state.current.private_effective_bytes == 2 * PAGE
        assert state.mapped_bytes == 3 * PAGE
        for index, alpha in enumerate((0.0, 1.0, 0.3, 0.5)):
            request_id = str(index)
            manager.request_resize(request_id, alpha)
            wait(manager, request_id)
            result = manager.resize_status(request_id)
            assert manager.request_resize(request_id, alpha) == result
            assert (
                manager.get_status().mapped_bytes
                == PAGE + result.target.private_effective_bytes
            )
            with pytest.raises(RuntimeError, match="another target"):
                manager.request_resize(request_id, 0.7)
        assert manager.close()
        assert pool.get_memory_usage() == (0, 0)
    finally:
        assert manager.close()
        l1.close()


def test_larger_units_release_backing_before_reassigning_it() -> None:
    """Larger private and shared segments keep the same physical envelope."""
    manager, l1, pool, controls = setup(allocation_unit_bytes=2 * PAGE)
    try:
        manager.initialize()
        controls.fail_release = True
        manager.request_resize("blocked-large-unit", 1.0)
        wait(manager, "blocked-large-unit", ResizePhase.FAILED)
        assert manager.get_status().mapped_bytes == 3 * PAGE
        assert controls.snapshot().shared_allocated_bytes == 2 * PAGE
        controls.fail_release = False
        for index, alpha in enumerate((0.0, 1.0, 0.5)):
            request_id = f"large-unit-{index}"
            manager.request_resize(request_id, alpha)
            wait(manager, request_id)
            status = manager.get_status()
            assert status.current.granularity_bytes == 2 * PAGE
            assert status.current.private_effective_bytes == alpha * 4 * PAGE
            assert status.charged_bytes == 6 * PAGE
            assert status.mapped_bytes == PAGE + alpha * 4 * PAGE
            assert all(
                segment.size == 2 * PAGE
                for segment in controls.snapshot().shared_segments
            )
        assert manager.close()
        assert pool.get_memory_usage() == (0, 0)
    finally:
        assert manager.close()
        l1.close()


def test_blocked_sdk_reports_reserved_charge_and_conflict() -> None:
    manager, l1, _, controls = setup(1.0)
    try:
        manager.initialize()
        controls.block = threading.Event()
        manager.request_resize("blocked", 0.0)
        assert controls.entered.wait(5)
        status = manager.get_status()
        assert status.pending_growth_bytes == PAGE
        assert status.quiesced
        with manager.admission() as allowed:
            assert not allowed
        with pytest.raises(RuntimeError, match="Another resize"):
            manager.request_resize("other", 0.5)
        controls.block.set()
        wait(manager, "blocked")
    finally:
        if controls.block is not None:
            controls.block.set()
        assert manager.close()
        l1.close()


def test_failed_release_retains_charge_and_retries() -> None:
    manager, l1, _, controls = setup(0.0)
    try:
        manager.initialize()
        controls.fail_release = True
        manager.request_resize("retry", 1.0)
        wait(manager, "retry", ResizePhase.FAILED)
        status = manager.get_status()
        assert status.charged_bytes == 6 * PAGE
        assert status.mapped_bytes == PAGE
        assert status.resize.sdk_error == 31
        assert status.quiesced
        controls.fail_release = False
        manager.request_resize("retry", 1.0)
        wait(manager, "retry")
    finally:
        assert manager.close()
        l1.close()


def test_failed_start_and_close_preserve_mapping() -> None:
    manager, l1, pool, controls = setup()
    controls.fail_register = True
    with pytest.raises(RuntimeError, match="Capacity control"):
        manager.initialize()
    controls.fail_close = True
    assert not manager.close()
    assert pool.get_memory_usage() == (0, PAGE)
    assert len(controls.snapshot().local_buffers) == 1
    controls.fail_close = False
    assert manager.close()
    assert pool.get_memory_usage() == (0, 0)
    l1.close()


def test_exact_writer_then_python_view_block_physical_handoff() -> None:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.internal_api import L1WriteRetention

    manager, l1, pool, controls = setup(1.0)
    try:
        manager.initialize()
        keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "topology", 0) for i in range(2)]
        layout = MemoryLayoutDesc([torch.Size([PAGE])], [torch.uint8])
        operations = [
            result[1]
            for result in l1.reserve_write_operations(
                keys, [L1WriteRetention.RETAIN] * 2, layout
            ).values()
        ]
        assert all(operations)
        tensor = operations[1].memory_obj.get_tensor(0)
        tensor.fill_(73)
        manager.request_resize("held", 0.0)
        time.sleep(0.15)
        assert manager.resize_status("held").phase is ResizePhase.RECLAIMING
        assert not controls.snapshot().shared_segments
        assert manager.get_status().mapped_bytes == 5 * PAGE
        l1.finish_write_operations(operations)
        operations.clear()
        time.sleep(0.15)
        assert manager.resize_status("held").phase is ResizePhase.RECLAIMING
        assert pool.get_memory_usage()[0] == 0
        assert torch.all(tensor == 73)
        assert not controls.snapshot().shared_segments
        unregister_count = len(controls.unregistered)
        time.sleep(0.1)
        assert len(controls.unregistered) == unregister_count
        del tensor
        wait(manager, "held")
        assert manager.get_status().mapped_bytes == PAGE
    finally:
        assert manager.close()
        l1.close()


def test_failed_private_registration_rebuilds_before_reopening() -> None:
    manager, l1, _, controls = setup(0.0)
    try:
        manager.initialize()
        controls.fail_register = True
        manager.request_resize("private", 1.0)
        wait(manager, "private", ResizePhase.FAILED)
        assert manager.get_status().mapped_bytes == 2 * PAGE
        assert manager.get_status().charged_bytes == 3 * PAGE
        assert manager.get_status().pending_growth_bytes == 0
        assert manager.get_status().quiesced
        failed_id = controls.snapshot().local_buffers[-1].registration_id
        controls.fail_register = False
        manager.request_resize("private", 1.0)
        wait(manager, "private")
        assert failed_id in controls.unregistered
        assert manager.get_status().mapped_bytes == 5 * PAGE
        assert len(controls.snapshot().local_buffers) == 5
    finally:
        assert manager.close()
        l1.close()


def test_exact_read_blocks_force_clear_and_shutdown() -> None:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.internal_api import L1WriteRetention

    manager, l1, pool, controls = setup(1.0)
    manager.initialize()
    key = ObjectKey(ObjectKey.IntHash2Bytes(9), "topology-read", 0)
    layout = MemoryLayoutDesc([torch.Size([PAGE])], [torch.uint8])
    write = l1.reserve_write_operations([key], [L1WriteRetention.RETAIN], layout)[key][
        1
    ]
    assert write is not None
    l1.finish_write_operations([write])
    del write
    read = l1.acquire_read_operations([key])[key][1]
    assert read is not None
    manager.request_resize("read", 0.0)
    time.sleep(0.15)
    assert manager.resize_status("read").phase is ResizePhase.RECLAIMING
    assert pool.get_memory_usage()[0] == PAGE
    assert not manager.close()
    assert not controls.snapshot().closed
    l1.finish_read_operations([read])
    del read
    assert manager.close()
    assert pool.get_memory_usage() == (0, 0)
    l1.close()


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_repeated_busy_growth_retries_same_backing(alpha: float) -> None:
    manager, l1, _, controls = setup(1.0 - alpha)
    try:
        manager.initialize()
        initial_attempts = len(controls.registration_attempts)
        controls.busy_register = alpha == 1.0
        controls.busy_allocate = alpha == 0.0
        manager.request_resize("busy", alpha)
        assert controls.repeated_busy.wait(5)
        status = manager.get_status()
        assert status.resize.phase is ResizePhase.GROWING
        assert status.quiesced
        assert status.mapped_bytes == (2 if alpha else 1) * PAGE
        assert status.charged_bytes == (3 if alpha else 2) * PAGE
        assert status.pending_growth_bytes == 0
        assert len(status.shared.local_buffers) == 1
        with manager.admission() as allowed:
            assert not allowed
        if alpha:
            attempts = controls.registration_attempts[initial_attempts:]
            assert len(attempts) >= 3
            assert len(set(attempts)) == 1
        controls.busy_register = False
        controls.busy_allocate = False
        wait(manager, "busy")
        status = manager.get_status()
        assert status.charged_bytes == 6 * PAGE
        assert status.mapped_bytes == (5 if alpha else 1) * PAGE
        assert status.shared.shared_allocated_bytes == (0 if alpha else 4) * PAGE
        assert len(status.shared.local_buffers) == (5 if alpha else 1)
        assert not status.quiesced
    finally:
        controls.busy_register = False
        controls.busy_allocate = False
        assert manager.close()
        l1.close()


def test_busy_staging_start_retains_owner_for_cleanup() -> None:
    manager, l1, pool, controls = setup()
    try:
        controls.busy_register = True
        with pytest.raises(RuntimeError, match="staging registration is busy"):
            manager.initialize()
        status = manager.get_status()
        assert status.resize.phase is ResizePhase.FAILED
        assert status.mapped_bytes == PAGE
        assert status.charged_bytes == 2 * PAGE
        assert status.quiesced
        assert len(controls.registration_attempts) == 1
        controls.fail_close = True
        assert not manager.close()
        assert pool.get_memory_usage() == (0, PAGE)
        with pytest.raises(RuntimeError, match="already attempted"):
            manager.initialize()
    finally:
        controls.fail_close = False
        assert manager.close()
        l1.close()
