# SPDX-License-Identifier: Apache-2.0
"""Controlled adapter writes into real CPU allocations across expiry and abort.

These tests establish controller ownership, not native Store transport drain.
"""

# Standard
import multiprocessing
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1AdmissionLimits
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from tests.v1.distributed.test_l1_write_operations import KEY, LAYOUT


class PausedLoadAdapter(MockL2Adapter):
    """Report a hit, then pause a real CPU destination write until released."""

    def __init__(self, success: bool = True, key: ObjectKey = KEY) -> None:
        super().__init__(MockL2AdapterConfig(0.01, 10.0))
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.written = threading.Event()
        self.key = key
        self._success = success
        self._completion = create_event_notifier()
        self._lookup_completion = create_event_notifier()
        self._lookup_result: Bitmap | None = None
        self._result: Bitmap | None = None
        self._writer: threading.Thread | None = None

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the controlled lookup notification descriptor."""
        return self._lookup_completion.fileno()

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> int:
        """Report the test key available without allocating another copy."""
        self._lookup_result = Bitmap(len(keys))
        self._lookup_result.set(keys.index(self.key))
        self._lookup_completion.notify()
        return 0

    def query_lookup_and_lock_result(self, task_id: int) -> Bitmap | None:
        """Return the single lookup result once."""
        result, self._lookup_result = self._lookup_result, None
        return result

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """The controlled lookup has no shared-cache lock to release."""

    def get_load_event_fd(self) -> int:
        """Return the controlled load notification descriptor."""
        return self._completion.fileno()

    def submit_load_task(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        """Accept one destination and start a barrier-delayed writer."""
        assert keys == [self.key] and self._writer is None
        self._writer = threading.Thread(target=self._write, args=(objects[0],))
        self._writer.start()
        return 1

    def query_load_result(self, task_id: int) -> Bitmap | None:
        """Return completion once, only after the writer has ended access."""
        result, self._result = self._result, None
        return result

    def close(self) -> None:
        """Drain the controlled writer before closing notification descriptors."""
        self.resume.set()
        if self._writer is not None:
            self._writer.join(timeout=5)
            assert not self._writer.is_alive()
        self._completion.close()
        self._lookup_completion.close()
        super().close()

    def _write(self, obj: MemoryObj) -> None:
        self.entered.set()
        assert self.resume.wait(10), "Paused write was not released"
        tensor = obj.get_tensor(0)
        assert tensor is not None
        tensor.fill_(7)
        self.written.set()
        result = Bitmap(1)
        if self._success:
            result.set(0)
        self._result = result
        self._completion.notify()


class UncertainLoadAdapter(PausedLoadAdapter):
    """Accept a real writer and then fail to return its task ID."""

    def submit_load_task(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        """Simulate an exception after acceptance, whose outcome is unknown."""
        super().submit_load_task(keys, objects)
        raise RuntimeError("injected exception after load acceptance")


def make_manager() -> L1Manager:
    """Use a single native slot and a short native write reservation TTL."""
    return L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False, shm_name=""),
            write_ttl_seconds=1,
        )
    )


def make_controller(
    manager: L1Manager, adapter: PausedLoadAdapter
) -> PrefetchController:
    """Route one row through the controlled adapter and retain successful bytes."""
    return PrefetchController(
        [manager],
        [L1ManagerDescriptor(0, manager.config)],
        [adapter],
        [L2AdapterDescriptor(0, MockL2AdapterConfig(0.01, 10.0))],
        RetainPrefetchPolicy(),
    )


def make_spec() -> PrefetchTaskSpec:
    """Request one row and one chunk."""
    return PrefetchTaskSpec(key_groups=[GroupedObjectKeys([KEY], 0, LAYOUT)])


@pytest.mark.parametrize("success", [True, False])
def test_load_owns_unpublished_destination_past_ttl(success: bool) -> None:
    """Only terminal success makes late bytes readable; failure frees its slot."""
    manager, adapter = make_manager(), PausedLoadAdapter(success)
    controller = make_controller(manager, adapter)
    controller.start()
    try:
        request = controller.submit_prefetch_request(make_spec())
        assert adapter.entered.wait(5)
        time.sleep(1.1)
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 4096
        assert manager.reserve_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST
        assert manager.delete([KEY], force=True)[KEY] == L1Error.KEY_IS_LOCKED
        assert controller.query_prefetch_result(request) is None
        adapter.resume.set()
        assert controller.wait_prefetch_result(request, timeout=5)
        result = controller.query_prefetch_result(request)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == ([0] if success else [])
        assert adapter.written.is_set()
        assert manager.report_status()["active_write_operation_count"] == 0
        if success:
            error, obj = manager.unsafe_read([KEY])[KEY]
            assert error == L1Error.SUCCESS and obj is not None
            assert torch.all(obj.get_tensor(0) == 7)
            manager.finish_read([KEY])
        else:
            assert manager.get_memory_usage()[0] == 0
    finally:
        adapter.resume.set()
        controller.stop(timeout=5)
        adapter.close()
        manager.close()


def test_shutdown_timeout_keeps_destination_then_drains() -> None:
    """Cancellation and timeout keep bytes charged until the delayed write ends."""
    manager, adapter = make_manager(), PausedLoadAdapter()
    controller = make_controller(manager, adapter)
    controller.start()
    stopped = False
    try:
        request = controller.submit_prefetch_request(make_spec())
        assert adapter.entered.wait(5)
        time.sleep(1.1)
        with pytest.raises(TimeoutError, match="buffer completion"):
            controller.stop(timeout=0.02)
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 4096
        with pytest.raises(RuntimeError, match="active write operations"):
            manager.close()
        with pytest.raises(RuntimeError, match="stopping"):
            controller.submit_prefetch_request(make_spec())
        adapter.resume.set()
        controller.stop(timeout=5)
        stopped = True
        controller.stop(timeout=0)
        assert adapter.written.is_set()
        assert manager.get_memory_usage()[0] == 0
        assert manager.report_status()["active_write_operation_count"] == 0
        result = controller.query_prefetch_result(request)
        assert result is not None and result.hit_cells[0].popcount() == 0
    finally:
        adapter.resume.set()
        if not stopped:
            controller.stop(timeout=5)
        adapter.close()
        manager.close()


def _verify_uncertain_submission() -> None:
    """Keep irreversible quarantine confined to a disposable process."""
    manager, adapter = make_manager(), UncertainLoadAdapter()
    controller = make_controller(manager, adapter)
    controller.start()
    try:
        controller.submit_prefetch_request(make_spec())
        assert adapter.entered.wait(5)
        with pytest.raises(RuntimeError, match="unresolved submissions"):
            controller.stop(timeout=5)
        status = controller.report_status()
        assert status["requires_process_exit"]
        assert status["unresolved_submission_count"] == 1
        assert status["unresolved_write_operation_count"] == 1
        assert not status["is_healthy"]
        time.sleep(1.1)
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 4096
        assert manager.delete([KEY], force=True)[KEY] == L1Error.KEY_IS_LOCKED
        with pytest.raises(RuntimeError, match="active write operations"):
            manager.close()
        with pytest.raises(RuntimeError, match="stopping"):
            controller.submit_prefetch_request(make_spec())
        adapter.resume.set()
    finally:
        adapter.close()
    assert adapter.written.is_set()
    assert manager.report_status()["active_write_operation_count"] == 1
    assert manager.reserve_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST


def test_ambiguous_submission_quarantines_accepted_destination() -> None:
    """An exception after acceptance cannot free the buffer a worker still uses."""
    child = multiprocessing.get_context("spawn").Process(
        target=_verify_uncertain_submission
    )
    child.start()
    try:
        child.join(timeout=20)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def _verify_partial_submission() -> None:
    """Drain accepted work, quarantine uncertain work, and free unsubmitted work."""
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(i), "write-lifetime", 0) for i in range(3)
    ]
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=12288, use_lazy=False, shm_name="")
        )
    )
    accepted = PausedLoadAdapter(key=keys[0])
    uncertain = UncertainLoadAdapter(key=keys[1])
    unsubmitted = PausedLoadAdapter(key=keys[2])
    adapters = [accepted, uncertain, unsubmitted]
    controller = PrefetchController(
        [manager],
        [L1ManagerDescriptor(0, manager.config)],
        adapters,
        [L2AdapterDescriptor(i, MockL2AdapterConfig(0.01, 10.0)) for i in range(3)],
        RetainPrefetchPolicy(),
    )
    controller.start()
    try:
        request = controller.submit_prefetch_request(
            PrefetchTaskSpec(key_groups=[GroupedObjectKeys(keys, 0, LAYOUT)])
        )
        assert accepted.entered.wait(5) and uncertain.entered.wait(5)
        with pytest.raises(TimeoutError, match="buffer completion"):
            controller.stop(timeout=0.02)
        assert manager.get_memory_usage()[0] == 8192
        assert manager.report_status()["active_write_operation_count"] == 2
        assert controller.report_status()["unresolved_write_operation_count"] == 1
        assert not unsubmitted.entered.is_set()
        accepted.resume.set()
        with pytest.raises(RuntimeError, match="unresolved submissions"):
            controller.stop(timeout=5)
        assert accepted.written.is_set()
        assert manager.get_memory_usage()[0] == 4096
        assert manager.report_status()["active_write_operation_count"] == 1
        result = controller.query_prefetch_result(request)
        assert result is not None and result.hit_cells[0].popcount() == 0
        assert all(
            error == L1Error.KEY_NOT_EXIST
            for error, _ in manager.reserve_read(keys).values()
        )
    finally:
        for adapter in adapters:
            adapter.close()
    assert uncertain.written.is_set()
    assert manager.get_memory_usage()[0] == 4096


def test_partial_submission_tracks_each_destination_owner() -> None:
    """An adapter submit failure does not release earlier accepted writers."""
    child = multiprocessing.get_context("spawn").Process(
        target=_verify_partial_submission
    )
    child.start()
    try:
        child.join(timeout=20)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


@pytest.mark.parametrize("retained_bytes", [0, 4096])
def test_prefetch_retention_falls_back_to_temporary(retained_bytes: int) -> None:
    """A locked remote hit remains readable without private retention quota."""
    manager, adapter = make_manager(), PausedLoadAdapter()
    manager.set_admission_limits(L1AdmissionLimits(retained_bytes, 4096))
    controller = make_controller(manager, adapter)
    controller.start()
    try:
        request = controller.submit_prefetch_request(make_spec())
        assert adapter.entered.wait(5)
        adapter.resume.set()
        assert controller.wait_prefetch_result(request, timeout=5)
        result = controller.query_prefetch_result(request)
        assert result is not None and result.hit_cells[0].popcount() == 1
        error, obj = manager.unsafe_read([KEY])[KEY]
        assert error is L1Error.SUCCESS and obj is not None
        assert torch.all(obj.get_tensor(0) == 7)
        manager.finish_read([KEY])
        usage = manager.get_admission_usage()
        assert usage.temporary_allocated_bytes == 0
        assert usage.retained_allocated_bytes == retained_bytes
        assert manager.reserve_read([KEY])[KEY][0] is (
            L1Error.SUCCESS if retained_bytes else L1Error.KEY_NOT_EXIST
        )
        if retained_bytes:
            manager.finish_read([KEY])
    finally:
        adapter.resume.set()
        controller.stop(timeout=5)
        adapter.close()
        manager.close()


def test_unlocked_prefetch_requires_retention_capacity() -> None:
    """An unlocked result cannot promise a hit in disposable temporary storage."""
    manager, adapter = make_manager(), PausedLoadAdapter()
    manager.set_admission_limits(L1AdmissionLimits(0, 4096))
    controller = make_controller(manager, adapter)
    controller.start()
    try:
        request = controller.submit_prefetch_request(
            PrefetchTaskSpec(
                key_groups=make_spec().key_groups,
                lock_mode=PrefetchLockMode.NO_LOCK,
            )
        )
        assert controller.wait_prefetch_result(request, timeout=5)
        result = controller.query_prefetch_result(request)
        assert result is not None and result.hit_cells[0].popcount() == 0
        assert not adapter.entered.is_set()
        assert manager.get_memory_usage()[0] == 0
    finally:
        controller.stop(timeout=5)
        adapter.close()
        manager.close()
