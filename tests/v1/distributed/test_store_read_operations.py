# SPDX-License-Identifier: Apache-2.0
"""Store source ownership across delayed completion and uncertain submission.

The adapter is controlled Python code; allocations and TTL locks are native.
These checks do not establish native SDK or transport fencing guarantees.
"""

# Standard
import multiprocessing
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import (
    BufferOnlyStorePolicy,
    DefaultStorePolicy,
    StorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from tests.v1.distributed.test_l1_read_operations import KEY, populate


class PausedStoreAdapter(MockL2Adapter):
    """Accept one task, then pause its actual CPU buffer read until released."""

    def __init__(self, result: L2StoreResult) -> None:
        super().__init__(MockL2AdapterConfig(0.01, 10.0))
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.observed: list[bytes] = []
        self._completion = create_event_notifier()
        self._result = result
        self._completed: dict[int, L2StoreResult] = {}
        self._completion_lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def get_store_event_fd(self) -> int:
        return self._completion.fileno()

    def submit_store_task(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        assert len(keys) == len(objects) == 1 and self._reader is None
        self._reader = threading.Thread(target=self._read, args=(objects,))
        self._reader.start()
        return 1

    def pop_completed_store_tasks(self) -> dict[int, L2StoreResult]:
        with self._completion_lock:
            completed, self._completed = self._completed, {}
            return completed

    def close(self) -> None:
        self.resume.set()
        if self._reader is not None:
            self._reader.join(timeout=5)
            assert not self._reader.is_alive()
        self._completion.close()
        super().close()

    def _read(self, objects: list[MemoryObj]) -> None:
        self.entered.set()
        assert self.resume.wait(10), "Paused read was not released"
        self.observed = [bytes(obj.byte_array) for obj in objects]
        with self._completion_lock:
            self._completed[1] = self._result
        self._completion.notify()


class UncertainSubmissionAdapter(PausedStoreAdapter):
    """Start a real delayed buffer reader, then fail to return its task ID."""

    def submit_store_task(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        super().submit_store_task(keys, objects)
        raise RuntimeError("injected exception after task acceptance")


def make_manager() -> L1Manager:
    """Create a native one-slot pool with a short legacy reservation TTL."""
    return L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False, shm_name=""),
            read_ttl_seconds=1,
        )
    )


def make_controller(
    manager: L1Manager, adapter: PausedStoreAdapter, policy: StorePolicy
) -> StoreController:
    """Route every committed object into the controlled adapter."""
    return StoreController(
        manager,
        [adapter],
        [L2AdapterDescriptor(0, MockL2AdapterConfig(0.01, 10.0))],
        policy,
    )


@pytest.mark.parametrize(
    ("policy", "result", "retained_bytes"),
    [
        (DefaultStorePolicy(), L2StoreResult(True, 4096), 4096),
        (BufferOnlyStorePolicy(), L2StoreResult(True, 4096), 0),
        (BufferOnlyStorePolicy(), L2StoreResult(False, 0), 4096),
    ],
)
def test_store_pins_until_completion_and_shutdown_drain(
    policy: StorePolicy, result: L2StoreResult, retained_bytes: int
) -> None:
    """TTL expiry and a shutdown timeout cannot release an accepted source read."""
    manager, adapter = make_manager(), PausedStoreAdapter(result)
    controller = make_controller(manager, adapter, policy)
    controller.start()
    stopped = False
    try:
        populate(manager)
        assert adapter.entered.wait(5)
        manager.reserve_read([KEY])
        error, obj = manager.unsafe_read([KEY])[KEY]
        assert error == L1Error.SUCCESS and obj is not None
        expected = bytes(obj.byte_array)
        time.sleep(1.1)
        with pytest.raises(TimeoutError, match="buffer completion"):
            controller.stop(timeout=0.02)
        assert manager.delete([KEY], force=True)[KEY] == L1Error.KEY_IS_LOCKED
        manager.clear(force=True)
        with pytest.raises(RuntimeError, match="active read operations"):
            manager.close()
        assert manager.get_memory_usage()[0] == 4096
        adapter.resume.set()
        controller.stop(timeout=5)
        stopped = True
        assert adapter.observed == [expected]
        assert manager.report_status()["active_read_operation_count"] == 0
        assert manager.get_memory_usage()[0] == retained_bytes
    finally:
        adapter.resume.set()
        if not stopped:
            controller.stop(timeout=5)
        adapter.close()
        manager.close()


def test_shutdown_during_acquisition_does_not_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch paused before dispatch releases its handles when stop is observed."""
    manager = make_manager()
    adapter = PausedStoreAdapter(L2StoreResult(True, 4096))
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    acquired, resume = threading.Event(), threading.Event()
    acquire = manager.acquire_read_operations

    def paused_acquire(keys):
        result = acquire(keys)
        acquired.set()
        assert resume.wait(5), "Acquisition was not released"
        return result

    monkeypatch.setattr(manager, "acquire_read_operations", paused_acquire)
    controller.start()
    stopped = False
    try:
        populate(manager)
        assert acquired.wait(5)
        with pytest.raises(TimeoutError):
            controller.stop(timeout=0.01)
        resume.set()
        controller.stop(timeout=5)
        stopped = True
        assert not adapter.entered.is_set()
        assert manager.report_status()["active_read_operation_count"] == 0
        assert manager.delete([KEY])[KEY] == L1Error.SUCCESS
    finally:
        resume.set()
        if not stopped:
            controller.stop(timeout=5)
        adapter.close()
        manager.close()


def _verify_uncertain_submission() -> None:
    """Run irrecoverable ownership ambiguity inside a disposable process."""
    manager = make_manager()
    adapter = UncertainSubmissionAdapter(L2StoreResult(True, 4096))
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        populate(manager)
        assert adapter.entered.wait(5)
        with pytest.raises(RuntimeError, match="unresolved submissions"):
            controller.stop(timeout=5)
        status = controller.report_status()
        assert status["unresolved_submission_count"] == 1 and not status["is_healthy"]
        assert status["unresolved_read_operation_count"] == 1
        assert status["requires_process_exit"]
        manager.reserve_read([KEY])
        error, obj = manager.unsafe_read([KEY])[KEY]
        assert error == L1Error.SUCCESS and obj is not None
        expected = bytes(obj.byte_array)
        time.sleep(1.1)
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 4096
        assert manager.delete([KEY], force=True)[KEY] == L1Error.KEY_IS_LOCKED
        with pytest.raises(RuntimeError, match="active read operations"):
            manager.close()
        adapter.resume.set()
    finally:
        adapter.close()
    assert adapter.observed == [expected]
    assert manager.report_status()["active_read_operation_count"] == 1
    # The missing task ID cannot be reconstructed safely. Process exit owns cleanup.


def test_submission_exception_retains_accepted_buffer() -> None:
    """An unknown submit failure cannot be treated as a safe rejection."""
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
