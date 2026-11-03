# SPDX-License-Identifier: Apache-2.0
"""Unpublished Store admission with real CPU allocations and controlled I/O.

The adapter delays real source reads until explicit completion. These checks
prove controller ownership, not native SDK or transport fencing.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
import multiprocessing
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1AdmissionLimits,
    L1WriteOperation,
    L1WriteRetention,
    L2StoreResult,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreAdmissionResult,
    StoreController,
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    BufferOnlyStorePolicy,
    DefaultStorePolicy,
    StorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
KEY = ObjectKey(ObjectKey.IntHash2Bytes(1), "staged-store", 0)
OTHER_KEY = ObjectKey(ObjectKey.IntHash2Bytes(2), "staged-store", 0)
ADAPTER_ID = 17


@dataclass
class Submission:
    """Buffers accepted by the controlled adapter but not yet read."""

    keys: tuple[ObjectKey, ...]
    objects: tuple[MemoryObj, ...]


class ControlledStoreAdapter(MockL2Adapter):
    """Retain exact sources and read them only when the test completes a task."""

    def __init__(self) -> None:
        super().__init__(MockL2AdapterConfig(0.01, 10.0))
        self.submission_entered = threading.Event()
        self.submission_gate = threading.Event()
        self.submission_gate.set()
        self.observed: dict[int, list[bytes]] = {}
        self._completion = create_event_notifier()
        self._task_condition = threading.Condition()
        self._submitted: dict[int, Submission] = {}
        self._completed: dict[int, L2StoreResult] = {}
        self._accepted_count = 0

    def get_store_event_fd(self) -> int:
        return self._completion.fileno()

    def submit_store_task(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        with self._task_condition:
            self._accepted_count += 1
            task_id = self._accepted_count
            self._submitted[task_id] = Submission(tuple(keys), tuple(objects))
            self._task_condition.notify_all()
        self.submission_entered.set()
        assert self.submission_gate.wait(10), "Submission was not released"
        return task_id

    def wait_submitted(self, count: int = 1) -> None:
        """Wait for the requested number of accepted tasks or fail the test."""
        with self._task_condition:
            assert self._task_condition.wait_for(
                lambda: self._accepted_count >= count, timeout=5
            ), "Store task was not submitted"

    def accepted_count(self) -> int:
        """Return the number of tasks accepted since construction."""
        with self._task_condition:
            return self._accepted_count

    def submitted_keys(self, task_id: int) -> tuple[ObjectKey, ...]:
        """Return the keys of an accepted task that has not completed."""
        with self._task_condition:
            return self._submitted[task_id].keys

    def complete(self, task_id: int, result: L2StoreResult) -> None:
        """Read accepted buffers before exposing a terminal adapter result."""
        with self._task_condition:
            submission = self._submitted.pop(task_id)
        self.observed[task_id] = [bytes(obj.byte_array) for obj in submission.objects]
        with self._task_condition:
            self._completed[task_id] = result
        self._completion.notify()

    def pop_completed_store_tasks(self) -> dict[int, L2StoreResult]:
        with self._task_condition:
            completed, self._completed = self._completed, {}
            return completed

    def close(self) -> None:
        self.submission_gate.set()
        self._completion.close()
        super().close()


class UncertainStoreAdapter(ControlledStoreAdapter):
    """Accept buffers without returning the task identity to the controller."""

    def submit_store_task(self, keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        super().submit_store_task(keys, objects)
        raise RuntimeError("injected exception after acceptance")


def make_manager(slots: int = 2) -> L1Manager:
    """Create a small CPU pool whose legacy write TTL expires after one second."""
    return L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                size_in_bytes=slots * 4096, use_lazy=False, shm_name=""
            ),
            write_ttl_seconds=1,
        )
    )


def make_controller(
    manager: L1Manager,
    adapter: ControlledStoreAdapter,
    policy: StorePolicy,
) -> StoreController:
    """Use a nonzero stable ID to distinguish adapter identity from position."""
    return StoreController(
        manager,
        [adapter],
        [L2AdapterDescriptor(ADAPTER_ID, MockL2AdapterConfig(0.01, 10.0))],
        policy,
    )


def reserve(
    manager: L1Manager,
    keys: tuple[ObjectKey, ...] = (KEY,),
    tag: str = "computed",
) -> dict[ObjectKey, L1WriteOperation]:
    """Fill unpublished allocations and return their exact ownership handles."""
    results = manager.reserve_write_operations(
        list(keys), [L1WriteRetention.TEMPORARY] * len(keys), LAYOUT, tag
    )
    handles = {}
    for index, (key, (error, handle)) in enumerate(results.items()):
        assert error == L1Error.SUCCESS and handle is not None
        tensor = handle.memory_obj.get_tensor(0)
        assert tensor is not None
        tensor.fill_(index + 7)
        handles[key] = handle
    return handles


def discard(manager: L1Manager, operations: dict[ObjectKey, L1WriteOperation]) -> None:
    """Release a rejected batch that remains owned by the caller."""
    handles = list(operations.values())
    manager.finish_write_operations(handles, discard=frozenset(handles))


@pytest.mark.parametrize("policy", [DefaultStorePolicy(), BufferOnlyStorePolicy()])
@pytest.mark.parametrize("successful", [True, False])
def test_staging_survives_ttl_then_cleanup_precedes_result(
    policy: StorePolicy, successful: bool
) -> None:
    """TTL, clear and stop cannot release native-owned staging or publish it."""
    manager, adapter = make_manager(), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, policy)
    listener = StoreListener()
    manager.register_listener(listener)
    controller.start()
    stopped = False
    try:
        operations = reserve(manager, (KEY, OTHER_KEY))
        expected = [bytes(op.memory_obj.byte_array) for op in operations.values()]
        future = controller.submit_staged_write_operations(
            operations, adapter_id=ADAPTER_ID
        )
        adapter.wait_submitted()
        cleaned = threading.Event()
        observations: list[tuple[int, int]] = []

        def completed(result: Future[StoreAdmissionResult]) -> None:
            observations.append(
                (
                    manager.get_memory_usage()[0],
                    manager.report_status()["active_write_operation_count"],
                )
            )
            cleaned.set()

        future.add_done_callback(completed)
        time.sleep(1.1)
        assert not future.done()
        for key in operations:
            assert manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST
            assert manager.delete([key], force=True)[key] == L1Error.KEY_IS_LOCKED
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 8192
        assert manager.get_staging_memory_usage() == 8192
        with pytest.raises(RuntimeError, match="active write operations"):
            manager.close()
        with pytest.raises(TimeoutError, match="buffer completion"):
            controller.stop(timeout=0.01)
        adapter.complete(1, L2StoreResult(successful, 8192 if successful else 0))
        keys = frozenset(operations)
        assert future.result(timeout=5) == StoreAdmissionResult(
            keys if successful else frozenset(),
            frozenset() if successful else keys,
        )
        assert cleaned.wait(5) and observations == [(0, 0)]
        controller.stop(timeout=5)
        stopped = True
        controller.stop(timeout=0)
        assert adapter.observed[1] == expected
        assert listener.pop_pending_keys() == []
        assert adapter.accepted_count() == 1
        assert manager.get_memory_usage()[0] == 0
    finally:
        if not stopped:
            adapter.complete(1, L2StoreResult(False, 0))
            controller.stop(timeout=5)
        listener.close()
        adapter.close()
        manager.close()


def test_cancelled_future_keeps_source_until_completion() -> None:
    """Cancelling observation cannot permit an allocator slot to be reused."""
    manager, adapter = make_manager(1), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        operations = reserve(manager)
        expected = bytes(operations[KEY].memory_obj.byte_array)
        future = controller.submit_staged_write_operations(
            operations, adapter_id=ADAPTER_ID
        )
        adapter.wait_submitted()
        assert future.cancel()
        manager.clear(force=True)
        assert manager.get_memory_usage()[0] == 4096
        refusal = manager.reserve_write_operations(
            [OTHER_KEY], [L1WriteRetention.TEMPORARY], LAYOUT, "replacement"
        )
        assert refusal[OTHER_KEY][0] == L1Error.OUT_OF_MEMORY
        adapter.complete(1, L2StoreResult(True, 4096))
        controller.stop(timeout=5)
        assert future.cancelled()
        assert adapter.observed[1] == [expected]
        assert manager.get_memory_usage()[0] == 0
        replacement = reserve(manager, (OTHER_KEY,))
        discard(manager, replacement)
    finally:
        adapter.close()
        manager.close()


def test_same_key_staged_sources_leave_existing_resident_untouched() -> None:
    """Each completion frees its own allocation without changing a same-key peer."""
    manager, adapter = make_manager(3), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, BufferOnlyStorePolicy())
    controller.start()
    try:
        resident = reserve(manager, tag="resident")[KEY]
        first = reserve(manager, tag="first")
        second = reserve(manager, tag="second")
        resident_tensor = resident.memory_obj.get_tensor(0)
        assert resident_tensor is not None
        resident_tensor.fill_(42)
        # Serving admission deliberately does not produce a computed-write event.
        manager.finish_write_operations([resident])
        one = controller.submit_staged_write_operations(first, adapter_id=ADAPTER_ID)
        two = controller.submit_staged_write_operations(second, adapter_id=ADAPTER_ID)
        adapter.wait_submitted(2)
        adapter.complete(2, L2StoreResult(False, 0))
        assert two.result(timeout=5).failed_keys == frozenset([KEY])
        assert not one.done()
        assert manager.get_memory_usage()[0] == 8192
        adapter.complete(1, L2StoreResult(True, 4096))
        assert one.result(timeout=5).admitted_keys == frozenset([KEY])
        controller.stop(timeout=5)
        assert manager.get_memory_usage()[0] == 4096
        error, obj = manager.unsafe_read([KEY])[KEY]
        assert error == L1Error.SUCCESS and obj is resident.memory_obj
        assert torch.all(obj.get_tensor(0) == 42)
        assert adapter.accepted_count() == 2
        manager.finish_read([KEY])
    finally:
        adapter.close()
        manager.close()


def test_rejection_preserves_caller_ownership_and_adapter_identity() -> None:
    """Unavailable targets and stopped queues refuse before ownership transfers."""
    manager, adapter = make_manager(1), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    operations = reserve(manager)
    try:
        with pytest.raises(RuntimeError, match="not accepting"):
            controller.submit_staged_write_operations(operations, adapter_id=ADAPTER_ID)
        controller.start()
        with pytest.raises(ValueError, match="unavailable"):
            controller.submit_staged_write_operations(operations, adapter_id=0)
        removal = controller.request_remove_adapter(ADAPTER_ID)
        with pytest.raises(ValueError, match="unavailable"):
            controller.submit_staged_write_operations(operations, adapter_id=ADAPTER_ID)
        assert removal.wait(5)
        controller.stop(timeout=5)
        with pytest.raises(RuntimeError, match="not accepting"):
            controller.submit_staged_write_operations(operations, adapter_id=ADAPTER_ID)
        assert adapter.accepted_count() == 0
        assert manager.report_status()["active_write_operation_count"] == 1
        discard(manager, operations)
        assert manager.get_memory_usage()[0] == 0
    finally:
        adapter.close()
        manager.close()


@pytest.mark.parametrize("invalid_batch", ["repeated", "mixed-shape"])
def test_invalid_batch_rejection_preserves_all_operations(invalid_batch: str) -> None:
    """Validation rejects the entire batch before transferring any capability."""
    manager, adapter = make_manager(2), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        second = (
            OTHER_KEY
            if invalid_batch == "repeated"
            else ObjectKey(ObjectKey.IntHash2Bytes(3), "other-model", 0)
        )
        operations = reserve(manager, (KEY, second))
        submitted = operations.copy()
        if invalid_batch == "repeated":
            submitted[second] = operations[KEY]
        with pytest.raises(ValueError, match="Repeated|share one shape"):
            controller.submit_staged_write_operations(submitted, adapter_id=ADAPTER_ID)
        assert adapter.accepted_count() == 0
        assert manager.report_status()["active_write_operation_count"] == 2
        discard(manager, operations)
        assert manager.get_memory_usage()[0] == 0
        controller.stop(timeout=5)
    finally:
        adapter.close()
        manager.close()


def test_shutdown_drains_handoffs_queued_behind_submission() -> None:
    """A stop request cannot strand accepted writes waiting in the queue."""
    manager, adapter = make_manager(2), ControlledStoreAdapter()
    adapter.submission_gate.clear()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        first = controller.submit_staged_write_operations(
            reserve(manager), adapter_id=ADAPTER_ID
        )
        assert adapter.submission_entered.wait(5)
        second = controller.submit_staged_write_operations(
            reserve(manager, (OTHER_KEY,)), adapter_id=ADAPTER_ID
        )
        assert controller.report_status()["pending_staged_admission_count"] == 1
        with pytest.raises(TimeoutError):
            controller.stop(timeout=0.01)
        assert manager.get_memory_usage()[0] == 8192
        adapter.submission_gate.set()
        adapter.wait_submitted(2)
        adapter.complete(1, L2StoreResult(True, 4096))
        adapter.complete(2, L2StoreResult(True, 4096))
        controller.stop(timeout=5)
        assert first.result().admitted_keys == frozenset([KEY])
        assert second.result().admitted_keys == frozenset([OTHER_KEY])
        assert manager.get_memory_usage()[0] == 0
    finally:
        adapter.submission_gate.set()
        adapter.close()
        manager.close()


def test_resident_and_staged_sources_share_completion_drain() -> None:
    """One completion consumer independently releases both source variants."""
    manager, adapter = make_manager(2), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        staged = controller.submit_staged_write_operations(
            reserve(manager), adapter_id=ADAPTER_ID
        )
        results = manager.reserve_write_operations(
            [OTHER_KEY], [L1WriteRetention.RETAIN], LAYOUT, "resident"
        )
        error, resident = results[OTHER_KEY]
        assert error == L1Error.SUCCESS and resident is not None
        resident_tensor = resident.memory_obj.get_tensor(0)
        assert resident_tensor is not None
        resident_tensor.fill_(42)
        manager.finish_write_operations([resident], read_locks=0)
        adapter.wait_submitted(2)
        for task_id in (1, 2):
            keys = adapter.submitted_keys(task_id)
            adapter.complete(task_id, L2StoreResult(keys == (OTHER_KEY,), 4096))
        assert staged.result(timeout=5).failed_keys == frozenset([KEY])
        controller.stop(timeout=5)
        assert manager.get_memory_usage()[0] == 4096
        assert manager.report_status()["active_read_operation_count"] == 0
        assert manager.report_status()["active_write_operation_count"] == 0
        assert manager.reserve_read([OTHER_KEY])[OTHER_KEY][0] == L1Error.SUCCESS
        error, obj = manager.unsafe_read([OTHER_KEY])[OTHER_KEY]
        assert error == L1Error.SUCCESS and obj is resident.memory_obj
        assert torch.all(obj.get_tensor(0) == 42)
        manager.finish_read([OTHER_KEY])
    finally:
        adapter.close()
        manager.close()


def test_empty_handoff_has_no_native_task() -> None:
    """An empty batch produces an empty result without submitting buffers."""
    manager, adapter = make_manager(1), ControlledStoreAdapter()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        future = controller.submit_staged_write_operations({}, adapter_id=ADAPTER_ID)
        assert future.result(timeout=5) == StoreAdmissionResult(
            frozenset(), frozenset()
        )
        assert adapter.accepted_count() == 0
        controller.stop(timeout=5)
    finally:
        adapter.close()
        manager.close()


def test_terminal_cleanup_restores_temporary_quota() -> None:
    """Admission completion releases the temporary charge before retry can run."""
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=8192, use_lazy=False, shm_name="")
        ),
        admission_limits=L1AdmissionLimits(retained_bytes=4096, temporary_bytes=4096),
    )
    adapter = ControlledStoreAdapter()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    try:
        future = controller.submit_staged_write_operations(
            reserve(manager), adapter_id=ADAPTER_ID
        )
        adapter.wait_submitted()
        usage = manager.get_admission_usage()
        assert usage.temporary_allocated_bytes == 4096
        assert usage.retained_allocated_bytes == 0
        refused = manager.reserve_write_operations(
            [OTHER_KEY], [L1WriteRetention.TEMPORARY], LAYOUT, "retry"
        )
        assert refused[OTHER_KEY][0] == L1Error.OUT_OF_MEMORY
        adapter.complete(1, L2StoreResult(True, 4096))
        assert future.result(timeout=5).admitted_keys == frozenset([KEY])
        assert manager.get_admission_usage().temporary_allocated_bytes == 0
        replacement = reserve(manager, (OTHER_KEY,), "retry")
        discard(manager, replacement)
        controller.stop(timeout=5)
    finally:
        adapter.close()
        manager.close()


def _verify_uncertain_submission() -> None:
    """Keep ambiguous native ownership quarantined until this process exits."""
    manager, adapter = make_manager(2), UncertainStoreAdapter()
    adapter.submission_gate.clear()
    controller = make_controller(manager, adapter, DefaultStorePolicy())
    controller.start()
    operations = reserve(manager)
    expected = bytes(operations[KEY].memory_obj.byte_array)
    future = controller.submit_staged_write_operations(
        operations, adapter_id=ADAPTER_ID
    )
    assert adapter.submission_entered.wait(5)
    queued = controller.submit_staged_write_operations(
        reserve(manager, (OTHER_KEY,)), adapter_id=ADAPTER_ID
    )
    adapter.submission_gate.set()
    with pytest.raises(RuntimeError, match="unresolved submissions"):
        controller.stop(timeout=5)
    status = controller.report_status()
    assert not status["is_healthy"] and status["requires_process_exit"]
    assert status["unresolved_submission_count"] == 1
    assert status["unresolved_write_operation_count"] == 1
    assert status["unresolved_read_operation_count"] == 0
    assert not future.done()
    assert queued.result().failed_keys == frozenset([OTHER_KEY])
    time.sleep(1.1)
    manager.clear(force=True)
    assert manager.get_memory_usage()[0] == 4096
    with pytest.raises(RuntimeError, match="active write operations"):
        manager.close()
    adapter.complete(1, L2StoreResult(True, 4096))
    assert adapter.observed[1] == [expected]
    assert not future.done()
    assert manager.report_status()["active_write_operation_count"] == 1
    adapter.close()


def test_uncertain_acceptance_quarantines_staging() -> None:
    """No task ID means neither native completion nor Future completion is safe."""
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
