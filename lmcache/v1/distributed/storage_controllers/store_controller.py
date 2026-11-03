# SPDX-License-Identifier: Apache-2.0
"""
Store Controller: asynchronously copies data from L1 to L2 after writes complete.

The controller runs a background thread with an event-driven loop that:
1. Listens for L1 write-completion events via StoreListener.
   It also accepts explicit handoffs of unpublished write operations.
2. Submits store tasks to L2 adapters based on StorePolicy decisions.
3. Monitors L2 task completion via event fds.
4. Releases exact source capabilities after terminal adapter completion.
"""

# Standard
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
import enum
import select
import threading
import weakref

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerInterface,
    L1ManagerListener,
    L1ReadOperation,
    L1WriteOperation,
)
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.adapter_lifecycle import (
    AddAdapterOp,
    RemoveAdapterOp,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    StorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

logger = init_logger(__name__)

# Poll timeout in milliseconds for the store loop
STORE_LOOP_POLL_TIMEOUT_MS = 500


def _group_keys_by_shape(
    keys: list[ObjectKey],
) -> dict[tuple, list[ObjectKey]]:
    """Group ``keys`` by the fields that determine their KV cache shape.

    Each bucket shares a single ``(shape, dtype)``, so each bucket can be
    submitted as one ``submit_store_task`` call. Today the shape is pinned
    by ``(model_name, kv_rank)`` — ``kv_rank`` packs ``world_size`` and
    parallelism config, so different TP/PP setups land in different
    buckets. Extend the grouping tuple when a new shape-affecting field is
    added to ``ObjectKey``.
    """
    groups: dict[tuple, list[ObjectKey]] = defaultdict(list)
    for key in keys:
        groups[(key.model_name, key.kv_rank)].append(key)
    return groups


# Helper classes (module-level, before main class)


class StoreListener(L1ManagerListener):
    """
    Listener that receives L1 write-completion callbacks and enqueues
    keys for the StoreController's background loop.

    The ``on_keys_write_finished`` callback is invoked inside L1Manager's
    lock, so it must be non-blocking. It appends keys to an internal list
    and signals an eventfd to wake up the controller's select.poll().
    """

    def __init__(self) -> None:
        self._pending_keys: list[ObjectKey] = []
        self._accepting = True
        self._lock = threading.Lock()
        self._event_fd = create_event_notifier()

    def get_event_fd(self) -> int:
        """
        Return the notifier fd that is signaled when new keys are
        available.

        Returns:
            int: The readable file descriptor for poller registration.
        """
        return self._event_fd.fileno()

    def notify(self) -> None:
        """Signal the notifier to wake any blocked poll() waiter."""
        self._event_fd.notify()

    def pop_pending_keys(self) -> list[ObjectKey]:
        """
        Pop all pending keys from the queue.

        This is non-blocking and should be called by the StoreController's
        main loop after select.poll() indicates the eventfd is ready.

        Returns:
            list[ObjectKey]: All keys enqueued since the last pop.
        """
        with self._lock:
            keys = self._pending_keys
            self._pending_keys = []
        return keys

    def pending_count(self) -> int:
        """Return the number of pending keys waiting to be processed."""
        with self._lock:
            return len(self._pending_keys)

    def stop_admission(self) -> None:
        """Discard unsubmitted notifications and reject further writes on shutdown."""
        with self._lock:
            self._accepting = False
            self._pending_keys.clear()

    # L1ManagerListener implementation

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """
        Enqueue keys and signal the notifier.

        Called inside L1Manager's lock. Must be fast and must not
        call any L1Manager methods (would deadlock).

        Args:
            keys (list[ObjectKey]): Keys that finished writing.
        """
        with self._lock:
            if not self._accepting:
                return
            self._pending_keys.extend(keys)
            self._event_fd.notify()

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        # No op here because we don't want to trigger store when the
        # objects are prefetched to L1.
        pass

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def close(self) -> None:
        """Close the notifier."""
        with self._lock:
            self._accepting = False
            self._pending_keys.clear()
            self._event_fd.close()


class StorePhase(enum.Enum):
    """Phases a store task may be in. Currently there is only an
    L2_STORE phase; new phases (e.g. verify, ack) would be added here and
    threaded through ``signaled_adapters`` without changing
    ``_advance_request``'s signature."""

    L2_STORE = enum.auto()


@dataclass(frozen=True)
class StoreAdmissionResult:
    """Whole-task shared admission outcome after local staging is released."""

    admitted_keys: frozenset[ObjectKey]
    failed_keys: frozenset[ObjectKey]


@dataclass(frozen=True)
class ResidentExport:
    """Read capabilities for already published L1 objects."""

    operations: tuple[L1ReadOperation, ...]


@dataclass(frozen=True)
class StagedAdmission:
    """Unpublished allocations and their cleanup-before-completion promise."""

    operations: tuple[L1WriteOperation, ...]
    completion: Future[StoreAdmissionResult]


StoreSource = ResidentExport | StagedAdmission


@dataclass
class InFlightStoreTask:
    """
    Tracks a single submitted L2 store task so the controller can
    release exact L1 allocations when local buffer access completes.
    """

    adapter_index: int
    """Which L2 adapter this task was submitted to."""

    keys: list[ObjectKey]
    """All keys that were submitted in this store task."""

    source: StoreSource
    """One allocation capability per key, held through adapter completion."""

    l2_store_result: bool | None = None
    """L2 outcome (True=success, False=failure, None=still in flight)."""

    l2_bytes_transferred: int = 0
    """Bytes actually transferred by the adapter for this task."""


# Main class


class StoreController(StorageControllerInterface):
    """
    Asynchronously stores L1 data to L2 adapters after write completion.

    The controller:
    1. Registers a StoreListener with L1Manager to receive
       on_keys_write_finished callbacks.
    2. Runs a background thread with an event-driven loop using
       select.poll() on the listener eventfd and all L2 adapter
       store eventfds.
    3. On new keys: consults StorePolicy to decide targets,
       acquires allocation read operations and submits their MemoryObjs in store
       tasks to L2 adapters.
    4. On L2 task completion: releases operations, atomically marking their
       allocations for discard per policy.
       Explicit staged admissions always discard and resolve their result only
       after cleanup, without publishing or notifying a resident write.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The store policy for deciding targets and deletions.
    """

    # Singleton dispatch for ``lmcache_mp.num_inflight_l2_stores``: tests may
    # construct multiple controllers but the OTel SDK only honors the first
    # gauge registration, so the callback reads from the most recently built
    # instance via ``_gauge_target``.
    _gauge_registered: bool = False
    _gauge_target: "StoreController | None" = None
    _gauge_targets: weakref.WeakValueDictionary[str, "StoreController"] = (
        weakref.WeakValueDictionary()
    )

    def __init__(
        self,
        l1_manager: L1ManagerInterface,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[L2AdapterDescriptor],
        policy: StorePolicy,
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters: dict[int, L2AdapterInterface] = {
            desc.index: adapter
            for desc, adapter in zip(adapter_descriptors, l2_adapters, strict=True)
        }
        self._adapter_descriptors: dict[int, L2AdapterDescriptor] = {
            desc.index: desc for desc in adapter_descriptors
        }
        self._policy = policy

        # Adapters that are being drained and will be removed after all
        # the in-flight operations are done.
        self._draining: dict[int, threading.Event] = {}

        # Control-plane queue for runtime add/remove, used by the internal
        # loop thread
        self._adapter_ops_lock = threading.Lock()
        self._pending_adapter_ops: list[AddAdapterOp | RemoveAdapterOp] = []
        self._pending_staged_admissions: list[InFlightStoreTask] = []
        # Target eligibility and staged queue admission share the shutdown lock.
        self._staged_adapter_ids = set(self._l2_adapters)
        self._adapter_ctrl_efd = create_event_notifier()

        self._listener = StoreListener()
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()

        # (adapter_index, task_id) -> InFlightStoreTask
        # Composite key is needed because task IDs are only unique
        # within a single adapter, not across adapters.
        self._in_flight_tasks: dict[tuple[int, L2TaskId], InFlightStoreTask] = {}
        self._unresolved_submissions: list[InFlightStoreTask] = []
        self._deferred_l1_deletion_count = 0

        # Shadow counter for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0

        StoreController._gauge_target = self
        StoreController._gauge_targets[l1_manager.config.tag] = self
        if not StoreController._gauge_registered:
            StoreController._gauge_registered = True
            register_gauge(
                "lmcache.l2_store",
                "lmcache_mp.num_inflight_l2_stores",
                "L2 store tasks currently executing, per adapter",
                lambda: [
                    observation
                    for c in list(StoreController._gauge_targets.values())
                    for observation in c.get_inflight_stores_observations()
                ],
            )
            register_gauge(
                "lmcache.l2_store",
                "lmcache_mp.l2_store_adapters",
                (
                    "Count of L2 adapters attached to the store controller, "
                    "tagged by ``state`` (active or draining)."
                ),
                lambda: [
                    (count, {**attrs, "l1_tag": tag})
                    for tag, c in list(StoreController._gauge_targets.items())
                    for count, attrs in c.get_adapter_state_observations()
                ],
            )

        # Map store eventfd -> adapter id for quick lookup in poll results
        self._efd_to_adapter_index: dict[int, int] = {}
        for adapter_id, adapter in self._l2_adapters.items():
            self._efd_to_adapter_index[adapter.get_store_event_fd()] = adapter_id

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._store_loop,
            daemon=True,
        )

    def start(self) -> None:
        """Start the background store loop thread."""
        logger.info("Starting StoreController...")
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop new exports and wait for existing local buffer access to finish.

        Args:
            timeout: Maximum seconds to wait for adapter completions.

        Raises:
            TimeoutError: Work still accesses allocations. The loop continues
                draining and a later stop call may complete shutdown.
            RuntimeError: A submission raised without establishing acceptance or
                rejection. Its allocations remain quarantined until process exit.

        Note:
            Repeated calls after successful shutdown are harmless.
            Neither a shutdown request nor a timeout releases allocation handles.
            A submission already being dispatched may finish after stop begins;
            it remains tracked and must drain before shutdown can succeed.
            Adapter completion must mean local source-buffer access has ended;
            it does not prove remote transport or shared-segment quiescence.
        """
        with self._adapter_ops_lock:
            self._stop_flag.set()
        self._listener.stop_admission()
        if self._thread.is_alive():
            self._listener.notify()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Store shutdown is waiting for buffer completion")
        if self._unresolved_submissions:
            raise RuntimeError(
                "Store shutdown has unresolved submissions; buffers pinned"
            )
        self._listener.close()
        self._adapter_ctrl_efd.close()
        if StoreController._gauge_targets.get(self._l1_manager.config.tag) is self:
            del StoreController._gauge_targets[self._l1_manager.config.tag]

    def submit_staged_write_operations(
        self,
        operations: dict[ObjectKey, L1WriteOperation],
        *,
        adapter_id: int,
    ) -> Future[StoreAdmissionResult]:
        """Transfer unpublished source ownership to one explicit adapter.

        Args:
            operations: Keys and original active write handles from this L1
                manager, all with the same layout. The caller must finish
                writing before handoff and must not access or complete accepted
                handles afterward. Keys must match their reserved handles.
            adapter_id: Stable descriptor index of the target adapter.

        Returns:
            A future resolving after terminal adapter completion and exact
            staging discard. A failed task fails every key in its batch.
            Cancellation only cancels observation; it never releases ownership.
            Unknown submission acceptance quarantines handles and leaves the
            future unresolved until process exit.

        Raises:
            ValueError: The target is unavailable, shapes differ, or handles
                repeat, are stale, copied, foreign, or bound to another key.
                No ownership transfers on rejection.
            RuntimeError: The controller is not running or has stopped
                admission. The caller retains all handles.

        Accepted submissions drain during shutdown. Queued submissions that
        have not reached an adapter when acceptance becomes uncertain fail
        safely. Buffers are never published to L1 or routed by StorePolicy.
        """
        operations = operations.copy()
        keys = list(operations)
        handles = tuple(operations.values())
        if len(set(handles)) != len(handles):
            raise ValueError("Repeated staged write operation")
        if len(_group_keys_by_shape(keys)) > 1:
            raise ValueError("Staged write operations must share one shape")
        completion: Future[StoreAdmissionResult] = Future()
        with self._adapter_ops_lock:
            if self._stop_flag.is_set() or not self._thread.is_alive():
                raise RuntimeError("StoreController is not accepting staged writes")
            if adapter_id not in self._staged_adapter_ids:
                raise ValueError(f"Store adapter {adapter_id} is unavailable")
            if not keys:
                completion.set_result(StoreAdmissionResult(frozenset(), frozenset()))
                return completion
            # Notify before appending so notifier failure cannot transfer ownership.
            self._adapter_ctrl_efd.notify()
            transferred = self._l1_manager.transfer_write_operations(operations)
            source = StagedAdmission(tuple(transferred.values()), completion)
            task = InFlightStoreTask(adapter_id, keys, source)
            self._pending_staged_admissions.append(task)
        return completion

    def report_status(self) -> dict:
        """Return a status dict for the store controller."""
        is_healthy = self._thread.is_alive() and not self._unresolved_submissions
        num_draining = len(self._draining)
        with self._adapter_ops_lock:
            pending_staged = len(self._pending_staged_admissions)
        return {
            "is_healthy": is_healthy,
            "thread_alive": self._thread.is_alive(),
            "pending_keys_count": self._listener.pending_count(),
            "pending_staged_admission_count": pending_staged,
            "in_flight_task_count": self._status_in_flight_count,
            "num_l2_adapters": len(self._l2_adapters),
            "num_active_adapters": len(self._l2_adapters) - num_draining,
            "num_draining_adapters": num_draining,
            "deferred_l1_deletion_count": self._deferred_l1_deletion_count,
            "unresolved_submission_count": len(self._unresolved_submissions),
            "unresolved_read_operation_count": sum(
                len(task.source.operations)
                for task in self._unresolved_submissions
                if isinstance(task.source, ResidentExport)
            ),
            "unresolved_write_operation_count": sum(
                len(task.source.operations)
                for task in self._unresolved_submissions
                if isinstance(task.source, StagedAdmission)
            ),
            "requires_process_exit": bool(self._unresolved_submissions),
        }

    def add_adapter(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        descriptor: L2AdapterDescriptor,
    ) -> None:
        """Blocking function to add a new adapter into the store controller
        with the specified adapter ID and descriptor.

        Args:
            adapter_id: Stable id assigned by the StorageManager.
            adapter: The adapter instance to attach.
            descriptor: The adapter's descriptor (``descriptor.index`` must
                equal ``adapter_id``).

        Raises:
            RuntimeError: If the background loop did not apply the op in
                time (e.g. the loop is not running).
        """
        op = AddAdapterOp(
            adapter_id=adapter_id,
            adapter=adapter,
            descriptor=descriptor,
            done=threading.Event(),
        )
        with self._adapter_ops_lock:
            self._pending_adapter_ops.append(op)
        self._adapter_ctrl_efd.notify()
        if not op.done.wait(timeout=STORE_LOOP_POLL_TIMEOUT_MS / 1000 + 5.0):
            raise RuntimeError(
                f"StoreController did not attach adapter {adapter_id} in time"
            )

    def request_remove_adapter(self, adapter_id: int) -> threading.Event:
        """Non-blocking function to request the removal of a L2 adapter
        specified by the adapter ID.

        New store tasks stop routing to the adapter immediately; in-flight
        tasks are allowed to complete.

        Args:
            adapter_id: Stable id of the adapter to drain.

        Returns:
            An Event signaled when the adapter is fully drained.
        """
        op = RemoveAdapterOp(adapter_id=adapter_id, done=threading.Event())
        with self._adapter_ops_lock:
            self._staged_adapter_ids.discard(adapter_id)
            self._pending_adapter_ops.append(op)
        self._adapter_ctrl_efd.notify()
        return op.done

    def get_adapter_state_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """``(count, {"state": ...})`` tuples for the ``lmcache_mp.l2_adapters``
        gauge. ``len()`` reads are GIL-atomic, safe from the OTel thread."""
        num_draining = len(self._draining)
        return [
            (len(self._l2_adapters) - num_draining, {"state": "active"}),
            (num_draining, {"state": "draining"}),
        ]

    def get_inflight_stores_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(count, attributes)`` snapshot for the
        ``lmcache_mp.num_inflight_l2_stores`` gauge.

        ``dict.copy()`` is GIL-atomic in CPython, so reading from the
        OTel reader thread while the store loop mutates is safe; the
        snapshot may be one mutation stale, which is fine at the 10 s
        scrape cadence.
        """
        counts: dict[int, int] = defaultdict(int)
        for adapter_index, _ in self._in_flight_tasks.copy():
            counts[adapter_index] += 1
        observations: list[tuple[int | float, dict[str, object]]] = []
        for idx, count in counts.items():
            # A just-finalized adapter may be gone from the descriptor map
            # while a stale in-flight snapshot still references it; skip it
            # rather than KeyError on the OTel reader thread.
            desc = self._adapter_descriptors.get(idx)
            if desc is None:
                continue
            observations.append(
                (count, {"l2_name": desc.type_name, "adapter_index": idx})
            )
        return observations

    # Private methods

    def _store_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The StoreListener's eventfd (new keys from L1 writes).
        - Each L2 adapter's store eventfd (completed store tasks).

        Stops admitting work when requested; exits after accepted tasks complete.
        """
        poller = select.poll()

        listener_efd = self._listener.get_event_fd()
        poller.register(listener_efd, select.POLLIN)
        poller.register(self._adapter_ctrl_efd.fileno(), select.POLLIN)

        for efd in self._efd_to_adapter_index:
            poller.register(efd, select.POLLIN)

        while True:
            # First, apply runtime add/remove of the L2 adapters.
            self._apply_pending_adapter_ops(poller)
            self._submit_pending_staged_admissions()
            with self._adapter_ops_lock:
                if (
                    self._stop_flag.is_set()
                    and not self._in_flight_tasks
                    and not self._pending_staged_admissions
                ):
                    break

            ready = poller.poll(STORE_LOOP_POLL_TIMEOUT_MS)

            signaled_adapters: dict[StorePhase, set[int]] = {
                phase: set() for phase in StorePhase
            }
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                # Consume the notifier value
                try:
                    consume_fd(fd)
                except (OSError, BlockingIOError):
                    pass

                try:
                    if fd == listener_efd:
                        keys = self._listener.pop_pending_keys()
                        if keys and not self._stop_flag.is_set():
                            self._process_new_keys(keys)
                    else:
                        adapter_idx = self._efd_to_adapter_index.get(fd)
                        if adapter_idx is not None:
                            signaled_adapters[StorePhase.L2_STORE].add(adapter_idx)
                except Exception:
                    logger.exception(
                        "Unexpected error in store loop while processing fd %d",
                        fd,
                    )

            if any(signaled_adapters.values()):
                try:
                    self._drain_l2_store_completions(
                        signaled_adapters[StorePhase.L2_STORE]
                    )
                except Exception:
                    logger.exception("Unexpected error draining L2 store completions")
                for task_key, task in list(self._in_flight_tasks.items()):
                    try:
                        self._advance_request(task_key, task)
                    except Exception:
                        logger.exception(
                            "Unexpected error advancing in-flight store task "
                            "(adapter %d, task %d)",
                            task_key[0],
                            task_key[1],
                        )

            # Finalize any draining adapter no longer have any in-flight
            # tasks.
            self._finalize_drained_adapters(poller)

            try:
                self._deferred_l1_deletion_count = (
                    self._l1_manager.reclaim_discarded_read_operations()
                )
            except Exception:
                logger.exception("Unexpected error retrying deferred L1 deletions")

    def _apply_pending_adapter_ops(self, poller: "select.poll") -> None:
        """Apply queued add/remove ops on the store loop thread."""
        with self._adapter_ops_lock:
            ops = self._pending_adapter_ops
            self._pending_adapter_ops = []
            for op in ops:
                if isinstance(op, AddAdapterOp):
                    self._l2_adapters[op.adapter_id] = op.adapter
                    self._adapter_descriptors[op.adapter_id] = op.descriptor
                    efd = op.adapter.get_store_event_fd()
                    self._efd_to_adapter_index[efd] = op.adapter_id
                    poller.register(efd, select.POLLIN)
                    self._staged_adapter_ids.add(op.adapter_id)
                    logger.info("StoreController attached adapter %d", op.adapter_id)
                    op.done.set()
                elif isinstance(op, RemoveAdapterOp):
                    self._staged_adapter_ids.discard(op.adapter_id)
                    if op.adapter_id not in self._l2_adapters:
                        # Already gone (double remove); signal so the caller
                        # doesn't block.
                        op.done.set()
                        continue
                    # Mark draining; routing skips it immediately. The adapter
                    # stays registered so in-flight completions still process.
                    self._draining[op.adapter_id] = op.done
                    logger.info(
                        "StoreController draining adapter %d (no new stores routed)",
                        op.adapter_id,
                    )

    def _adapter_in_use(self, adapter_id: int) -> bool:
        """True if any in-flight store task targets ``adapter_id``."""
        for task_adapter_id, _ in self._in_flight_tasks:
            if task_adapter_id == adapter_id:
                return True
        return any(
            task.adapter_index == adapter_id for task in self._unresolved_submissions
        )

    def _submit_pending_staged_admissions(self) -> None:
        """Drain transferred writes, including batches accepted before stop."""
        with self._adapter_ops_lock:
            tasks = self._pending_staged_admissions
            self._pending_staged_admissions = []
        for task in tasks:
            source = task.source
            assert isinstance(source, StagedAdmission)
            if self._unresolved_submissions:
                self._l1_manager.finish_write_operations(
                    list(source.operations), discard=frozenset(source.operations)
                )
                self._resolve_staged_admission(task, success=False)
            else:
                self._submit_task(task)

    def _finalize_drained_adapters(self, poller: "select.poll") -> None:
        """Detach draining adapters that no longer have in-flight tasks."""
        for adapter_id in list(self._draining):
            if self._adapter_in_use(adapter_id):
                continue
            adapter = self._l2_adapters.pop(adapter_id)
            self._adapter_descriptors.pop(adapter_id, None)
            efd = adapter.get_store_event_fd()
            self._efd_to_adapter_index.pop(efd, None)
            try:
                poller.unregister(efd)
            except (KeyError, OSError):
                pass
            done = self._draining.pop(adapter_id)
            logger.info("StoreController detached adapter %d", adapter_id)
            done.set()

    def _process_new_keys(self, keys: list[ObjectKey]) -> None:
        """
        Process a batch of newly written keys.

        1. Ask the policy which adapters each key should go to.
        2. For each adapter target, acquire allocation operations on L1
           (skip keys that fail; exports are best-effort).
        3. Submit store tasks to L2 adapters.
        4. Track in-flight tasks for later cleanup.

        Args:
            keys (list[ObjectKey]): Keys that finished writing to L1.
        """

        for group in _group_keys_by_shape(keys).values():
            if self._stop_flag.is_set():
                break
            self._submit_store_for_single_shape(group)

    def _submit_store_for_single_shape(self, keys: list[ObjectKey]) -> None:
        """Submit ``keys`` (all same shape) to their target adapters."""
        # Only route to adapters that are live (not draining). Descriptors
        # for draining adapters are kept for in-flight completion handling
        # but excluded here so no new task targets them.
        routing_descriptors = [
            desc
            for adapter_id, desc in self._adapter_descriptors.items()
            if adapter_id not in self._draining
        ]
        plan = self._policy.select_store_targets(keys, routing_descriptors)

        l1_mgr = self._l1_manager

        for adapter_index, target_keys in plan.items():
            if self._stop_flag.is_set():
                break
            if not target_keys:
                continue

            if adapter_index not in self._l2_adapters:
                logger.error(
                    "StorePolicy returned invalid adapter id %d "
                    "(not among attached adapters). Skipping.",
                    adapter_index,
                )
                continue

            read_results = l1_mgr.acquire_read_operations(target_keys)
            if self._stop_flag.is_set():
                l1_mgr.finish_read_operations(
                    [
                        operation
                        for _, operation in read_results.values()
                        if operation is not None
                    ]
                )
                break

            successful_keys = []
            read_operations: list[L1ReadOperation] = []
            not_found_keys: list[ObjectKey] = []
            write_locked_keys: list[ObjectKey] = []
            for key, (err, operation) in read_results.items():
                if err != L1Error.SUCCESS or operation is None:
                    if err == L1Error.KEY_NOT_EXIST:
                        not_found_keys.append(key)
                    elif err == L1Error.KEY_NOT_READABLE:
                        write_locked_keys.append(key)
                    logger.debug(
                        "Skipping key %s for L2 store (adapter %d): %s",
                        key,
                        adapter_index,
                        err,
                    )
                    continue
                successful_keys.append(key)
                read_operations.append(operation)

            # L1 read-failure anomaly reporting: target_keys come from an
            # L1_WRITE_FINISHED notification, so failing to acquire them
            # immediately after means an unexpected eviction or lock race.
            if not_found_keys:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L1_READ_FAILED,
                        metadata={
                            "during": "l2_store",
                            "reason": "not_found",
                            "keys": not_found_keys,
                        },
                    )
                )
            if write_locked_keys:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L1_READ_FAILED,
                        metadata={
                            "during": "l2_store",
                            "reason": "write_locked",
                            "keys": write_locked_keys,
                        },
                    )
                )

            if not successful_keys:
                continue

            self._submit_task(
                InFlightStoreTask(
                    adapter_index,
                    successful_keys,
                    ResidentExport(tuple(read_operations)),
                )
            )

    def _submit_task(self, task: InFlightStoreTask) -> None:
        """Submit either source capability, quarantining ambiguous acceptance."""
        adapter_index = task.adapter_index
        objects = [operation.memory_obj for operation in task.source.operations]
        adapter = self._l2_adapters[adapter_index]
        try:
            task_id = adapter.submit_store_task(task.keys, objects)
        except Exception:
            logger.exception(
                "Unknown store acceptance at adapter %d for %d keys; "
                "source buffers remain pinned until process exit.",
                adapter_index,
                len(task.keys),
            )
            self._unresolved_submissions.append(task)
            with self._adapter_ops_lock:
                self._stop_flag.set()
            self._listener.stop_admission()
            return

        self._in_flight_tasks[(adapter_index, task_id)] = task
        self._status_in_flight_count += 1
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": adapter_index,
                    "task_id": task_id,
                    "l2_name": self._adapter_descriptors[adapter_index].type_name,
                    "key_count": len(task.keys),
                    "total_bytes": sum(obj.get_size() for obj in objects),
                    "key_count_per_salt": Counter(k.cache_salt for k in task.keys),
                },
            )
        )
        logger.debug(
            "Submitted store task %d to adapter %d with %d keys.",
            task_id,
            adapter_index,
            len(task.keys),
        )

    def _drain_l2_store_completions(self, signaled_adapters: set[int]) -> None:
        """Deposit each signaled adapter's L2 outcomes onto their in-flight
        tasks, to be consumed by ``_advance_request``."""
        for adapter_index in signaled_adapters:
            adapter = self._l2_adapters[adapter_index]
            completed = adapter.pop_completed_store_tasks()
            for task_id, result in completed.items():
                task = self._in_flight_tasks.get((adapter_index, task_id))
                if task is None:
                    logger.warning(
                        "Completed store task %d (adapter %d) not found in tracking.",
                        task_id,
                        adapter_index,
                    )
                    continue
                task.l2_store_result = result.is_successful()
                task.l2_bytes_transferred = result.bytes_transferred()

    def _advance_request(
        self,
        task_key: tuple[int, L2TaskId],
        task: InFlightStoreTask,
    ) -> None:
        """State-transition dispatcher. Delegate to ``_finalize_store``
        once the L2 outcome has been recorded by
        ``_drain_l2_store_completions``."""
        if task.l2_store_result is None:
            return
        self._finalize_store(task_key, task)

    def _finalize_store(
        self,
        task_key: tuple[int, L2TaskId],
        task: InFlightStoreTask,
    ) -> None:
        """Release exact allocations, publish completion, apply policy discards
        on success, and remove the tracking entry."""
        adapter_index, task_id = task_key
        l1_mgr = self._l1_manager
        success = task.l2_store_result

        if isinstance(task.source, ResidentExport):
            delete_keys = (
                set(self._policy.select_l1_deletions(task.keys)) if success else set()
            )
            discard = frozenset(
                operation
                for key, operation in zip(
                    task.keys, task.source.operations, strict=True
                )
                if key in delete_keys
            )
            l1_mgr.finish_read_operations(list(task.source.operations), discard=discard)
        else:
            l1_mgr.finish_write_operations(
                list(task.source.operations), discard=frozenset(task.source.operations)
            )
        del self._in_flight_tasks[task_key]
        self._status_in_flight_count -= 1

        l2_name = self._adapter_descriptors[adapter_index].type_name
        completion_meta: dict[str, object] = {
            "adapter_index": adapter_index,
            "task_id": task_id,
            "l2_name": l2_name,
            "bytes_transferred": task.l2_bytes_transferred,
        }
        if success:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        **completion_meta,
                        "succeeded_count": len(task.keys),
                        "failed_count": 0,
                        "key_count_per_salt": Counter(k.cache_salt for k in task.keys),
                    },
                )
            )
            logger.debug(
                "L2 store task %d completed: adapter %d, %d keys.",
                task_id,
                adapter_index,
                len(task.keys),
            )
        else:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        **completion_meta,
                        "succeeded_count": 0,
                        "failed_count": len(task.keys),
                    },
                )
            )
            logger.warning(
                "Store task %d to adapter %d failed for keys: %s",
                task_id,
                adapter_index,
                task.keys,
            )
        self._resolve_staged_admission(task, success=bool(success))

    def _resolve_staged_admission(
        self, task: InFlightStoreTask, *, success: bool
    ) -> None:
        """Resolve observation only after source cleanup, tolerating cancellation."""
        if isinstance(task.source, StagedAdmission):
            completion = task.source.completion
            if completion.set_running_or_notify_cancel():
                keys = frozenset(task.keys)
                completion.set_result(
                    StoreAdmissionResult(
                        admitted_keys=keys if success else frozenset(),
                        failed_keys=frozenset() if success else keys,
                    )
                )
