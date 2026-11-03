# SPDX-License-Identifier: Apache-2.0
"""Single topology owner for bounded private and shared physical backing."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import (
    L1AdmissionLimits,
    L1AdmissionState,
    L1ManagerInterface,
)
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    DRAMSegmentHandle,
    DRAMSegmentState,
    SegmentedDRAML1MemoryManager,
)
from lmcache.v1.distributed.partition_config import PartitionCapacity, PartitionConfig
from lmcache.v1.distributed.partition_controls import (
    CapacityControlCode,
    CapacityControlResult,
    LocalRegistrationState,
    SharedCapacityControls,
    SharedCapacitySnapshot,
)

logger = init_logger(__name__)


class ResizePhase(Enum):
    """Progress of the single serialized topology mutation."""

    QUIESCING = "quiescing"
    RECLAIMING = "reclaiming"
    GROWING = "growing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class ResizeStatus:
    """Idempotent management request and its latest observed progress."""

    request_id: str
    alpha: float
    target: PartitionCapacity
    phase: ResizePhase
    error: str = ""
    sdk_error: int = 0


@dataclass(frozen=True)
class PartitionStatus:
    """Cached ownership; status reads never wait for an SDK control call.

    Registration metadata and allocation usage are subsets of mapped backing.
    Pending growth is reserved before allocation, including during blocked calls.
    ``current`` is the last completed split, never the unfulfilled target.
    """

    current: PartitionCapacity | None
    resize: ResizeStatus | None
    quiesced: bool
    closed: bool
    mapped_bytes: int
    draining_bytes: int
    shared: SharedCapacitySnapshot
    pending_growth_bytes: int
    charged_bytes: int


@dataclass
class _LocalMapping:
    handle: DRAMSegmentHandle
    registration_id: int = 0


class PartitionManager:
    """Own physical mutations of caller-supplied L1 and native controls.

    Args:
        config: Managed physical budget and initial split.
        pool: Segmented pool with exactly one empty permanent staging mapping.
        l1: Persistent L1 backed by pool; all access must use exact operations.
        controls: Exclusive ownership capability of the data-path backend.
        quiesce: Nonblocking predicate that accepted controller work has drained.

    Hold ``admission()`` briefly around request enqueue/reservation. Never hold
    it during SDK work or waiting for completion. Failed startup retains backing:
    callers must keep this owner and L1 alive until ``close()`` returns True.
    The owner never closes L1 or the pool itself.
    """

    def __init__(
        self,
        config: PartitionConfig,
        pool: SegmentedDRAML1MemoryManager,
        l1: L1ManagerInterface,
        controls: SharedCapacityControls,
        *,
        quiesce: Callable[[], bool],
    ) -> None:
        self._config = config
        self._pool = pool
        self._l1 = l1
        self._controls = controls
        self._quiesce = quiesce
        self._gate = threading.RLock()
        self._status_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._locals: list[_LocalMapping] = []
        self._requests: dict[str, ResizeStatus] = {}
        self._job: ResizeStatus | None = None
        self._current: PartitionCapacity | None = None
        self._quiesced = True
        self._closed = False
        self._pending = 0
        self._rebuild = False
        self._native = controls.snapshot()
        self._status: PartitionStatus
        self._publish()

    def initialize(self) -> PartitionStatus:
        """Register staging and realize the initial split before opening ingress.

        Raises:
            ValueError: Initial topology or native reserves violate the contract.
            RuntimeError: A control fails or is busy; retain this owner and
                retry cleanup. Startup makes one attempt, without a worker.
        """
        if self._locals or self._thread is not None:
            raise RuntimeError("Partition initialization already attempted")
        self._l1.set_admission_state(L1AdmissionState.QUIESCING)
        self._l1.set_admission_limits(L1AdmissionLimits(0, 0))
        native = self._native
        segments = self._pool.get_segment_statuses()
        if (
            not native.enabled
            or native.closing
            or native.closed
            or native.scratch_bytes > self._config.sdk_scratch_bytes
            or native.shared_allocated_bytes
            or native.retained_failed_bytes
            or native.pending_shared_bytes
            or native.local_buffers
            or len(segments) != 1
            or segments[0].mapped_bytes != self._config.staging_bytes
            or segments[0].used_bytes
            or segments[0].state is not DRAMSegmentState.ACTIVE
        ):
            raise ValueError(
                "Partition activation requires empty staging-only ownership"
            )
        target = self._config.capacity(native.granularity_bytes)
        self._locals.append(_LocalMapping(segments[0].handle))
        self._job = ResizeStatus("", self._config.alpha, target, ResizePhase.GROWING)
        self._publish()
        try:
            if not self._register(self._locals[0]):
                raise RuntimeError("Initial staging registration is busy")
            if not self._advance():
                raise RuntimeError("Initial topology could not complete")
        except Exception as error:
            self._fail(error)
            raise
        self._thread = threading.Thread(
            target=self._run, name="partition-resize", daemon=True
        )
        self._thread.start()
        return self.get_status()

    @contextmanager
    def admission(self) -> Iterator[bool]:
        """Serialize a short enqueue/reserve with closure of the ingress gate."""
        with self._gate:
            yield not self._quiesced and not self._closed

    def request_resize(self, request_id: str, alpha: float) -> ResizeStatus:
        """Submit or retry one request, rejecting conflicting IDs or targets.

        Failed requests can be retried with the same ID and alpha. Completed
        requests remain idempotent even after a newer request has completed.
        Raises ValueError for invalid targets; RuntimeError for conflicts.
        """
        if not request_id:
            raise ValueError("request_id must be nonempty")
        target = self._config.capacity(self._native.granularity_bytes, alpha=alpha)
        with self._gate:
            if self._closed or self._stop.is_set() or self._current is None:
                raise RuntimeError("Partition manager is not running")
            with self._status_lock:
                prior = self._requests.get(request_id)
                active = self._status.resize
            if prior is not None:
                if prior.alpha != alpha:
                    raise RuntimeError("Request ID already has another target")
                if prior.phase is not ResizePhase.FAILED:
                    return prior
            if active is not None and active.phase not in (
                ResizePhase.COMPLETE,
                ResizePhase.FAILED,
            ):
                raise RuntimeError("Another resize is active")
            self._rebuild = active is not None and active.phase is ResizePhase.FAILED
            self._quiesced = True
            self._l1.set_admission_state(L1AdmissionState.QUIESCING)
            self._job = ResizeStatus(request_id, alpha, target, ResizePhase.QUIESCING)
            self._publish()
            self._wake.set()
            return self._job

    def get_status(self) -> PartitionStatus:
        """Return the immutable cached topology without querying SDK state."""
        with self._status_lock:
            return self._status

    def resize_status(self, request_id: str) -> ResizeStatus:
        """Return a known request's cached status; raise KeyError if unknown."""
        with self._status_lock:
            return self._requests[request_id]

    def close(self) -> bool:
        """Stop resize and attempt cleanup; True permits adapter/L1 destruction.

        Caller must stop data-path controllers first. False preserves caller
        mappings and admission closure; retry this method after outstanding
        operations/views or SDK failures resolve. No worker restarts on retry.
        """
        with self._gate:
            self._quiesced = True
            self._l1.set_admission_state(L1AdmissionState.QUIESCING)
            self._stop.set()
            self._wake.set()
            self._publish()
        if self._thread is not None:
            self._thread.join()
        if self._closed:
            return True
        try:
            if not self._quiesce():
                self._publish()
                return False
            self._l1.clear(force=True)
            if self._pool.get_memory_usage()[0]:
                self._publish()
                return False
            result = self._controls.close()
            self._native = result.snapshot
            native_closed = (
                result.code is CapacityControlCode.SUCCESS
                and result.snapshot.closed
                and not result.snapshot.local_buffers
                and not result.snapshot.shared_allocated_bytes
                and not result.snapshot.retained_failed_bytes
                and not result.snapshot.pending_shared_bytes
            )
            if not native_closed:
                self._publish()
                return False
            self._pending = 0
            for mapping in list(reversed(self._locals)):
                mapping.registration_id = 0
                if not self._remove(mapping):
                    return False
            self._closed = True
            self._publish()
            return self._closed
        except Exception as error:
            self._fail(error)
            return False

    def _publish(self) -> None:
        segments = self._pool.get_segment_statuses()
        mapped = sum(item.mapped_bytes for item in segments)
        pending = max(self._pending, self._native.pending_shared_bytes)
        charged = (
            mapped
            + self._native.scratch_bytes
            + self._native.shared_allocated_bytes
            + self._native.retained_failed_bytes
            + pending
        )
        status = PartitionStatus(
            self._current,
            self._job,
            self._quiesced,
            self._closed,
            mapped,
            sum(
                item.mapped_bytes
                for item in segments
                if item.state is DRAMSegmentState.DRAINING
            ),
            self._native,
            pending,
            charged,
        )
        with self._status_lock:
            self._status = status
            if self._job is not None and self._job.request_id:
                self._requests[self._job.request_id] = self._job

    def _accept(self, result: CapacityControlResult) -> bool:
        self._native = result.snapshot
        self._publish()
        if result.code is CapacityControlCode.BUSY:
            return False
        if result.code is not CapacityControlCode.SUCCESS:
            if self._job is not None:
                self._job = replace(self._job, sdk_error=result.sdk_error)
            raise RuntimeError(
                f"Capacity control: {result.code.value} ({result.sdk_error})"
            )
        return True

    def _register(self, mapping: _LocalMapping) -> bool:
        """Retry the same mapping until its exact registration is confirmed."""
        segment = self._pool.get_segment_status(mapping.handle)
        if any(
            item.registration_id == mapping.registration_id
            and item.base == segment.base
            and item.size == segment.mapped_bytes
            and item.state is LocalRegistrationState.REGISTERED
            for item in self._native.local_buffers
        ):
            return True
        try:
            result = self._controls.register_local(segment.base, segment.mapped_bytes)
        finally:
            self._native = self._controls.snapshot()
            for registration in self._native.local_buffers:
                if registration.base == segment.base:
                    mapping.registration_id = registration.registration_id
            self._publish()
        # A failed registration can retain an exact native cleanup identity.
        mapping.registration_id = result.local_buffer_id or mapping.registration_id
        if not self._accept(result):
            return False
        if not any(
            item.registration_id == mapping.registration_id
            and item.base == segment.base
            and item.size == segment.mapped_bytes
            and item.state is LocalRegistrationState.REGISTERED
            for item in self._native.local_buffers
        ):
            raise RuntimeError("Local registration incomplete")
        return True

    def _remove(self, mapping: _LocalMapping) -> bool:
        segment = self._pool.begin_drain(mapping.handle)
        self._publish()
        if segment.used_bytes:
            return False
        if mapping.registration_id:
            if not self._accept(
                self._controls.unregister_local(mapping.registration_id)
            ):
                return False
            if any(
                item.registration_id == mapping.registration_id
                for item in self._native.local_buffers
            ):
                raise RuntimeError("Native unregister retained the exact registration")
            mapping.registration_id = 0
        removed = self._pool.remove_segment(mapping.handle)
        if removed.state is DRAMSegmentState.REMOVED:
            self._locals.remove(mapping)
        self._publish()
        return removed.state is DRAMSegmentState.REMOVED

    def _reserve(self, size: int) -> None:
        self._native = self._controls.snapshot()
        self._publish()
        if self.get_status().charged_bytes + size > self._config.total_bytes:
            raise RuntimeError("Actual retained backing leaves no growth credit")
        self._pending = size
        self._publish()

    def _advance(self) -> bool:
        assert self._job is not None
        target = self._job.target
        if self._job.phase is ResizePhase.QUIESCING:
            if not self._quiesce():
                return False
            self._job = replace(self._job, phase=ResizePhase.RECLAIMING)
        if self._job.phase is ResizePhase.RECLAIMING:
            count = (
                0
                if self._rebuild
                else target.private_effective_bytes // target.granularity_bytes
            )
            for mapping in self._locals[1 + count :]:
                self._pool.begin_drain(mapping.handle)
            self._l1.clear(force=True)
            self._publish()
            if self._pool.get_memory_usage()[0]:
                return False
            while len(self._locals) - 1 > count:
                if not self._remove(self._locals[-1]):
                    return False
            while self._native.shared_allocated_bytes > target.shared_effective_bytes:
                segment = self._native.shared_segments[-1]
                result = self._controls.release_shared(segment.segment_id)
                if not self._accept(result):
                    return False
                if (
                    segment.segment_id
                    in {s.segment_id for s in self._native.shared_segments}
                    or result.released_shared_bytes < segment.size
                ):
                    raise RuntimeError("Shared release did not confirm physical credit")
            self._rebuild = False
            self._job = replace(self._job, phase=ResizePhase.GROWING)
        for mapping in self._locals:
            if not self._register(mapping):
                return False
        while (
            len(self._locals) - 1
            < target.private_effective_bytes // target.granularity_bytes
        ):
            self._reserve(target.granularity_bytes)
            try:
                segment = self._pool.add_segment(target.granularity_bytes)
            except (OSError, RuntimeError):
                self._pending = 0
                self._publish()
                raise
            mapping = _LocalMapping(segment.handle)
            self._locals.append(mapping)
            self._pending = 0
            self._publish()
            if not self._register(mapping):
                return False
        while self._native.shared_allocated_bytes < target.shared_effective_bytes:
            self._reserve(target.granularity_bytes)
            previous = self._native.shared_allocated_bytes
            result = self._controls.allocate_shared(target.granularity_bytes)
            self._pending = 0
            if not self._accept(result):
                return False
            if (
                self._native.shared_allocated_bytes
                != previous + target.granularity_bytes
            ):
                raise RuntimeError("Native shared growth differs from reserved granule")
        if self._native.retained_failed_bytes or self._native.pending_shared_bytes:
            raise RuntimeError("Native ownership still has unresolved allocations")
        if self.get_status().charged_bytes > self._config.total_bytes:
            raise RuntimeError("Backend exceeded reserved physical budget")
        with self._gate:
            if self._stop.is_set():
                return False
            self._l1.set_admission_limits(
                L1AdmissionLimits(
                    target.private_effective_bytes, self._config.staging_bytes
                )
            )
            self._current = target
            self._job = replace(self._job, phase=ResizePhase.COMPLETE)
            self._quiesced = False
            self._l1.set_admission_state(L1AdmissionState.OPEN)
            self._publish()
        return True

    def _fail(self, error: Exception) -> None:
        logger.exception("Partition topology mutation failed; ownership retained")
        if self._job is not None:
            self._job = replace(self._job, phase=ResizePhase.FAILED, error=str(error))
        self._publish()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            if self._stop.is_set():
                return
            if self._job is None or self._job.phase in (
                ResizePhase.COMPLETE,
                ResizePhase.FAILED,
            ):
                continue
            try:
                self._advance()
            except Exception as error:
                self._fail(error)
