# SPDX-License-Identifier: Apache-2.0
"""
Managing objects and memory for L1 cache
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
import threading
import weakref

# First Party
from lmcache.lmcache_native import TTLLock
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import L1ManagerConfig, get_configured_capacity_bytes
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.internal_api import (
    L1AdmissionLimits,
    L1AdmissionState,
    L1AdmissionUsage,
    L1ManagerInterface,
    L1ManagerListener,
    L1ObjectMeta,
    L1OperationResult,
    L1ReadOperation,
    L1ReadOperationResult,
    L1WriteOperation,
    L1WriteOperationResult,
    L1WriteRetention,
)
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1ManagerProtocol,
    L1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    SegmentedDRAML1MemoryManager,
)
from lmcache.v1.gpu_connector.gds_context import (
    close_l1_gds_context,
    initialize_l1_gds_context,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
    DevDaxRemoveMode,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)
_l1_manager_ids = count()

# Upper bound for the count parameter in reserve_read / finish_read
# to prevent a single call from holding the global lock for too long.
MAX_READ_LOCK_COUNT = 128


def next_l1_manager_id() -> int:
    """Return a fresh process-local L1 identity.

    Every L1 binding draws from this counter, so the identity doubles as
    the owner tag stamped on memory objects and stays unique across bindings.
    """
    return next(_l1_manager_ids)


def validate_read_locks(read_locks: int) -> int:
    """Validate and clamp a per-key read-lock count.

    Args:
        read_locks: Total read locks to take or release per key.

    Returns:
        Clamped value in [1, MAX_READ_LOCK_COUNT].
    """
    if read_locks < 1:
        logger.warning(
            "L1Manager: read_locks=%d is invalid, clamping to 1",
            read_locks,
        )
        return 1
    if read_locks > MAX_READ_LOCK_COUNT:
        logger.warning(
            "L1Manager: read_locks=%d exceeds limit=%d, clamping",
            read_locks,
            MAX_READ_LOCK_COUNT,
        )
        return MAX_READ_LOCK_COUNT
    return read_locks


# Internal classes and helper functions
@dataclass
class L1ObjectState:
    """
    The internal state of an object in L1 cache
    """

    memory_obj: MemoryObj
    """ The memory object stored in L1 cache. """

    write_lock: TTLLock
    """ The writer's reservation; held while the object is staged. """

    read_lock: TTLLock
    """ The read lock with TTL for the object. """

    retention: L1WriteRetention
    """Selected RETAIN or TEMPORARY lifetime; never PREFER_RETAIN."""

    write_operation: bool = False
    """An explicit writer owns this staging allocation regardless of TTL."""

    read_operations: int = 0
    """Explicit readers whose lifetime does not expire with a reservation TTL."""

    @property
    def is_temporary(self) -> bool:
        """Whether the allocation should be freed after its readers finish."""
        return self.retention is L1WriteRetention.TEMPORARY


@dataclass
class ActiveReadOperation:
    """Manager-owned binding between a capability and its exact allocation."""

    key: ObjectKey
    state: L1ObjectState
    handle: L1ReadOperation


@dataclass
class ActiveWriteOperation:
    """Bind the original writer capability to its exact staging entry."""

    key: ObjectKey
    tag: str
    state: L1ObjectState
    handle: L1WriteOperation


def l1_mgr_synchronized(func):
    """
    Decorator to mark L1Manager methods as thread-safe
    """

    def wrapper(self: "L1Manager", *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def _l1_usage_ratio_or_zero(target: "L1Manager | None") -> float:
    """Return ``target.get_memory_usage()`` as a 0.0-1.0 ratio.

    Returns 0.0 when ``target`` is None or ``total_bytes`` is zero so the
    observable-gauge callback never raises during scrape.
    """
    if target is None:
        return 0.0
    used, total = target.get_memory_usage()
    if total <= 0:
        return 0.0
    return used / total


def _l1_staging_bytes_or_zero(target: "L1Manager | None") -> int:
    """Return ``target.get_staging_memory_usage()``, or 0 without a target."""
    if target is None:
        return 0
    return target.get_staging_memory_usage()


# Main classes


class L1Manager(L1ManagerInterface):
    """
    Object lifecycle state machine for L1 cache

    A write creates a *staging object* owned by the writer's ``tag``. Staging
    objects are kept apart from the resident objects: readers of the key do
    not see them, and writers with different tags may stage the same key at
    the same time.

          +--------+
          |  None  | <---------------------------------------+
          +--------+                                         |
            |   ^                                            |
            |   | (write lock expired: evictable)            | delete()
            |   |                                            |
    reserve |   |                                            |
    write() |   |                                            |
    (key,   v   |                                            |
     tag) +--------------+           +-----------+           |
          | write_locked |  finish_  |           |-----------+
          | staging      |---------->|   ready   |
          | (key, tag)   |  write()  |           |---------------+
          +--------------+ (admit or +-----------+               |
            |               discard)       |                     |
            |                              | reserve_read()      | finish_read()
            | finish_write_and_            |                     | (if count becomes 0)
            | delete() -> None             v                     |
            |                      +-----------------+           |
            |                      |   read_locked   |-----------+
            |                      |   (count = 1)   |
            |                      +-----------------+
            |                            |     ^
            |             reserve_read() |     | finish_read()
            |                            v     |
            |                      +-----------------+
            |                      |   read_locked   |
            |                      |   (count = 2)   |
            |                      +-----------------+
            |                            |     ^
            |             reserve_read() |     | finish_read()
            |                            v     |
            v                          (...)  (...)
          (freed)                  (Higher Counts)

    Each operation on a list of keys holds one manager lock. Bounded write
    admission may refuse individual keys in deterministic input order.
    """

    # OTel registers each gauge once; its callback snapshots all live managers.
    _gauge_registered: bool = False
    _gauge_lock = threading.Lock()
    _gauge_targets: weakref.WeakValueDictionary[int, "L1Manager"] = (
        weakref.WeakValueDictionary()
    )

    def __init__(
        self,
        config: L1ManagerConfig,
        *,
        memory_manager: SegmentedDRAML1MemoryManager | None = None,
        admission_limits: L1AdmissionLimits | None = None,
    ) -> None:
        """Construct L1, optionally owning an experimental segmented CPU pool.

        Args:
            config: L1 backend and object-lifetime configuration.
            memory_manager: Unused segmented DRAM pool with capacity and
                alignment matching config. Ownership transfers to this L1.
                Reconfiguration requires caller-proven native quiescence.
            admission_limits: Optional retained/temporary allocation quotas.
                Limits include aligned staging bytes and pinned discards.

        Raises:
            ValueError: If injected backing contradicts the configuration.
        """
        self._l1_manager_id = next_l1_manager_id()
        if memory_manager is not None:
            if config.gds_l1_config is not None:
                raise ValueError("segmented DRAM cannot back a GDS configuration")
        self._config = config
        self._lock = threading.Lock()
        self._admission_limits = admission_limits
        self._admission_state = L1AdmissionState.OPEN

        # Resident objects: readable, never write-locked.
        self._objects: dict[ObjectKey, L1ObjectState] = {}
        self._write_operation_ids = count()
        self._write_operations: dict[int, ActiveWriteOperation] = {}
        self._read_operation_ids = count()
        self._read_operations: dict[int, ActiveReadOperation] = {}
        self._discarded_reads: dict[ObjectKey, L1ObjectState] = {}
        # Staging objects: key -> writer tag -> write-locked object that is
        # invisible to readers until it is admitted by finish_write.
        self._staging: dict[ObjectKey, dict[str, L1ObjectState]] = {}
        # Bytes held by staging objects (kept in sync with ``_staging``).
        self._staging_bytes: int = 0

        # GDS, Device-DAX, and CPU L1 are mutually exclusive tiers. Each tier
        # owns its backing allocator instead of branching inside the CPU path.
        self._memory_manager: L1ManagerProtocol
        if memory_manager is not None:
            self._memory_manager = memory_manager
        elif config.gds_l1_config is not None:
            if config.memory_config.devdax_path:
                raise ValueError("GDS and Device-DAX require separate L1 managers")
            self._memory_manager = GDSL1MemoryManager(config.gds_l1_config)
            initialize_l1_gds_context(self._l1_manager_id, config.gds_l1_config)
            logger.info("L1Manager: GDS L1 tier enabled; CPU pinned-DRAM L1 disabled")
        elif config.memory_config.devdax_path:
            self._memory_manager = DevDaxL1MemoryManager(config.memory_config)
            logger.info("L1Manager: Device-DAX L1 tier enabled; CPU-only L1 disabled")
        else:
            self._memory_manager = L1MemoryManager(config.memory_config)

        # ``use_hugepages`` only ever backs a DRAM L1 pool, so it is a no-op
        # for the device-backed GDS tier and for a pure (non-hybrid)
        # Device-DAX arena.
        l1_cfg = config.memory_config
        if l1_cfg.use_hugepages:
            if config.gds_l1_config is not None:
                logger.warning(
                    "l1-use-hugepages is a no-op for the GDS L1 tier; "
                    "hugepages are ignored."
                )
            elif l1_cfg.devdax_path and not l1_cfg.devdax_size_in_bytes:
                logger.warning(
                    "l1-use-hugepages is a no-op for a pure Device-DAX L1 "
                    "arena (no local DRAM pool)."
                )

        # CPU and GDS capacity is fixed at boot. Device-DAX overlays its entry
        # from the live arena pool because devices can be added or drained.
        self._boot_capacity_bytes_by_backend = get_configured_capacity_bytes(config)
        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds

        self._registered_listeners: list[L1ManagerListener] = []

        self._event_bus = get_event_bus()

        with L1Manager._gauge_lock:
            if not L1Manager._gauge_registered:
                L1Manager._gauge_registered = True
                register_gauge(
                    "lmcache.l1_manager",
                    "lmcache_mp.l1_memory_usage_bytes",
                    "Bytes currently held in L1 cache",
                    lambda: L1Manager._observations(lambda m: m.get_memory_usage()[0]),
                )
                register_gauge(
                    "lmcache.l1_manager",
                    "lmcache_mp.l1_usage_ratio",
                    "L1 used/total ratio (0.0–1.0)",
                    lambda: L1Manager._observations(_l1_usage_ratio_or_zero),
                )
                register_gauge(
                    "lmcache.l1_manager",
                    "lmcache_mp.l1_staging_bytes",
                    "Bytes held by L1 staging objects (write-reserved, not admitted)",
                    lambda: L1Manager._observations(_l1_staging_bytes_or_zero),
                )
            if memory_manager is not None:
                memory_manager.attach_to_l1(config.memory_config, self._l1_manager_id)
            L1Manager._gauge_targets[self._l1_manager_id] = self

    @property
    def l1_manager_id(self) -> int:
        """Return this manager's stable process-local memory-object owner tag."""
        return self._l1_manager_id

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for L1Manager events.

        Args:
            listener: The listener to register.
        """
        with self._lock:
            self._registered_listeners.append(listener)

    @l1_mgr_synchronized
    def reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys.

        Args:
            keys: The list of object keys to reserve
                read access for.
            read_locks: Total read locks acquired per key --
                one per worker that consumes a read lock
                for the same key (e.g. MLA models with
                TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple
            of (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.

        Note:
            Staging objects are never readable; a key that is only
            being written is reported as ``KEY_NOT_EXIST``.
            QUIESCING refuses new readers with ``KEY_NOT_READABLE``.
        """
        if self._admission_state is L1AdmissionState.QUIESCING:
            return {key: (L1Error.KEY_NOT_READABLE, None) for key in keys}
        total = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            # TODO(perf): support a count argument in
            # TTLLock.lock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        self._report_read_reserved(successful_keys)
        return ret

    @l1_mgr_synchronized
    def acquire_read_operations(
        self, keys: list[ObjectKey], *, consume_read_lock: bool = False
    ) -> dict[ObjectKey, L1ReadOperationResult]:
        """Pin resident allocations without a TTL until explicit completion.

        Args:
            keys: Resident keys to read; duplicate keys acquire one handle.
            consume_read_lock: Require and consume one current TTL read lock
                while pinning the allocation, for prefetched serving reads.

        Returns:
            Per-key SUCCESS and a handle, KEY_NOT_EXIST for a missing key, or
            KEY_IS_LOCKED at the per-allocation limit of 128 active operations.
            KEY_NOT_READABLE if the required TTL reservation is absent/expired.
            QUIESCING also returns KEY_NOT_READABLE without consuming a lock.
            The caller must release each handle after all buffer access ends.
        """
        if self._admission_state is L1AdmissionState.QUIESCING:
            return {key: (L1Error.KEY_NOT_READABLE, None) for key in keys}
        results: dict[ObjectKey, L1ReadOperationResult] = {}
        acquired: list[ObjectKey] = []
        for key in dict.fromkeys(keys):
            entry = self._objects.get(key)
            if entry is None:
                results[key] = (L1Error.KEY_NOT_EXIST, None)
            elif consume_read_lock and not entry.read_lock.is_locked():
                results[key] = (L1Error.KEY_NOT_READABLE, None)
            elif entry.read_operations >= MAX_READ_LOCK_COUNT:
                results[key] = (L1Error.KEY_IS_LOCKED, None)
            else:
                operation_id = next(self._read_operation_ids)
                handle = L1ReadOperation(
                    self._l1_manager_id, operation_id, entry.memory_obj
                )
                self._read_operations[operation_id] = ActiveReadOperation(
                    key, entry, handle
                )
                entry.read_operations += 1
                if consume_read_lock:
                    entry.read_lock.unlock()
                results[key] = (L1Error.SUCCESS, handle)
                acquired.append(key)
        if not consume_read_lock:
            self._report_read_reserved(acquired)
        return results

    @l1_mgr_synchronized
    def finish_read_operations(
        self,
        operations: list[L1ReadOperation],
        *,
        discard: frozenset[L1ReadOperation] = frozenset(),
    ) -> None:
        """Complete a validated batch, optionally discarding its exact allocations.

        Args:
            operations: Original handles issued by this manager, each once.
            discard: Subset whose allocations should be discarded after every
                operation and legacy TTL reader releases them.

        Raises:
            ValueError: A handle is foreign, copied, already completed, repeated,
                or discard is not a subset. The entire batch remains unchanged.

        Note:
            Call only after buffer access has ended. Cancellation and timeout
            alone do not establish that condition. No completion tombstones are
            kept; live handle identity binds release to its original allocation.
        """
        if len(set(operations)) != len(operations) or not discard.issubset(operations):
            raise ValueError("Repeated operation or discard outside completion batch")
        active: list[ActiveReadOperation] = []
        for handle in operations:
            tracked = self._read_operations.get(handle.operation_id)
            if tracked is None or tracked.handle is not handle:
                raise ValueError("Read operation is not active on this manager")
            active.append(tracked)
        completed: list[ObjectKey] = []
        for operation in active:
            del self._read_operations[operation.handle.operation_id]
            operation.state.read_operations -= 1
            if operation.handle in discard or operation.state.is_temporary:
                self._discarded_reads[operation.key] = operation.state
            completed.append(operation.key)
        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(completed)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"l1_tag": self._config.tag, "keys": completed},
            )
        )
        self._reclaim_discarded_reads()

    @l1_mgr_synchronized
    def reclaim_discarded_read_operations(self) -> int:
        """Reclaim exact discarded allocations after their legacy readers expire.

        Returns:
            The number of discarded allocations still held by readers.
        """
        self._reclaim_discarded_reads()
        return len(self._discarded_reads)

    @l1_mgr_synchronized
    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Unsafe read the read-locked objects without adding new read locks.

        This method does not acquire read locks. Therefore, the caller need
        to make sure the `unsafe_read` is called between `reserve_read` and
        `finish_read` calls.

        Args:
            keys: The list of object keys to read.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key is not readable (in this case, not read-locked).
        """
        ret: dict[ObjectKey, L1OperationResult] = {}

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.read_lock.is_locked():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        return ret

    @l1_mgr_synchronized
    def finish_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys.

        Will delete the object if it is temporary and read
        count reaches zero.

        Args:
            keys: The list of object keys to finish read
                access for.
            read_locks: Read locks to release per key.  A caller
                releasing only its own read lock passes 1 (the
                default); the reservation owner releasing the
                whole reservation passes the ``reserve_read``
                total.

        Returns:
            A dictionary mapping each object key to an
            L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not read-locked, which
                means the reader may read inconsistent data.
        """
        total = validate_read_locks(read_locks)
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                logger.warning(
                    "L1Manager: finish read on non-existing key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if not entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on non-read-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            # TODO(perf): support a count argument in
            # TTLLock.unlock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.unlock()
            if (
                (entry.is_temporary or key in self._discarded_reads)
                and not entry.read_lock.is_locked()
                and not entry.read_operations
            ):
                need_to_free.append(entry.memory_obj)
                need_to_free_keys.append(key)
                del self._objects[key]
                self._discarded_reads.pop(key, None)

            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        freed_meta = [self._object_meta(obj) for obj in need_to_free]
        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"l1_tag": self._config.tag, "keys": successful_keys},
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": need_to_free_keys,
                    "meta": freed_meta,
                },
            )
        )

        return ret

    @l1_mgr_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve a staging object for each of the given keys.

        Args:
            keys: The list of object keys to reserve write access for.
            is_temporary: The list of booleans indicating whether each key is
                temporary.
            layout_desc: The memory layout description for the objects to be
                allocated.
            tag: The writer's identity; the same tag must be passed to the
                ``finish_write`` variant that completes the write.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Raises:
            ValueError: If ``keys`` and ``is_temporary`` differ in length.
            RuntimeError: Bounded admission requires explicit write operations.

        Errors:
            KEY_NOT_WRITABLE: The key already has a resident object, or
                ``tag`` already stages the key, or admission is quiescing.
            OUT_OF_MEMORY: Not enough memory to allocate for the object.

        Note:
            A staging object is invisible to readers and to other tags until
            it is admitted. Different tags may stage the same key at the same
            time.
        """
        if self._admission_limits is not None:
            raise RuntimeError("Bounded admission requires reserve_write_operations")
        if self._admission_state is L1AdmissionState.QUIESCING:
            return {key: (L1Error.KEY_NOT_WRITABLE, None) for key in keys}
        return self._reserve_write(keys, is_temporary, layout_desc, tag)

    @l1_mgr_synchronized
    def reserve_write_operations(
        self,
        keys: list[ObjectKey],
        retention: list[L1WriteRetention],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1WriteOperationResult]:
        """Allocate and pin unpublished destinations in one critical section.

        Args:
            keys: Distinct object keys to reserve, in admission order.
            retention: One intent per key. PREFER_RETAIN takes retained quota
                first, then temporary quota. Without limits it means RETAIN.
            layout_desc: Tensor layout shared by the destinations.
            tag: Writer identity, independent from other staging writers.

        Returns:
            Per-key SUCCESS and an allocation-bound handle, OUT_OF_MEMORY when
            quota or physical slots are exhausted, or KEY_NOT_WRITABLE when
            the key/tag is occupied or admission is QUIESCING. Bounded mode
            admits independently in input order. Unbounded mode retains its
            whole-batch physical allocation contract. Returned memory objects
            are borrowed; only finish_write_operations releases ownership.

        Raises:
            ValueError: Keys repeat or retention intents differ in length.
            TypeError: A retention intent is not L1WriteRetention.
        """
        if len(set(keys)) != len(keys):
            raise ValueError("Repeated keys in write operation reservation")
        if len(keys) != len(retention):
            raise ValueError("Keys and retention intents differ in length")
        if any(not isinstance(intent, L1WriteRetention) for intent in retention):
            raise TypeError("Retention intents must be L1WriteRetention")
        if self._admission_state is L1AdmissionState.QUIESCING:
            return {key: (L1Error.KEY_NOT_WRITABLE, None) for key in keys}
        if self._admission_limits is None:
            reserved = self._reserve_write(
                keys,
                [intent is L1WriteRetention.TEMPORARY for intent in retention],
                layout_desc,
                tag,
            )
        else:
            reserved = self._reserve_bounded_write(keys, retention, layout_desc, tag)
        results: dict[ObjectKey, L1WriteOperationResult] = {}
        for key, (error, obj) in reserved.items():
            if obj is None:
                results[key] = (error, None)
                continue
            state = self._staging[key][tag]
            operation_id = next(self._write_operation_ids)
            handle = L1WriteOperation(self._l1_manager_id, operation_id, obj)
            state.write_operation = True
            self._write_operations[operation_id] = ActiveWriteOperation(
                key, tag, state, handle
            )
            results[key] = (L1Error.SUCCESS, handle)
        return results

    @l1_mgr_synchronized
    def transfer_write_operations(
        self, operations: dict[ObjectKey, L1WriteOperation]
    ) -> dict[ObjectKey, L1WriteOperation]:
        """Transfer an entire batch by replacing its active capabilities.

        Args:
            operations: Reserved keys mapped to their current original handles.
                All local access must have ended before transfer.

        Returns:
            Replacement handles for the same keys and allocations. Only these
            handles can subsequently transfer or complete the operations.
            The input handles no longer authorize either action.

        Raises:
            ValueError: Any handle is foreign, copied, stale, repeated, or bound
                to another key. The whole batch remains unchanged on rejection.
        """
        active: list[ActiveWriteOperation] = []
        for key, handle in operations.items():
            tracked = self._write_operations.get(handle.operation_id)
            if tracked is None or tracked.handle is not handle:
                raise ValueError("Write operation is not active on this manager")
            if tracked.key != key:
                raise ValueError("Write operation does not match its reserved key")
            active.append(tracked)
        replacements = {
            entry.key: L1WriteOperation(
                self._l1_manager_id, entry.handle.operation_id, entry.handle.memory_obj
            )
            for entry in active
        }
        for entry in active:
            entry.handle = replacements[entry.key]
        return replacements

    @l1_mgr_synchronized
    def finish_write_operations(
        self,
        operations: list[L1WriteOperation],
        *,
        discard: frozenset[L1WriteOperation] = frozenset(),
        read_locks: int = 1,
    ) -> dict[L1WriteOperation, L1OperationResult]:
        """Complete destinations after local buffer access has ended.

        Args:
            operations: Original active handles issued by this manager.
            discard: Subset to free without publishing, including failed loads.
            read_locks: Nonnegative serving TTL locks taken on admission.
                Zero publishes a normal store and notifies storage listeners.

        Returns:
            SUCCESS and the resident object for admitted handles;
            SUCCESS and None for discarded handles. An existing resident wins
            same-key admission and receives any requested serving locks instead.

        Raises:
            ValueError: A handle is stale, copied, foreign, repeated, or discard
                is not a subset, or read_locks is negative. Validation precedes
                every state change.

        Cancellation and timeout alone are not completion. No tombstones remain
        after release. Serving TTL locks do not extend operation ownership.
        """
        if len(set(operations)) != len(operations) or not discard.issubset(operations):
            raise ValueError("Repeated operation or discard outside completion batch")
        active: list[ActiveWriteOperation] = []
        for handle in operations:
            tracked = self._write_operations.get(handle.operation_id)
            if tracked is None or tracked.handle is not handle:
                raise ValueError("Write operation is not active on this manager")
            active.append(tracked)
        if read_locks < 0:
            raise ValueError("read_locks must be nonnegative")
        total = validate_read_locks(read_locks) if read_locks else 0
        results: dict[L1WriteOperation, L1OperationResult] = {}
        freed: list[MemoryObj] = []
        admitted: list[ObjectKey] = []
        metadata: list[L1ObjectMeta] = []
        resident_keys: list[ObjectKey] = []
        gone: list[ObjectKey] = []
        for operation in active:
            handle, key = operation.handle, operation.key
            entry = self._pop_staging(key, operation.tag)
            del self._write_operations[handle.operation_id]
            entry.write_operation = False
            entry.write_lock.unlock()
            if handle in discard:
                freed.append(entry.memory_obj)
                results[handle] = (L1Error.SUCCESS, None)
                if key not in self._objects and key not in self._staging:
                    gone.append(key)
                continue
            resident = self._objects.get(key)
            if resident is None:
                self._objects[key] = entry
                if total or not entry.is_temporary:
                    admitted.append(key)
                    metadata.append(self._object_meta(entry.memory_obj))
            else:
                freed.append(entry.memory_obj)
                entry = resident
                resident_keys.append(key)
            for _ in range(total):
                entry.read_lock.lock()
            results[handle] = (L1Error.SUCCESS, entry.memory_obj)
        self._memory_manager.free(freed)
        self._report_staging_gone(gone)
        if resident_keys and total:
            self._report_read_reserved(resident_keys)
        if admitted:
            for listener in self._registered_listeners:
                if total:
                    listener.on_l1_keys_finish_write_and_reserve_read(admitted)
                else:
                    listener.on_l1_keys_write_finished(admitted)
            self._event_bus.publish(
                Event(
                    event_type=(
                        EventType.L1_WRITE_FINISHED_AND_READ_RESERVED
                        if total
                        else EventType.L1_WRITE_FINISHED
                    ),
                    metadata={
                        "l1_tag": self._config.tag,
                        "keys": admitted,
                        "meta": metadata,
                    },
                )
            )
        return results

    @l1_mgr_synchronized
    def finish_write(
        self,
        keys: list[ObjectKey],
        tag: str = "",
    ) -> dict[ObjectKey, L1Error]:
        """Finish write access for the given keys.

        Admits ``tag``'s staging objects as the resident objects of their
        keys.

        Temporary objects are unlocked normally but do not emit write-finished
        notifications because they are internal staging buffers that must not
        be routed to L2 storage.

        Args:
            keys: The list of object keys to finish write access for.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: ``tag`` stages nothing for the key.
            KEY_IN_WRONG_STATE: The staging object is not write-locked (its
                reservation expired), which means the writer may have
                caused inconsistent data.

        Note:
            If the key became resident before admission, the staging object
            is discarded, the resident object is kept and ``SUCCESS`` is
            still reported: the data is in L1 either way.
        """
        ret: dict[ObjectKey, L1Error] = {}
        notification_keys: list[ObjectKey] = []
        notification_keys_meta: list[L1ObjectMeta] = []
        discarded: list[MemoryObj] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish write")
            ret[key] = err
            if err != L1Error.SUCCESS or entry is None:
                continue
            if key in self._objects:
                logger.debug(
                    "L1Manager: discarding staging object for key %s (tag %r): "
                    "the key is already resident",
                    key,
                    tag,
                )
                discarded.append(entry.memory_obj)
                continue
            self._objects[key] = entry
            if not entry.is_temporary:
                notification_keys.append(key)
                notification_keys_meta.append(self._object_meta(entry.memory_obj))

        self._memory_manager.free(discarded)

        if notification_keys:
            for listener in self._registered_listeners:
                listener.on_l1_keys_write_finished(notification_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_WRITE_FINISHED,
                    metadata={
                        "l1_tag": self._config.tag,
                        "keys": notification_keys,
                        "meta": notification_keys_meta,
                    },
                )
            )
        return ret

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire read lock for the given keys.

        This is used by the prefetch controller after successfully loading
        data from L2 into write-reserved L1 buffers. It transitions the
        object from write-locked to read-locked in a single atomic step,
        preventing a race window where eviction could interfere.

        Args:
            keys: Keys to transition from write-locked to read-locked.
            read_locks: Total read locks acquired per key -- one per TP
                worker that consumes a read lock for the same key
                (e.g. MLA models with TP > 1).
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]); the memory object is the one that
            is now resident and read-locked.

        Errors:
            KEY_NOT_EXIST: ``tag`` stages nothing for the key.
            KEY_IN_WRONG_STATE: The staging object is not write-locked (its
                reservation expired).

        Note:
            Admission follows :meth:`finish_write`. If the key became
            resident before admission, the staging object is discarded and
            the read locks are taken on the resident object, so the caller
            always holds the object that readers see.
        """
        total = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        successful_keys_meta: list[L1ObjectMeta] = []
        resident_keys: list[ObjectKey] = []
        discarded: list[MemoryObj] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish_write_and_reserve_read")
            if err != L1Error.SUCCESS or entry is None:
                ret[key] = (err, None)
                continue
            resident = self._objects.get(key, None)
            if resident is None:
                self._objects[key] = entry
                successful_keys.append(key)
                successful_keys_meta.append(self._object_meta(entry.memory_obj))
            else:
                logger.debug(
                    "L1Manager: discarding staging object for key %s (tag %r): "
                    "the key is already resident; read-locking the resident one",
                    key,
                    tag,
                )
                discarded.append(entry.memory_obj)
                resident_keys.append(key)
                entry = resident
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        self._memory_manager.free(discarded)
        if resident_keys:
            self._report_read_reserved(resident_keys)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": successful_keys,
                    "meta": successful_keys_meta,
                },
            )
        )
        return ret

    @l1_mgr_synchronized
    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from L1 cache.

        Deletes the resident object and reclaims the key's staging objects
        whose write lock expired (all of them when ``force`` is True).

        Args:
            keys: The list of object keys to delete.
            force: When True, delete even a read-locked key and discard its
                live staging objects. This may free memory a concurrent
                legacy reader still uses; explicit operation handles always
                prevent deletion, including when force is True.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IS_LOCKED: The key is read-locked, or a live staging object
                exists for it, or an explicit read operation owns its allocation.
                Explicit operations also return this error when force is True.
        """
        need_to_free: list[MemoryObj] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []
        gone_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None and key not in self._staging:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            self._reclaim_staging(key, force)
            if key in self._staging:
                # A live reservation still pins the key.
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if entry is None:
                # Only expired reservations existed; they are gone now.
                ret[key] = L1Error.SUCCESS
                gone_keys.append(key)
                continue

            if entry.read_operations:
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            locked = entry.read_lock.is_locked()
            if locked and not force:
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if locked:
                logger.warning("L1Manager: force-deleting locked key %s", key)

            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        self._free_and_report_deleted(successful_keys, need_to_free)
        self._report_staging_gone(gone_keys)
        return ret

    @l1_mgr_synchronized
    def finish_write_and_delete(
        self,
        keys: list[ObjectKey],
        tag: str = "",
    ) -> dict[ObjectKey, L1Error]:
        """Atomically finish write access and discard the staging objects.

        Unlock and discard happen in one critical section, so no other
        component can observe or lock the object in between. The staging
        objects never become resident, and no write-finished notification is
        emitted.

        Args:
            keys: The list of object keys whose staging objects to discard.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: ``tag`` stages nothing for the key.
            KEY_IN_WRONG_STATE: The staging object is not write-locked (its
                reservation expired).
        """
        ret: dict[ObjectKey, L1Error] = {}
        discarded: list[MemoryObj] = []
        gone_keys: list[ObjectKey] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish_write_and_delete")
            ret[key] = err
            if err != L1Error.SUCCESS or entry is None:
                continue
            logger.debug(
                "L1Manager: discarding staging object for key %s (tag %r)",
                key,
                tag,
            )
            discarded.append(entry.memory_obj)
            if key not in self._objects and key not in self._staging:
                gone_keys.append(key)

        self._memory_manager.free(discarded)
        self._report_staging_gone(gone_keys)
        return ret

    def touch_keys(self, keys: list[ObjectKey]):
        """Touch the given keys, marking the keys as accessed(retrieved or stored).

        Args:
            keys: The list of object keys to touch.
        """
        for listener in self._registered_listeners:
            listener.on_l1_keys_accessed(keys)
        if self._event_bus.has_subscribers(EventType.L1_KEYS_ACCESSED):
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_ACCESSED,
                    metadata={"l1_tag": self._config.tag, "keys": keys},
                )
            )

    @l1_mgr_synchronized
    def clear(self, force: bool = False) -> None:
        """Clear objects from L1 cache.

        Args:
            force: If True, clear objects with legacy TTL read locks and
                every staging object. This may corrupt legacy serving reads or
                legacy writes. Explicit read and write operations always
                prevent freeing their allocations. If False (default), only clear
                unlocked resident objects and staging objects whose write
                lock expired, keeping read-locked objects and live staging
                objects intact.
        """
        keys_to_clear: list[ObjectKey] = []
        objs_to_free: list[MemoryObj] = []
        locked_count = 0

        for key, entry in list(self._objects.items()):
            if entry.read_operations or (entry.read_lock.is_locked() and not force):
                locked_count += 1
                continue
            keys_to_clear.append(key)
            objs_to_free.append(entry.memory_obj)

        for key in keys_to_clear:
            del self._objects[key]

        if keys_to_clear:
            self._free_and_report_deleted(keys_to_clear, objs_to_free)

        reclaimed_count = 0
        gone_keys: list[ObjectKey] = []
        for key in list(self._staging.keys()):
            reclaimed = self._reclaim_staging(key, force=force)
            reclaimed_count += reclaimed
            if reclaimed and key not in self._staging and key not in self._objects:
                gone_keys.append(key)
        self._report_staging_gone(gone_keys)
        staging_count = sum(len(per_tag) for per_tag in self._staging.values())

        logger.info(
            "L1Manager: cleared %d objects and %d expired staging objects, "
            "%d locked objects and %d staging objects remaining.",
            len(keys_to_clear),
            reclaimed_count,
            locked_count,
            staging_count,
        )

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Check if a key is eligible for eviction (not locked).

        This method does NOT acquire the global L1Manager lock.
        L1Manager.delete() will check again and safely reject a key
        that became locked between the check and the actual deletion.

        Args:
            key: The object key to check.

        Returns:
            True if the key exists and no resident or staging allocation
            has a live lock or operation; False otherwise. Bounded admission
            additionally excludes keys holding only temporary allocations.
        """
        entry = self._objects.get(key, None)
        if self._admission_limits is not None and entry is not None:
            if entry.is_temporary:
                return False
        if entry is not None and (entry.read_operations or entry.read_lock.is_locked()):
            return False
        per_tag = self._staging.get(key, None)
        if per_tag is None:
            return entry is not None
        # Snapshot: this runs without the manager lock.
        staged_objects = list(per_tag.values())
        if self._admission_limits is not None and entry is None:
            if all(staged.is_temporary for staged in staged_objects):
                return False
        return all(
            not staged.write_operation and not staged.write_lock.is_locked()
            for staged in staged_objects
        )

    def get_memory_usage(self) -> tuple[int, int]:
        """Get the current memory usage of L1 cache.

        Returns:
            A tuple of (used_memory_bytes, total_memory_bytes).

        Note:
            In the future, we many want to make a "callback" based mechanism
            via "L1ManagerListener" to notify the memory usage changes.
        """
        return self._memory_manager.get_memory_usage()

    @l1_mgr_synchronized
    def set_admission_limits(self, limits: L1AdmissionLimits) -> None:
        """Set quotas without moving or freeing any existing allocations.

        Args:
            limits: New aligned-allocation quotas. Lowering a quota below
                usage refuses further admission in that class until it drains.

        Raises:
            ValueError: First activation requires an empty L1, so no legacy
                writer or borrowed reference can bypass bounded ownership.
        """
        if self._admission_limits is None and (self._objects or self._staging):
            raise ValueError("First admission-limit activation requires an empty L1")
        self._admission_limits = limits

    @l1_mgr_synchronized
    def get_admission_limits(self) -> L1AdmissionLimits | None:
        """Return configured quotas, or None when bounded admission is disabled."""
        return self._admission_limits

    @l1_mgr_synchronized
    def set_admission_state(self, state: L1AdmissionState) -> None:
        """Pause or resume acquisition of new read/write ownership.

        Args:
            state: QUIESCING refuses new reservations and read operations.
                All completion, discard and deletion paths remain available.
                Existing writes may complete and take their serving read locks.
        """
        self._admission_state = state

    @l1_mgr_synchronized
    def get_admission_state(self) -> L1AdmissionState:
        """Return the current read/write admission state."""
        return self._admission_state

    @l1_mgr_synchronized
    def get_admission_usage(self) -> L1AdmissionUsage:
        """Snapshot aligned owned allocation bytes and optional class limits.

        Includes residents, all staging tags, and pinned deferred discards.
        Bytes are derived from ownership tables, never a separate charge ledger.
        """
        return self._admission_usage()

    @l1_mgr_synchronized
    def get_eviction_pressure(self) -> tuple[int, int]:
        """Return retained used/limit when bounded, otherwise physical used/total."""
        if self._admission_limits is None:
            return self._memory_manager.get_memory_usage()
        return (
            self._admission_usage().retained_allocated_bytes,
            self._admission_limits.retained_bytes,
        )

    @l1_mgr_synchronized
    def reclaim_temporary_objects(self) -> int:
        """Free bounded temporary residents abandoned after serving TTL expiry.

        Returns:
            Number of allocations freed. Disabled mode performs no work.

        Exact read/write operations never expire. They prevent this sweep from
        releasing their allocations even when every serving TTL has expired.
        """
        if self._admission_limits is None:
            return 0
        keys: list[ObjectKey] = []
        objects: list[MemoryObj] = []
        for key, entry in list(self._objects.items()):
            if (
                entry.is_temporary
                and not entry.read_operations
                and not entry.read_lock.is_locked()
            ):
                del self._objects[key]
                keys.append(key)
                objects.append(entry.memory_obj)
        if keys:
            self._free_and_report_deleted(keys, objects)
        return len(keys)

    @l1_mgr_synchronized
    def get_staging_memory_usage(self) -> int:
        """Get the bytes currently held by staging objects.

        Returns:
            The total size in bytes of all write-reserved objects that have
            not been admitted yet.

        Note:
            The value is part of :meth:`get_memory_usage`'s used bytes, not
            in addition to it.
        """
        return self._staging_bytes

    def get_capacity_bytes_by_backend(self) -> dict[L1BackendType, int]:
        """Return the current declared L1 capacity per backing medium.

        Ordinary CPU and GDS retain their boot-configured capacity. For
        Device-DAX and injected segmented DRAM, only active regions count
        as usable capacity; draining regions are excluded immediately.

        Returns:
            A fresh mapping from backing-medium type to usable capacity in
            bytes, with zero-sized media omitted.

        Note:
            Runtime arena and segment state is snapshotted under the backing
            manager's pool lock. A separate usage query may observe an adjacent
            topology if reconfiguration is concurrent.
        """
        capacities = self._boot_capacity_bytes_by_backend.copy()
        manager = self._memory_manager
        if isinstance(manager, SegmentedDRAML1MemoryManager):
            active_bytes = manager.get_active_capacity()
            return {L1BackendType.DRAM: active_bytes} if active_bytes else {}
        if isinstance(manager, DevDaxL1MemoryManager):
            active_bytes = sum(
                status.size_in_bytes
                for status in manager.get_arena_statuses()
                if status.state is DevDaxArenaState.ACTIVE
            )
            if active_bytes > 0:
                capacities[L1BackendType.DEVDAX] = active_bytes
            else:
                capacities.pop(L1BackendType.DEVDAX, None)
        return capacities

    def get_l1_memory_desc(self):
        """Return an L1MemoryDesc describing the underlying L1 memory buffer."""
        return self._memory_manager.get_l1_memory_desc()

    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this L1's primary pool.

        Returns:
            ``GDS`` for a GDS slab, ``CXL`` for a pure Device-DAX tier, and
            ``DRAM`` otherwise, all ``LOCAL``.

        Note:
            A hybrid DRAM + Device-DAX tier reports ``DRAM``, its primary
            pool. Tags are fixed at construction.
        """
        memory_config = self._config.memory_config
        if self._config.gds_l1_config is not None:
            segment = MemorySegment.GDS
        elif memory_config.devdax_path and not memory_config.devdax_size_in_bytes:
            segment = MemorySegment.CXL
        else:
            segment = MemorySegment.DRAM
        return MemoryTags(segment=segment, residency=MemoryResidency.LOCAL)

    def get_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return runtime status for every Device-DAX arena.

        Returns:
            One status per mapped arena, in pool order.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed.
        """
        return self._require_devdax_memory_manager().get_arena_statuses()

    def get_devdax_arena_status(
        self, device_path: str, offset_in_bytes: int = 0
    ) -> DevDaxArenaStatus:
        """Return the status of the Device-DAX arena at ``(device_path, offset)``.

        Args:
            device_path: The exact path used when the arena was added.
            offset_in_bytes: Device offset the arena was added with.

        Returns:
            The arena's current status.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed (409) or no
                arena matches (404).
        """
        return self._require_devdax_memory_manager().get_arena_status(
            device_path, offset_in_bytes
        )

    def owns_range(
        self, device_path: str, offset_in_bytes: int, size_in_bytes: int
    ) -> bool:
        """Return whether L1 maps a range overlapping one on a physical device.

        Args:
            device_path: Candidate device path or alias.
            offset_in_bytes: Start of the candidate range on that device.
            size_in_bytes: Length of the candidate range.

        Returns:
            ``True`` while an overlapping arena is mapped; ``False`` for
            non-DAX L1.
        """
        manager = self._memory_manager
        return isinstance(manager, DevDaxL1MemoryManager) and manager.owns_range(
            device_path, offset_in_bytes, size_in_bytes
        )

    def memory_region_count(self) -> int:
        """Return the number of memory regions backing L1.

        Returns:
            For Device-DAX, the optional DRAM region plus all mapped arenas,
            including draining arenas. For segmented DRAM, all still-mapped
            segments. For other memory managers, 1.
        """
        manager = self._memory_manager
        if isinstance(manager, (DevDaxL1MemoryManager, SegmentedDRAML1MemoryManager)):
            return manager.memory_region_count()
        return 1

    def add_devdax_device(
        self,
        device_path: str,
        size_in_bytes: int,
        offset_in_bytes: int = 0,
    ) -> DevDaxArenaStatus:
        """Add a Device-DAX range to the L1 arena pool.

        Args:
            device_path: Path of the Device-DAX device to map.
            size_in_bytes: Number of bytes to map.
            offset_in_bytes: Byte offset into the device.

        Returns:
            Status of the newly added arena.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed or the request
                cannot be applied.
        """
        return self._require_devdax_memory_manager().add_device(
            device_path, size_in_bytes, offset_in_bytes
        )

    def remove_devdax_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
        offset_in_bytes: int = 0,
    ) -> DevDaxArenaStatus:
        """Remove a Device-DAX arena from the L1 arena pool.

        Args:
            device_path: Path of the mapped Device-DAX device.
            mode: Removal strategy. Only drain mode is currently supported.
            offset_in_bytes: Device offset the arena was added with.

        Returns:
            Status of the arena after the removal request.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed or the request
                cannot be applied.
        """
        return self._require_devdax_memory_manager().remove_device(
            device_path, mode, offset_in_bytes
        )

    @property
    def config(self) -> L1ManagerConfig:
        """Return this manager's backend and lifetime configuration."""
        return self._config

    def close(self) -> None:
        """Free resources only when explicit operations have completed.

        Raises:
            RuntimeError: Read or write operations remain active. Allocations
                stay valid; callers may drain completions and retry closing.
        """
        with self._lock:
            if self._write_operations:
                raise RuntimeError("Cannot close L1 with active write operations")
            if self._read_operations:
                raise RuntimeError("Cannot close L1 with active read operations")
            close_l1_gds_context(self._l1_manager_id)
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            for per_tag in self._staging.values():
                all_memory_objs.extend(staged.memory_obj for staged in per_tag.values())
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            self._discarded_reads.clear()
            self._staging.clear()
            self._staging_bytes = 0

        self._memory_manager.close()
        with L1Manager._gauge_lock:
            L1Manager._gauge_targets.pop(self._l1_manager_id, None)

    # Status reporting
    @l1_mgr_synchronized
    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state.

        ``total_object_count`` covers resident and staging objects;
        ``staging_object_count`` / ``staging_bytes`` report the staging
        subset and ``write_locked_count`` the live reservations among them.
        """
        read_locked = 0
        temporary = 0
        for entry in self._objects.values():
            if entry.read_lock.is_locked():
                read_locked += 1
            if entry.is_temporary:
                temporary += 1
        staging = 0
        write_locked = 0
        for per_tag in self._staging.values():
            staging += len(per_tag)
            for staged in per_tag.values():
                if staged.write_lock.is_locked():
                    write_locked += 1
                if staged.is_temporary:
                    temporary += 1
        used, total = self._memory_manager.get_memory_usage()
        capacities = self.get_capacity_bytes_by_backend()
        # ``memory_total_bytes`` is what the allocator currently backs (the
        # grown heap on the lazy tier). ``memory_configured_bytes`` is the
        # current declared capacity, summed to fit this dict's flat shape.
        return {
            "is_healthy": self._memory_manager.memcheck(),
            "total_object_count": len(self._objects) + staging,
            "write_locked_count": write_locked,
            "read_locked_count": read_locked,
            "active_write_operation_count": len(self._write_operations),
            "active_read_operation_count": len(self._read_operations),
            "pending_discard_count": len(self._discarded_reads),
            "temporary_count": temporary,
            "staging_object_count": staging,
            "staging_bytes": self._staging_bytes,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_configured_bytes": sum(capacities.values()),
            "capacity_bytes_by_backend": {
                backend.value: size for backend, size in capacities.items()
            },
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }

    # Debugging APIs
    @l1_mgr_synchronized
    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Get the internal state of the resident object with the given key.

        Staging objects are not reported here.

        Args:
            key: The object key.

        Returns:
            The L1ObjectState if the object exists, None otherwise.
        """
        return self._objects.get(key, None)

    @l1_mgr_synchronized
    def resident_keys(self, keys: list[ObjectKey]) -> list[bool]:
        """Return membership in the stable resident index without side effects."""
        return [
            key in self._objects and not self._objects[key].is_temporary
            for key in keys
        ]

    @l1_mgr_synchronized
    def memcheck(self) -> bool:
        """Perform memory check for L1 cache."""
        mem_check_result = self._memory_manager.memcheck()

        # Log the locked objects for debugging
        num_read_locked = sum(
            1 for entry in self._objects.values() if entry.read_lock.is_locked()
        )
        num_staging = sum(len(per_tag) for per_tag in self._staging.values())

        logger.info(
            "L1Manager memcheck: total objects = %d, read-locked = %d, "
            "staging = %d (%d bytes)",
            len(self._objects),
            num_read_locked,
            num_staging,
            self._staging_bytes,
        )
        return mem_check_result

    # Private helpers

    def _get_staging(self, key: ObjectKey, tag: str) -> L1ObjectState | None:
        """Return ``tag``'s staging object for ``key``, or None."""
        per_tag = self._staging.get(key, None)
        if per_tag is None:
            return None
        return per_tag.get(tag, None)

    def _put_staging(self, key: ObjectKey, tag: str, entry: L1ObjectState) -> None:
        """Store ``entry`` as ``tag``'s staging object for ``key``."""
        self._staging.setdefault(key, {})[tag] = entry
        self._staging_bytes += entry.memory_obj.get_size()

    def _pop_staging(self, key: ObjectKey, tag: str) -> L1ObjectState:
        """Remove and return ``tag``'s staging object for ``key``.

        The caller must have checked that it exists.
        """
        per_tag = self._staging[key]
        entry = per_tag.pop(tag)
        if not per_tag:
            del self._staging[key]
        self._staging_bytes -= entry.memory_obj.get_size()
        return entry

    def _reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str,
    ) -> dict[ObjectKey, L1OperationResult]:
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"L1Manager.reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )

        need_to_allocate: list[tuple[ObjectKey, bool]] = []
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key, is_temp in zip(keys, is_temporary, strict=True):
            if key in self._objects:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            staged = self._get_staging(key, tag)
            if staged is not None:
                if staged.write_operation or staged.write_lock.is_locked():
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                # The previous reservation expired: hand the buffer over.
                logger.warning(
                    "L1Manager: write reservation on key %s (tag %r) expired; "
                    "handing the buffer to a new writer",
                    key,
                    tag,
                )
                staged.write_lock.lock()
                staged.retention = (
                    L1WriteRetention.TEMPORARY if is_temp else L1WriteRetention.RETAIN
                )
                ret[key] = (L1Error.SUCCESS, staged.memory_obj)
                successful_keys.append(key)
                continue

            need_to_allocate.append((key, is_temp))

        # Early return if no allocation is needed
        if len(need_to_allocate) == 0:
            return ret

        err, allocated_objs = self._memory_manager.allocate(
            layout_desc, len(need_to_allocate)
        )

        if err != L1Error.SUCCESS:
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)

            # Free the memory if partial allocation succeeded
            if allocated_objs:
                self._memory_manager.free(allocated_objs)

        else:
            for (key, is_temp), mem_obj in zip(
                need_to_allocate, allocated_objs, strict=True
            ):
                mem_obj.set_l1_manager(self._l1_manager_id)
                entry = L1ObjectState(
                    memory_obj=mem_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    retention=(
                        L1WriteRetention.TEMPORARY
                        if is_temp
                        else L1WriteRetention.RETAIN
                    ),
                )
                entry.write_lock.lock()
                self._put_staging(key, tag, entry)
                ret[key] = (L1Error.SUCCESS, mem_obj)
                successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_RESERVED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": successful_keys,
                    "tag": tag,
                },
            )
        )
        return ret

    def _admission_usage(self) -> L1AdmissionUsage:
        """Derive physical charges while holding the manager lock."""
        retained = 0
        temporary = 0
        entries = list(self._objects.values())
        entries.extend(
            entry for per_tag in self._staging.values() for entry in per_tag.values()
        )
        for entry in entries:
            if entry.is_temporary:
                temporary += entry.memory_obj.get_physical_size()
            else:
                retained += entry.memory_obj.get_physical_size()
        limits = self._admission_limits
        return L1AdmissionUsage(
            retained,
            temporary,
            limits.retained_bytes if limits is not None else None,
            limits.temporary_bytes if limits is not None else None,
        )

    def _reserve_bounded_write(
        self,
        keys: list[ObjectKey],
        retention: list[L1WriteRetention],
        layout_desc: MemoryLayoutDesc,
        tag: str,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Select final retention and reserve physical slots under the L1 lock."""
        limits = self._admission_limits
        assert limits is not None
        usage = self._admission_usage()
        retained = usage.retained_allocated_bytes
        temporary = usage.temporary_allocated_bytes
        results: dict[ObjectKey, L1OperationResult] = {}
        admitted: list[ObjectKey] = []
        completed = False
        try:
            for key, intent in zip(keys, retention, strict=True):
                if key in self._objects or self._get_staging(key, tag) is not None:
                    results[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                error, objects = self._memory_manager.allocate(layout_desc, 1)
                if error is not L1Error.SUCCESS:
                    self._memory_manager.free(objects)
                    results[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue
                obj = objects[0]
                size = obj.get_physical_size()
                if (
                    intent is not L1WriteRetention.TEMPORARY
                    and retained + size <= limits.retained_bytes
                ):
                    selected = L1WriteRetention.RETAIN
                    retained += size
                elif (
                    intent is not L1WriteRetention.RETAIN
                    and temporary + size <= limits.temporary_bytes
                ):
                    selected = L1WriteRetention.TEMPORARY
                    temporary += size
                else:
                    self._memory_manager.free(objects)
                    results[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue
                obj.set_l1_manager(self._l1_manager_id)
                # Borrowed reference-count cleanup cannot release manager ownership.
                # All manager free paths explicitly invalidate the allocation.
                obj.pin()
                entry = L1ObjectState(
                    obj,
                    TTLLock(self._write_ttl_seconds),
                    TTLLock(self._read_ttl_seconds),
                    selected,
                )
                entry.write_lock.lock()
                self._put_staging(key, tag, entry)
                admitted.append(key)
                results[key] = (L1Error.SUCCESS, obj)
            completed = True
        finally:
            if not completed:
                self._memory_manager.free(
                    [self._pop_staging(key, tag).memory_obj for key in admitted]
                )
        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(admitted)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_RESERVED,
                metadata={"l1_tag": self._config.tag, "keys": admitted, "tag": tag},
            )
        )
        return results

    def _take_staging(
        self,
        key: ObjectKey,
        tag: str,
        op: str,
    ) -> tuple[L1Error, "L1ObjectState | None"]:
        """Unlock and remove ``tag``'s staging object for ``key``.

        Args:
            key: The object key.
            tag: The writer's tag.
            op: Operation name used in the wrong-state warning logs.

        Returns:
            (SUCCESS, entry) with the entry removed from the staging table;
            (KEY_NOT_EXIST, None) when ``tag`` stages nothing for ``key``;
            (KEY_IN_WRONG_STATE, None) when the reservation's write lock
            expired (the object stays staged for eviction to reclaim).
        """
        staged = self._get_staging(key, tag)
        if staged is None:
            return L1Error.KEY_NOT_EXIST, None

        if staged.write_operation:
            return L1Error.KEY_IN_WRONG_STATE, None
        if not staged.write_lock.is_locked():
            logger.warning(
                "L1Manager: %s on key %s (tag %r) whose write reservation "
                "expired, potential inconsistent data might be written",
                op,
                key,
                tag,
            )
            return L1Error.KEY_IN_WRONG_STATE, None

        staged.write_lock.unlock()
        return L1Error.SUCCESS, self._pop_staging(key, tag)

    def _reclaim_staging(self, key: ObjectKey, force: bool) -> int:
        """Free ``key``'s staging objects whose write lock expired.

        Args:
            key: The object key.
            force: When True, free live (still write-locked) staging objects
                too, with a warning.

        Returns:
            The number of staging objects freed.
        """
        per_tag = self._staging.get(key, None)
        if per_tag is None:
            return 0
        freed: list[MemoryObj] = []
        for tag, staged in list(per_tag.items()):
            if staged.write_operation:
                continue
            if staged.write_lock.is_locked():
                if not force:
                    continue
                logger.warning(
                    "L1Manager: force-discarding live staging object %s (tag %r)",
                    key,
                    tag,
                )
            else:
                logger.debug(
                    "L1Manager: reclaiming expired staging object %s (tag %r)",
                    key,
                    tag,
                )
            freed.append(self._pop_staging(key, tag).memory_obj)
        self._memory_manager.free(freed)
        return len(freed)

    def _report_read_reserved(self, keys: list[ObjectKey]) -> None:
        """Notify listeners and the event bus that ``keys`` got read locks."""
        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_RESERVED,
                metadata={"l1_tag": self._config.tag, "keys": keys},
            )
        )

    def _report_staging_gone(self, keys: list[ObjectKey]) -> None:
        """Tell listeners that ``keys`` left L1 without ever becoming resident.

        Note:
            No event is published: nothing readable was evicted.
        """
        if not keys:
            return
        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(keys)

    def _free_and_report_deleted(
        self,
        keys: list[ObjectKey],
        objs: list[MemoryObj],
    ) -> None:
        """Free ``objs`` and report ``keys`` as deleted to listeners and
        the event bus."""
        for key in keys:
            self._discarded_reads.pop(key, None)
        freed_meta = [self._object_meta(obj) for obj in objs]
        self._memory_manager.free(objs)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"l1_tag": self._config.tag, "keys": keys, "meta": freed_meta},
            )
        )

    def _reclaim_discarded_reads(self) -> None:
        """Reclaim retired allocations, never resolving deletion through a new key."""
        keys: list[ObjectKey] = []
        objects: list[MemoryObj] = []
        for key, entry in list(self._discarded_reads.items()):
            if entry.read_operations or entry.read_lock.is_locked():
                continue
            assert self._objects.get(key) is entry
            del self._objects[key]
            keys.append(key)
            objects.append(entry.memory_obj)
        if keys:
            self._free_and_report_deleted(keys, objects)

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Build the listener-facing metadata for one resident object."""
        return L1ObjectMeta(
            size_bytes=memory_obj.get_size(),
            backend=self._memory_manager.get_backend_type(memory_obj),
        )

    def _require_devdax_memory_manager(self) -> DevDaxL1MemoryManager:
        """Return the Device-DAX manager or raise a reconfiguration error."""
        if not isinstance(self._memory_manager, DevDaxL1MemoryManager):
            raise L1ReconfigureError(
                409,
                "L1 is not Device-DAX backed (--l1-devdax-path not set)",
            )
        return self._memory_manager

    @classmethod
    def _observations(
        cls, value: Callable[["L1Manager"], int | float]
    ) -> list[tuple[int | float, dict[str, object]]]:
        with cls._gauge_lock:
            managers = list(cls._gauge_targets.values())
        return [
            (
                value(m),
                {
                    "l1_tag": m.config.tag,
                    "backend": "gds"
                    if m.config.gds_l1_config
                    else "dram+devdax"
                    if m.config.memory_config.devdax_path
                    and m.config.memory_config.devdax_size_in_bytes
                    else "devdax"
                    if m.config.memory_config.devdax_path
                    else "dram",
                },
            )
            for m in managers
        ]
