# SPDX-License-Identifier: Apache-2.0
"""Two local retention quotas over one real LMCache CPU allocation."""

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from threading import Event, RLock
from uuid import UUID, uuid4

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_management import TensorMemoryObj

ALIGNMENT = 4096
MAX_ARENA_BYTES = 64 << 20
MAX_OPERATIONS = 128


def copy_bytes(source: memoryview, destination: memoryview) -> None:
    """Copy equal-length scoped CPU views synchronously without retaining them."""
    destination[:] = source


class Partition(Enum):
    """Local quota identity; shared does not imply network exposure."""

    PRIVATE = auto()
    SHARED = auto()


class Access(Enum):
    """Operation rights attached to an allocation generation."""

    READ = auto()
    WRITE = auto()


class Phase(Enum):
    """Only READY entries are visible to readers or eligible for eviction."""

    STAGED = auto()
    WRITTEN = auto()
    READY = auto()


class ResizePhase(Enum):
    """Distinguish blocked draining, committed requests, and stale retries."""

    DRAINING = auto()
    COMMITTED = auto()
    STALE = auto()


@dataclass(frozen=True)
class Allocation:
    """Immutable identity surviving key/address reuse."""

    incarnation: UUID
    generation: int
    partition: Partition
    key: ObjectKey


@dataclass(frozen=True)
class Operation:
    """Unforgeable-by-accident local read/write completion identity."""

    allocation: Allocation
    sequence: int
    access: Access


@dataclass
class Entry:
    """Native object and operation count; owned only by CommonArena."""

    allocation: Allocation
    memory: TensorMemoryObj
    phase: Phase = Phase.STAGED
    operations: set[Operation] = field(default_factory=set)


@dataclass(frozen=True)
class QuotaUsage:
    """Aligned byte accounting for one view; pinned is a subset of used."""

    effective: int
    admission_limit: int
    used: int
    staged: int
    pinned: int
    residents: int


@dataclass(frozen=True)
class Snapshot:
    """The physical backing is counted once, separately from live usage."""

    backing_bytes: int
    backing_address: int
    shm_name: str
    native_used: int
    active_operations: int
    private: QuotaUsage
    shared: QuotaUsage


@dataclass(frozen=True)
class ResizeTicket:
    """Identity and requested split of a single resize operation."""

    incarnation: UUID
    sequence: int
    private_bytes: int


class CommonArena:
    """Own one CPU arena and all private/shared object lifetimes.

    Args:
        total_bytes: Page-aligned backing size, at most 64 MiB.
        private_bytes: Page-aligned private quota; shared gets the remainder.

    Raises:
        ValueError: Budgets are invalid.

    All methods are thread-safe. Operations have no deadline and must be
    explicitly completed. The only exported native views are scoped to the
    synchronous copier callback. The object is not a network cache service.
    """

    def __init__(
        self, total_bytes: int = 8 << 20, private_bytes: int = 4 << 20
    ) -> None:
        if not 0 < total_bytes <= MAX_ARENA_BYTES or total_bytes % ALIGNMENT:
            raise ValueError("arena must be page aligned and within 64 MiB")
        self._validate_target(private_bytes, total_bytes)
        self._lock = RLock()
        self._incarnation = uuid4()
        self._sequence = count(1)
        self._total = total_bytes
        self._shm_name = f"lmcache_l1_pool_experiment_{self._incarnation.hex}"
        self._quotas = {
            Partition.PRIVATE: private_bytes,
            Partition.SHARED: total_bytes - private_bytes,
        }
        self._pool = L1MemoryManager(
            L1MemoryManagerConfig(
                size_in_bytes=total_bytes,
                use_lazy=False,
                shm_name=self._shm_name,
            )
        )
        self._entries: dict[Allocation, Entry] = {}
        self._resident: OrderedDict[tuple[Partition, ObjectKey], Entry] = OrderedDict()
        self._operations: dict[Operation, Entry] = {}
        self._resize: ResizeTicket | None = None
        self._resize_phase = ResizePhase.COMMITTED
        self._closed = False

    def reserve_write(
        self, partition: Partition, key: ObjectKey, size: int
    ) -> Operation:
        """Charge a staged destination and return its writer identity.

        Args:
            partition: Quota to charge.
            key: Original LMCache content identity.
            size: Positive logical byte length.

        Returns:
            Handle for write(), publish(), or discard().

        Raises:
            ValueError: Size is invalid.
            MemoryError: Quota, physical arena, or operation bound is full.
            RuntimeError: The arena is closed.
        """
        if size <= 0 or size > self._total:
            raise ValueError("object size must be positive and fit the arena")
        with self._lock:
            self._check_open()
            self._check_operations()
            charge = (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            if self._used(partition) + charge > self._limit(partition):
                raise MemoryError("partition admission limit reached")
            layout = MemoryLayoutDesc([torch.Size([size])], [torch.uint8])
            error, objects = self._pool.allocate(layout, 1)
            if error != L1Error.SUCCESS:
                raise MemoryError("physical arena is full or fragmented")
            memory = objects[0]
            if not isinstance(memory, TensorMemoryObj):
                self._pool.free(objects)
                raise RuntimeError("CPU arena returned a non-tensor allocation")
            if memory.get_physical_size() != charge:
                self._pool.free(objects)
                raise RuntimeError("native allocation alignment changed")
            allocation = Allocation(
                self._incarnation, next(self._sequence), partition, key
            )
            entry = Entry(allocation, memory)
            self._entries[allocation] = entry
            return self._new_operation(entry, Access.WRITE)

    def write(self, operation: Operation, data: bytes) -> None:
        """Fill a staged object without publishing it.

        Args:
            operation: Live writer returned by reserve_write().
            data: Exact logical object bytes, copied before returning.

        Raises:
            ValueError: Handle is stale/wrong-kind or data length differs.
        """
        with self._lock:
            entry = self._require(operation, Access.WRITE)
            if len(data) != entry.memory.get_size():
                raise ValueError("write length must match the reserved object")
            with entry.memory.byte_array.cast("B") as destination:
                destination[:] = data
            entry.phase = Phase.WRITTEN

    def publish(self, operation: Operation) -> bool:
        """Publish a completely written destination, consuming its handle.

        Args:
            operation: Writer whose complete payload has been written.

        Returns:
            True for first admission. False for a stale completion or a
            duplicate-key loser, whose allocation is freed.

        Raises:
            ValueError: A live handle is not a writer or bytes are incomplete.
        """
        with self._lock:
            if operation not in self._operations:
                return False
            entry = self._require(operation, Access.WRITE)
            if entry.phase != Phase.WRITTEN:
                raise ValueError("cannot publish incomplete bytes")
            self._release(operation)
            key = (entry.allocation.partition, entry.allocation.key)
            if key in self._resident:
                self._free(entry)
                return False
            entry.phase = Phase.READY
            self._resident[key] = entry
            return True

    def discard(self, operation: Operation) -> bool:
        """Abort a caller-owned staged writer after synchronous access ends.

        Args:
            operation: Writer returned by reserve_write().

        Returns:
            True on release, False for repeated/stale completion.

        Raises:
            ValueError: A live handle is not a writer.
        """
        with self._lock:
            if operation not in self._operations:
                return False
            entry = self._require(operation, Access.WRITE)
            self._release(operation)
            self._free(entry)
            return True

    def reserve_read(self, partition: Partition, key: ObjectKey) -> Operation | None:
        """Pin a resident generation and update that partition's LRU order.

        Args:
            partition: View to search; there is no implicit fallback.
            key: Exact LMCache content identity.

        Returns:
            A read handle, or None on a miss, including staged-only objects.

        Raises:
            MemoryError: The simultaneous operation bound is reached.
            RuntimeError: The arena is closed.
        """
        with self._lock:
            self._check_open()
            entry = self._resident.get((partition, key))
            if entry is None:
                return None
            self._check_operations()
            self._resident.move_to_end((partition, key))
            return self._new_operation(entry, Access.READ)

    def read(self, operation: Operation) -> bytes:
        """Return a caller-owned copy of a pinned generation's logical bytes.

        Args:
            operation: Live read handle from reserve_read().

        Returns:
            Independent bytes outside the cache budget.

        Raises:
            ValueError: Handle is stale or not a reader.
        """
        with self._lock:
            entry = self._require(operation, Access.READ)
            with entry.memory.byte_array.cast("B") as source:
                return bytes(source)

    def finish_read(self, operation: Operation) -> bool:
        """Release a read pin; repeated or stale completion returns False.

        Args:
            operation: Read handle whose synchronous access has ended.

        Raises:
            ValueError: A live handle is not a reader.
        """
        with self._lock:
            if operation not in self._operations:
                return False
            self._require(operation, Access.READ)
            self._release(operation)
            return True

    def delete(self, partition: Partition, key: ObjectKey) -> bool:
        """Free an unpinned resident, returning False if missing or active."""
        with self._lock:
            self._check_open()
            entry = self._resident.get((partition, key))
            if entry is None or entry.operations:
                return False
            self._free(entry)
            return True

    def copy(
        self,
        source: Partition,
        destination: Partition,
        key: ObjectKey,
        *,
        copier: Callable[[memoryview, memoryview], None] = copy_bytes,
        cancel: Event | None = None,
    ) -> bool:
        """Copy across views, retaining both buffers until copier returns.

        Args:
            source: Existing copy's quota view.
            destination: Quota view charged for the independent copy.
            key: Content identity retained by both copies.
            copier: Synchronous CPU callback that fills the destination or
                raises. Views must never escape it.
            cancel: Optional cancellation request; cannot stop native access.

        Returns:
            True when the destination is admitted; False for a source miss,
            cancellation after copier completion, or a duplicate loser.

        Raises:
            MemoryError: Destination quota/arena/operation bound is full.
            Exception: Copier errors propagate after both pins are released
                and the incomplete destination is freed.

        There is no timeout-based release. A hung copier keeps its charges
        and prevents close. Use a separate thread to observe blocked drain.
        """
        reader = self.reserve_read(source, key)
        if reader is None:
            return False
        writer: Operation | None = None
        try:
            with self._lock:
                source_entry = self._require(reader, Access.READ)
                writer = self.reserve_write(
                    destination, key, source_entry.memory.get_size()
                )
                destination_entry = self._require(writer, Access.WRITE)
            with (
                source_entry.memory.byte_array.cast("B") as source_bytes,
                destination_entry.memory.byte_array.cast("B") as destination_bytes,
                source_bytes.toreadonly() as readonly_source,
            ):
                copier(readonly_source, destination_bytes)
            if cancel is not None and cancel.is_set():
                return False
            with self._lock:
                destination_entry.phase = Phase.WRITTEN
                return self.publish(writer)
        finally:
            if writer is not None:
                self.discard(writer)
            self.finish_read(reader)

    def request_resize(self, private_bytes: int) -> ResizeTicket:
        """Cap admission for a requested split without granting growth yet.

        Args:
            private_bytes: Page-aligned target within the existing arena.

        Returns:
            Latest ticket; retries of the same target return the same ticket.

        Raises:
            ValueError: Target is invalid.
            RuntimeError: Closed, or a different resize is still draining.
        """
        self._validate_target(private_bytes, self._total)
        with self._lock:
            self._check_open()
            if self._resize is not None:
                if self._resize.private_bytes == private_bytes:
                    return self._resize
                if self._resize_phase == ResizePhase.DRAINING:
                    raise RuntimeError("another resize is still draining")
            self._resize = ResizeTicket(
                self._incarnation, next(self._sequence), private_bytes
            )
            self._resize_phase = ResizePhase.DRAINING
            return self._resize

    def advance_resize(self, ticket: ResizeTicket) -> ResizePhase:
        """Evict eligible LRU residents, then atomically commit both quotas.

        Args:
            ticket: Request returned by request_resize().

        Returns:
            DRAINING while active/staged bytes exceed a target; COMMITTED
            when both fit; STALE for a different request or arena.

        Raises:
            RuntimeError: The arena is closed.

        No bytes change partition identity. Retries may evict more objects
        after operations finish, but cannot grant the same capacity twice.
        """
        with self._lock:
            self._check_open()
            if ticket != self._resize:
                return ResizePhase.STALE
            if self._resize_phase == ResizePhase.COMMITTED:
                return self._resize_phase
            targets = {
                Partition.PRIVATE: ticket.private_bytes,
                Partition.SHARED: self._total - ticket.private_bytes,
            }
            for partition, target in targets.items():
                for entry in list(self._resident.values()):
                    if self._used(partition) <= target:
                        break
                    if entry.allocation.partition == partition and not entry.operations:
                        self._free(entry)
            if all(self._used(partition) <= n for partition, n in targets.items()):
                self._quotas = targets
                self._resize_phase = ResizePhase.COMMITTED
            return self._resize_phase

    def snapshot(self) -> Snapshot:
        """Return charged/pinned/staged bytes and the actual native arena.

        Raises:
            RuntimeError: Closed, or logical accounting differs from native.
        """
        with self._lock:
            self._check_open()
            native_used, total = self._pool.get_memory_usage()
            private = self._usage(Partition.PRIVATE)
            shared = self._usage(Partition.SHARED)
            if native_used != private.used + shared.used:
                raise RuntimeError("quota/native accounting diverged")
            return Snapshot(
                total,
                self._pool.get_l1_memory_desc().ptr,
                self._shm_name,
                native_used,
                len(self._operations),
                private,
                shared,
            )

    def close(self) -> None:
        """Free all residents and backing once, refusing active operations.

        Raises:
            RuntimeError: Readers, writers, or a copy still own native access.
        """
        with self._lock:
            if self._closed:
                return
            if self._operations:
                raise RuntimeError("active operations prevent arena close")
            for entry in list(self._entries.values()):
                self._free(entry)
            self._pool.close()
            self._closed = True

    @staticmethod
    def _validate_target(private_bytes: int, total_bytes: int) -> None:
        if not 0 <= private_bytes <= total_bytes or private_bytes % ALIGNMENT:
            raise ValueError("private target must be page aligned within the arena")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("arena is closed")

    def _check_operations(self) -> None:
        if len(self._operations) >= MAX_OPERATIONS:
            raise MemoryError("simultaneous operation limit reached")

    def _new_operation(self, entry: Entry, access: Access) -> Operation:
        operation = Operation(entry.allocation, next(self._sequence), access)
        self._operations[operation] = entry
        entry.operations.add(operation)
        return operation

    def _require(self, operation: Operation, access: Access) -> Entry:
        entry = self._operations.get(operation)
        if entry is None or operation.access != access:
            raise ValueError("stale or wrong-kind operation")
        return entry

    def _release(self, operation: Operation) -> None:
        entry = self._operations.pop(operation)
        entry.operations.remove(operation)

    def _free(self, entry: Entry) -> None:
        self._pool.free([entry.memory])
        del self._entries[entry.allocation]
        if entry.phase == Phase.READY:
            del self._resident[(entry.allocation.partition, entry.allocation.key)]

    def _used(self, partition: Partition) -> int:
        return sum(
            entry.memory.get_physical_size()
            for entry in self._entries.values()
            if entry.allocation.partition == partition
        )

    def _limit(self, partition: Partition) -> int:
        quota = self._quotas[partition]
        if self._resize is None or self._resize_phase != ResizePhase.DRAINING:
            return quota
        private = self._resize.private_bytes
        requested = private if partition == Partition.PRIVATE else self._total - private
        return min(quota, requested)

    def _usage(self, partition: Partition) -> QuotaUsage:
        entries = [
            entry
            for entry in self._entries.values()
            if entry.allocation.partition == partition
        ]
        return QuotaUsage(
            self._quotas[partition],
            self._limit(partition),
            sum(entry.memory.get_physical_size() for entry in entries),
            sum(
                entry.memory.get_physical_size()
                for entry in entries
                if entry.phase != Phase.READY
            ),
            sum(
                entry.memory.get_physical_size()
                for entry in entries
                if entry.operations
            ),
            sum(entry.phase == Phase.READY for entry in entries),
        )
