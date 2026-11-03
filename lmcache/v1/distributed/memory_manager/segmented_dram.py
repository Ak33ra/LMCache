# SPDX-License-Identifier: Apache-2.0
"""Removable CPU DRAM segments for quiescent L1 resize experiments."""

# Standard
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4
import ctypes
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryObj, TensorMemoryObj


def _map_dram(name: str, size: int) -> mmap.mmap:
    """Create a unique Linux memfd, retaining only the mmap-owned descriptor."""
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        os.ftruncate(fd, size)
        return mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
    finally:
        os.close(fd)


class DRAMSegmentState(Enum):
    """A segment admits allocations, drains, or has no remaining mapping."""

    ACTIVE = "active"
    DRAINING = "draining"
    REMOVED = "removed"


@dataclass(frozen=True)
class DRAMSegmentHandle:
    """Manager-specific identity; sequence numbers are never reused."""

    owner: UUID
    sequence: int


@dataclass(frozen=True)
class DRAMSegmentStatus:
    """Physical accounting snapshot, including unavailable draining backing.

    ``mapped_bytes`` is zero for a retired handle. ``used_bytes`` counts aligned
    allocations, not native accesses or resident pages. ``mapping_name`` can be
    matched against Linux ``/proc/self/maps``; it is empty after removal.
    ``base`` is the mapped address, including during a blocked removal, and
    zero after removal. The snapshot does not pin the mapping or its contents.
    """

    handle: DRAMSegmentHandle
    state: DRAMSegmentState
    mapped_bytes: int
    used_bytes: int
    active_allocations: int
    mapping_name: str
    base: int


class _DRAMSegment(TensorMemoryAllocator):
    """One mapping and its allocator, sharing the pool's bookkeeping lock."""

    def __init__(
        self,
        handle: DRAMSegmentHandle,
        size: int,
        align_bytes: int,
        lock: threading.RLock,
    ) -> None:
        self.handle = handle
        self.size = size
        self.pool_lock = lock
        self.state = DRAMSegmentState.ACTIVE
        self.mapping_name = f"lmcache_dram_{handle.owner.hex}_{handle.sequence}"
        super().__init__(
            torch.empty(0, dtype=torch.uint8), align_bytes, init_address_space=size
        )
        self.mapping = _map_dram(self.mapping_name, size)
        # The ctypes export makes mmap.close refuse surviving Python views.
        # torch.frombuffer(mmap) alone does not preserve that protection.
        buffer = (ctypes.c_ubyte * size).from_buffer(self.mapping)
        try:
            self.buffer = torch.frombuffer(buffer, dtype=torch.uint8)
            self.base = self.buffer.data_ptr()
        except (RuntimeError, MemoryError, ValueError):
            del buffer
            self.mapping.close()
            raise

    def free(self, memory_obj: MemoryObj, allocator_type: str | None = None) -> None:
        """Invalidate an allocation without retaining its own mapping view."""
        with self.pool_lock:
            if memory_obj.parent() is not self:
                raise ValueError("memory object belongs to another segment")
            super().free(memory_obj, allocator_type)
            if isinstance(memory_obj, TensorMemoryObj):
                memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        """Free a batch under the same lock as pool allocation and removal."""
        with self.pool_lock:
            for memory_obj in memory_objs:
                self.free(memory_obj, allocator_type)

    def status(self) -> DRAMSegmentStatus:
        """Snapshot this mapped segment while the caller holds the pool lock."""
        return DRAMSegmentStatus(
            self.handle,
            self.state,
            self.size,
            self.total_allocated_size,
            self.num_active_allocations,
            self.mapping_name,
            self.base,
        )


class SegmentedDRAML1MemoryManager:
    """Own page-aligned memfd mappings behind the L1 allocation contract.

    This Linux CPU backend is opt-in through ``L1Manager(memory_manager=...)``.
    It starts empty and does not pin or register memory with a transfer engine.
    Reconfiguration preserves other segments and never relocates allocations.

    Removal and close require caller-proven quiescence: all native reads and
    writes must have completed, including timed-out or cancelled operations.
    Neither zero allocator usage nor expired L1 locks proves that condition.
    Python tensor views additionally prevent unmapping until dropped.
    """

    def __init__(self, *, align_bytes: int = mmap.PAGESIZE) -> None:
        """Create an empty pool with a power-of-two alignment up to one page.

        Args:
            align_bytes: Allocation alignment; segment sizes remain page aligned.

        Raises:
            ValueError: If alignment cannot be guaranteed by page-aligned mmap.
        """
        if (
            align_bytes <= 0
            or align_bytes & (align_bytes - 1)
            or align_bytes > mmap.PAGESIZE
        ):
            raise ValueError("alignment must be a power of two up to page size")
        self._align_bytes = align_bytes
        self._owner = uuid4()
        self._issued = 0
        self._segments: dict[int, _DRAMSegment] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._l1_owner: int | None = None

    def attach_to_l1(self, config: L1MemoryManagerConfig, l1_manager_id: int) -> None:
        """Validate configuration and transfer ownership to one L1 manager.

        Args:
            config: CPU configuration used by the enclosing L1 manager.
            l1_manager_id: Unique ID of that manager.

        Raises:
            ValueError: If the pool already has an owner, or backing options,
                alignment, capacity or usage differ. Ownership stays unchanged.
        """
        with self._lock:
            if self._l1_owner is not None:
                raise ValueError("segmented DRAM is already attached to an L1 manager")
            if (
                self._closed
                or config.use_lazy
                or config.shm_name
                or config.use_hugepages
                or config.mlock
                or config.devdax_path
                or config.align_bytes != self._align_bytes
                or self.get_memory_usage() != (0, config.size_in_bytes)
                or self.get_active_capacity() != config.size_in_bytes
            ):
                raise ValueError(
                    "segmented DRAM requires matching capacity/alignment, no live "
                    "objects or drains, and no lazy/SHM/hugepage/mlock/DAX options"
                )
            self._l1_owner = l1_manager_id

    def add_segment(self, size_in_bytes: int) -> DRAMSegmentStatus:
        """Map exactly the requested bytes and admit subsequent allocations.

        Args:
            size_in_bytes: Positive multiple of the OS page size.

        Returns:
            The active segment's handle, accounting and mapping name.

        Raises:
            ValueError: If size is not positive and page aligned.
            RuntimeError: If close has begun.
            OSError: If the OS cannot create or map the backing.
        """
        if size_in_bytes <= 0 or size_in_bytes % mmap.PAGESIZE:
            raise ValueError("segment size must be a positive multiple of page size")
        with self._lock:
            if self._closed:
                raise RuntimeError("segmented DRAM manager is closed")
            handle = DRAMSegmentHandle(self._owner, self._issued + 1)
            segment = _DRAMSegment(handle, size_in_bytes, self._align_bytes, self._lock)
            self._segments[handle.sequence] = segment
            self._issued = handle.sequence
            return segment.status()

    def begin_drain(self, handle: DRAMSegmentHandle) -> DRAMSegmentStatus:
        """Stop admission to a segment, retaining its mapping and full charge.

        Args:
            handle: Identity returned by this manager's successful add.

        Returns:
            DRAINING status, or REMOVED for an already retired identity.

        Raises:
            KeyError: If the handle belongs elsewhere or was never issued.
        """
        with self._lock:
            segment = self._find_segment(handle)
            if segment is not None:
                segment.state = DRAMSegmentState.DRAINING
            return self.get_segment_status(handle)

    def remove_segment(self, handle: DRAMSegmentHandle) -> DRAMSegmentStatus:
        """Try physical release after the caller has established quiescence.

        Args:
            handle: Previously drained identity from this manager.

        Returns:
            REMOVED after unmap, including retries of a retired handle. DRAINING
            while allocations or Python tensor views retain the full mapping.

        Raises:
            KeyError: If the handle belongs elsewhere or was never issued.
            ValueError: If admission has not first been stopped by begin_drain.

        Native pointers are not tracked here. Caller quiescence is mandatory
        even when used_bytes is zero. A busy result never releases budget.
        """
        with self._lock:
            segment = self._find_segment(handle)
            if segment is None:
                return self.get_segment_status(handle)
            if segment.state is not DRAMSegmentState.DRAINING:
                raise ValueError("begin_drain must precede remove_segment")
            if segment.num_active_allocations:
                return segment.status()
            segment.buffer = torch.empty(0, dtype=torch.uint8)
            try:
                segment.mapping.close()
            except BufferError:
                return segment.status()
            del self._segments[handle.sequence]
            return self.get_segment_status(handle)

    def get_segment_status(self, handle: DRAMSegmentHandle) -> DRAMSegmentStatus:
        """Return mapped accounting or zero bytes for an issued retired handle.

        Args:
            handle: Segment identity issued by this manager.

        Raises:
            KeyError: If the handle belongs elsewhere or was never issued.
        """
        with self._lock:
            segment = self._find_segment(handle)
            if segment is not None:
                return segment.status()
            return DRAMSegmentStatus(handle, DRAMSegmentState.REMOVED, 0, 0, 0, "", 0)

    def get_segment_statuses(self) -> list[DRAMSegmentStatus]:
        """Return all mapped segments, including draining ones, in add order."""
        with self._lock:
            return [segment.status() for segment in self._segments.values()]

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate a whole batch across active segments, rolling back on failure.

        Args:
            layout_desc: Nonempty tensor layout for each object.
            count: Nonnegative number of objects to allocate.

        Returns:
            SUCCESS and all objects, or OUT_OF_MEMORY and an empty list.

        Raises:
            ValueError: If count is negative or the layout has no bytes.
        """
        object_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        if count < 0 or object_bytes <= 0:
            raise ValueError("count must be nonnegative and layout must have bytes")
        with self._lock:
            result: list[MemoryObj] = []
            for segment in self._segments.values():
                if segment.state is not DRAMSegmentState.ACTIVE:
                    continue
                while len(result) < count:
                    if segment.address_manager.get_free_size() < object_bytes:
                        break
                    obj = segment.allocate(layout_desc.shapes, layout_desc.dtypes)
                    if obj is None:
                        break
                    result.append(obj)
                if len(result) == count:
                    break
            if len(result) != count:
                self.free(result)
                return L1Error.OUT_OF_MEMORY, []
            return L1Error.SUCCESS, result

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Free this pool's objects; repeated frees are harmless.

        Args:
            mem_objs: Objects allocated by this manager, including retired ones.

        Returns:
            SUCCESS after invalidating each allocation.

        Raises:
            ValueError: If an object comes from another allocator.
        """
        with self._lock:
            for obj in mem_objs:
                parent = obj.parent()
                if (
                    not isinstance(parent, _DRAMSegment)
                    or parent.handle.owner != self._owner
                ):
                    raise ValueError("memory object belongs to another manager")
                parent.free(obj)
            return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """Return aligned used bytes and exact mapped bytes, including drains."""
        with self._lock:
            return (
                sum(
                    segment.total_allocated_size for segment in self._segments.values()
                ),
                sum(segment.size for segment in self._segments.values()),
            )

    def get_active_capacity(self) -> int:
        """Return capacity accepting allocations; draining mappings are excluded."""
        with self._lock:
            return sum(
                segment.size
                for segment in self._segments.values()
                if segment.state is DRAMSegmentState.ACTIVE
            )

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return DRAM for an object allocated by this CPU-only backend."""
        return L1BackendType.DRAM

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None: removable segments have no stable registration lifetime."""
        return None

    def memory_region_count(self) -> int:
        """Return mapped segment count, including draining regions."""
        with self._lock:
            return len(self._segments)

    def memcheck(self) -> bool:
        """Verify allocator bookkeeping; this does not verify native quiescence."""
        with self._lock:
            return all(segment.memcheck() for segment in self._segments.values())

    def close(self) -> None:
        """Stop admission and release quiescent, unused segments; allow retries.

        Raises:
            RuntimeError: If objects or Python views still retain any mapping.
                Those bytes remain charged and later close calls retry release.

        The caller must first finish all native access and free all allocations.
        """
        with self._lock:
            self._closed = True
            for segment in list(self._segments.values()):
                self.begin_drain(segment.handle)
                self.remove_segment(segment.handle)
            if self._segments:
                raise RuntimeError(
                    "DRAM segments still have allocations or tensor views"
                )

    def _find_segment(self, handle: DRAMSegmentHandle) -> _DRAMSegment | None:
        if handle.owner != self._owner or not 0 < handle.sequence <= self._issued:
            raise KeyError("segment handle was not issued by this manager")
        return self._segments.get(handle.sequence)
