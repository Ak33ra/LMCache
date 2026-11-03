# Segmented DRAM for quiescent L1 resizing

`SegmentedDRAML1MemoryManager` supplies independently removable Linux DRAM
mappings behind `L1ManagerProtocol`. It is an internal experiment backend.
Ordinary CPU, Device-DAX, and GDS construction keeps its existing behavior.

## Ownership and accounting

Each segment owns a unique memfd, one page-aligned shared mapping, and one
`TensorMemoryAllocator`. Segment sizes must be positive multiples of the OS
page size. Allocation alignment must be a power of two no larger than a page.
There is no extra alignment allocation or hidden mapped page.

`get_memory_usage()` returns aligned object bytes and mapped backing bytes.
Both active and draining segments contribute to mapped backing. A segment
contributes to `get_active_capacity()` only while it admits allocations.
Neither counter measures RSS, page residency, or transfer registration.
memfd pages are faulted on access and can be swapped by the OS. No page-locking
or GPU registration is performed.

Allocations use the first active segment with space. A batch can span
segments but an individual object cannot. A failed batch releases all its
partial allocations. Frees route through the exact owning segment allocator,
including reference-count-triggered frees. They do not remove mappings.

The pool uses one reentrant lock for allocation, free, and topology changes.
That lock protects allocator bookkeeping. It does not track native users of
an address that was handed to another component.

## Explicit segment lifecycle

```text
add_segment(size) -> ACTIVE
begin_drain(handle) -> DRAINING
remove_segment(handle) -> DRAINING while allocations or Python views remain
remove_segment(handle) -> REMOVED after actual mmap.close succeeds
```

`begin_drain` stops admission and retains the whole mapping charge. Callers
delete or drain the objects through their current owner. Removal requires
that the segment has already entered DRAINING. It never migrates objects or
evicts keys. Other segments remain mapped and readable throughout.

A handle contains a manager UUID and an increasing sequence number. The
manager issues a number only after a successful add. It stores live segments
and the highest issued number, so it recognizes retired handles without an
unbounded tombstone collection. Removed-handle retries report REMOVED with
zero mapped bytes. A foreign or unissued handle raises `KeyError`. Sequence
numbers and mapping names are never reused within a manager.

`close()` stops all admission and attempts removal of every segment. Busy
mappings remain charged and `close()` raises `RuntimeError`. Calling it again
after releasing allocations and views retries removal. Adding segments after
close has begun raises `RuntimeError`.

## Native quiescence remains a caller obligation

Before removal or close, the caller must establish that all native reads and
writes have completed, including cancelled or timed-out operations. Zero
allocator usage, a released L1 reservation, or an expired TTL does not prove
this. The backend does not enable concurrent resizing in the MP server.

A ctypes array exports the mmap buffer before torch views are constructed.
This makes a surviving Python tensor view block `mmap.close()`. A direct
`torch.frombuffer(mmap)` does not provide that protection. Logical free clears
the invalidated `MemoryObj`'s view, while an independently held view still
keeps the mapping charged. Retrying removal after the view is dropped can
release the mapping. This Python protection cannot see a native pointer held
without its Python tensor.

## Construction choice

Two shapes were considered. A pool of ordinary `L1MemoryManager` instances
would inherit fixed-size registration descriptors and allocation padding.
The selected shape reuses `TensorMemoryAllocator` while owning each mapping
directly, making physical release and exact accounting observable.

The allocator's address manager is initialized before the memfd is created.
An allocator-initialization failure therefore owns no mapping or descriptor.
An mmap failure closes the newly created descriptor. Runtime failures during
the tensor view conversion close the mapping after dropping its local export.

The design follows Device-DAX's per-region allocation pattern, but its tests
use real memfd-backed DRAM. Device-DAX results are separate evidence.

## L1 integration boundary

The intended caller supplies this backend explicitly through the keyword-only
`L1Manager(..., memory_manager=pool)` argument. The pool must have no live
objects or draining segments. Initial capacity and alignment must match the
L1 config. Lazy allocation, named SHM, hugepages, mlock, Device-DAX, and GDS
configuration conflict with this backend and are rejected.

Ownership transfers to exactly one L1 manager. A second attachment is rejected
even if the pool is empty. The caller retains the pool reference for the
experimental resize methods, and L1 closes the pool when L1 closes.

`get_l1_memory_desc()` always returns `None`. Even a single removable segment
cannot promise the fixed address lifetime that registration consumers expect.
No new CLI, HTTP endpoint, transfer registration, partition controller, or
Store integration is introduced. Public MP configuration cannot select this
backend.

The executable checks and their dated results are in
[USER_VERIFY.md](../../../../../USER_VERIFY.md).
