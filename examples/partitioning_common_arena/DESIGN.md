# Common DRAM arena alternative

This branch is a bounded local prototype of the archived common-arena
alternative, independently authorized on 2026-11-03. It does not change the
selected Mooncake Store architecture or wire into the MP server.

The decision is whether a single physical allocation can safely serve two
retention quotas without moving bytes or borrowing capacity prematurely.

## Ownership sketches

| Sketch | Ownership and interfaces | Consequence |
| --- | --- | --- |
| Two existing L1Manager instances | Inject two allocator views; each object manager retains its current tables and TTL locks | Requires changing production construction, close, and every reclamation path before active copies are safe |
| One prototype operation owner | Existing L1MemoryManager owns the physical arena; CommonArena owns private/shared tables, allocation generations, read/write operations, and resize tickets | Small runnable lifetime experiment with no production-path changes; existing L1Manager integration remains unproved |

The second sketch is selected. `L1Manager.__init__` constructs its allocator;
its write/read state uses `TTLLock`. Extending only allocation would leave
expiry-based reclamation unsolved. The existing `L1MemoryManager.allocate`,
`free`, `get_memory_usage`, and `get_l1_memory_desc` supply the real buffer
operations for this experiment.

The pool uses a uniquely named POSIX shared-memory mapping through that
existing allocator. Its `/dev/shm` size is the backing capacity, and close
unlinks it. The ordinary CPU torch fallback was also tried, but its
`_alloc_page_aligned_pinned_view` allocates an extra page beyond the reported
usable arena. The SHM path makes the mapping bound directly inspectable.
Pages become resident when touched. RSS and OS page locking are not inferred
from the mapping size. No other process attaches to this mapping here.

## Data shape and contract

- `Partition` selects a private or shared quota view of the same pool.
- `Allocation` has an arena incarnation, monotonically increasing generation,
  partition, and original LMCache `ObjectKey`.
- `Operation` adds a unique sequence and read/write kind. Completed handles
  disappear from the active table, so repeated or stale completion is a no-op.
- `Entry` owns one native `MemoryObj`. Its phase is staged, written, or ready;
  active operations prevent deletion and draining.
- `ResizeTicket` records one requested private target; the shared target is
  the rest. At most one request drains. Repeating it is idempotent. A stale
  ticket cannot alter a newer request.

Both resident and staged destinations count once at their physical aligned
size. Active bytes are a subset of used bytes. The backing allocation is
constant across every resize. Quota changes do not return DRAM to the OS or
prove physical handoff between separate allocators.
The `pinned` field counts bytes protected by active operations. It does not
report OS page locking or add a second allocation charge.

The public API copies bytes in or out while holding the metadata lock.
`copy()` reserves both buffers, then runs a synchronous copier outside that
lock. Its scoped memoryviews must not escape the callback. A request to
cancel is observed only after the copier returns; it never frees live bytes.
The default copier performs a real CPU copy. There is no TTL-based release.

Resize first caps new admission at the smaller of old and requested quotas,
then evicts unpinned resident objects in each partition's local LRU order.
Staging and active reads keep their charge. Both effective quotas change
atomically only after usage fits. Caller-driven retry makes blocked draining
explicit. Close refuses active operations and frees the arena once.

## Boundaries

The default arena is 8 MiB; the hard maximum is 64 MiB. Read responses are
caller-owned byte copies outside the arena, bounded individually by the
object size. At most 128 simultaneous operations can pin metadata. Runtime,
keys, caller inputs/outputs, and Python metadata are outside cache accounting.

There is no peer discovery, network fetch, global shared eviction, MP worker
integration, GPU/RDMA lifetime proof, automatic admission eviction, or
production shared-first store policy. `shared` here names a logical local
partition. It does not establish cross-process or cross-host visibility.
The parent task evaluates transport separately.
