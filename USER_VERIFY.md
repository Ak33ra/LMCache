# Verify the common-arena alternative

This branch contains a standalone local experiment in
[`examples/partitioning_common_arena`](examples/partitioning_common_arena).
It uses LMCache's real CPU allocator with one POSIX shared-memory mapping,
two local quotas, generation-scoped operations, and drain-then-commit resize.
It does not change the MP server or the selected Mooncake Store design.

## Run the native byte and resize check

Use the prepared environment without installing or rebuilding packages:

```bash
cd /scratch/akira/kvc-worktrees/common-arena
export PYTHONPATH="$PWD"
/scratch/akira/lmcache-dev/bin/python \
  -m examples.partitioning_common_arena.verify
```

The default arena is 8 MiB. The runner writes a 1 MiB shared object, makes
an independent private copy, holds a private read during shrink, commits
all-shared after the read ends, and reverses to all-private. It also checks
the backing segment's file size and that close unlinks it.

Read the JSON snapshots in order:

| Stage | Expected observation |
| --- | --- |
| `shared_staged` | Shared used, staged, and pinned are each 1 MiB; the object is unreadable |
| `two_independent_copies` | Allocator usage is 2 MiB, with 1 MiB charged to each partition |
| `resize_blocked_by_private_read` | Private admission limit is zero; both effective quotas remain 4 MiB |
| `all_shared_committed` | Private effective quota is zero, shared is 8 MiB; shared bytes remain readable |
| `all_private_committed` | Private effective quota is 8 MiB and accepts a new allocation |
| `all_object_charges_released` | Allocator usage is zero; the mapping still has the same address and 8 MiB capacity |

The run exits with `PASS: native bytes, independent copies, drain, retry,
stale handles`. The mapping address and name vary between runs. There is one
backing mapping throughout a run. Quota resizing does not free that mapping
or establish memory handoff between independent allocation owners.

## Run failure and lifetime checks

```bash
/scratch/akira/lmcache-dev/bin/python -m pytest --noconftest -q \
  examples/partitioning_common_arena/test_arena.py
```

Expected result: 21 passed. These tests use real CPU objects and allocation
accounting. `--noconftest` excludes unrelated GPU fixtures; no allocator or
buffer is mocked. The tests use 32 KiB arenas and bounded thread barriers.

To inspect the partial-copy and cancellation schedule separately:

```bash
/scratch/akira/lmcache-dev/bin/python -m pytest --noconftest -vv \
  examples/partitioning_common_arena/test_arena.py \
  -k copy_barrier
```

Both cases pause a real CPU copier after its first three bytes. The source
cannot be evicted; the destination cannot be read; resize remains draining;
and close refuses the active operations. Setting the cancellation event
does not release either charge. After the copier returns, the successful
case publishes the copy and the canceled case releases its destination.

Other tests cover incomplete publication, independent eviction in both
directions, duplicate staging losers, stale and foreign handles, failed
copies, full destination quotas, LRU drain, repeated resize tickets, 20
quota moves, native fragmentation failure, zero/full partitions, and the
128-operation bound.

## Reuse this on a new worktree

Python 3.12, PyTorch, LMCache dependencies, and the compiled common extension
are required by the prepared environment. The ignored native artifact in
this worktree is a symlink to the existing build. If it is absent, recreate
only that link:

```bash
ln -s \
  /home/akira/kvc-partitioning/LMCache/lmcache/lmcache_native.cpython-312-x86_64-linux-gnu.so \
  lmcache/lmcache_native.cpython-312-x86_64-linux-gnu.so
```

This command is specific to the existing Python 3.12 build. A different
machine needs its own compatible LMCache CPU installation. `/dev/shm` must
have space for the selected arena. Each run generates a fresh segment name.
Do not clean unrelated shared-memory segments.

## Optional resource observation

This command reports whole-process resource use; it is not a cache or
network throughput benchmark:

```bash
/usr/bin/time -v /scratch/akira/lmcache-dev/bin/python \
  -m examples.partitioning_common_arena.verify --arena-mib 8
```

`--arena-mib` accepts 1 through 64. RSS includes Python, PyTorch, object
metadata, caller payloads, copied read responses, and resident mapping pages.
The `pinned` counter means protected by active operations, not OS page-locked
memory. Do not add it to `used` or compare it directly to RSS.

## What remains unproved

There is no peer or MP traffic, cross-host visibility, automatic shared-first
admission, reuse of two existing `L1Manager` instances, or GPU/RDMA transfer
completion. The callback contract requires synchronous access and forbids
retaining its memoryviews. A hung callback deliberately retains its charges;
the prototype does not solve abandoned-process recovery.

See [design and ownership](examples/partitioning_common_arena/DESIGN.md) and
[recorded results](examples/partitioning_common_arena/RESULTS.md).
