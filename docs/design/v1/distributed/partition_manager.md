# Physical partition ownership

`PartitionManager` owns one segmented L1 topology and the same backend capacity
controls used for object I/O. StorageManager owns data-path controllers and must
hold `admission()` briefly around enqueue/reservation. Every L1 memory access
must hold an exact operation capability; this is what permits force-clearing
unused TTL reservations during resize. The supplied `quiesce()` predicate must
report that accepted Store/prefetch work has drained, without blocking.

Startup requires one empty permanent staging mapping. It installs zero admission
limits, registers staging, verifies native scratch against its explicit reserve,
and adds whole-granule private and shared backing. Failed initialization does
not undo ownership. Keep the manager and caller L1 alive until `close()` returns
True, including when an enclosing constructor fails.

A single worker drains accepted work, marks excess private mappings draining,
clears L1 objects, waits for exact capabilities, unregisters exact native ranges,
and removes mappings. Python views can keep a mapping draining after successful
unregister; retry only the unmap. Shared reductions release exact IDs and verify
physical release before any opposite-owner growth. Every growth reserves credit
before entering mmap or native controls. Mappings are charged independently of
object usage; registrations are metadata, not a second physical charge.

Status is an immutable cached value. `current` is the last completed split;
`resize.target` is requested capacity, potentially still unfulfilled. Pending
reservations are visible while native calls block. The admission gate and status
lock are separate from blocking work. Completed request IDs remain idempotent.
Failed requests retain ownership and quiescence; resubmission rebuilds removable
private mappings before retrying growth. Native failed allocations that expose
no release identity remain charged and may require terminal cleanup.

Shutdown first stops the worker and waits for controller/L1 ownership to drain.
Native terminal cleanup must confirm no registrations, shared allocations,
retained failures, or pending growth before any remaining caller mapping is
removed. A held Python view can still make shutdown return False. Only True
permits the caller to close its adapter and L1. Retry False without restarting
controllers. Shutdown can wait for a currently blocked native call; status reads
remain available from other threads.

## Verification

Run from the LMCache checkout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /scratch/akira/lmcache-dev/bin/python -m pytest --noconftest -q \
  tests/v1/distributed/test_partition_manager.py
```

These tests use real CPU segmented mappings and native L1 reservation locks,
with a controlled capacity capability. They exercise endpoint/rounded cycles,
request conflicts/idempotency, blocked native-call reservation reporting,
failed-release retry, failed-start cleanup, exact writer drain, and exported
Python views preventing physical handoff. They do not establish live SDK or MP
integration safety. Each fixture maps at most 20 KiB of L1 backing.
