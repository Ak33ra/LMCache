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
True, including when an enclosing constructor fails. Synchronous initialization
makes one control attempt per operation. `BUSY` fails startup with ingress closed
and ownership retained for cleanup; it never starts a retry worker or silently
opens partial capacity.

A single worker drains accepted work, marks excess private mappings draining,
clears L1 objects, waits for exact capabilities, unregisters exact native ranges,
and removes mappings. Python views can keep a mapping draining after successful
unregister; retry only the unmap. Shared reductions release exact IDs and verify
physical release before any opposite-owner growth. Every growth reserves credit
before entering mmap or native controls. During resize, `BUSY` registration
retries the same mapped range before any additional mapping is created. Exact
registered identities are recognized from the native snapshot, so retries never
register a completed range again. Shared allocation `BUSY` releases the local
reservation after the call; native pending or retained bytes remain charged
from the returned snapshot. Mappings are charged independently of
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

## MP activation and data ownership

`StorageManagerConfig.partition` activates the topology through the normal MP
factory. CLI `--kv-partition` owns a single eager segmented CPU L1 and one
backend with capacity controls. The Mooncake config checks the reserved scratch
size and disables setup-time shared allocation. Native startup additionally
requires the fenced SDK and strict CPU TCP settings.

StorageManager uses explicit Store admission while shared capacity exists.
Synchronous inline copies fill exact temporary write capabilities. The Store
controller atomically takes those capabilities and resolves admission only
after terminal publication and staging cleanup. A 30-second observation timeout
returns failure to the caller while leaving native ownership intact. Alpha one
publishes directly into retained private capacity and suppresses remote lookup.

Locked prefetch requests prefer retained quota and fall back to temporary quota.
The selected lifetime travels with the L1 allocation. Unlocked prefetch requires
retained quota because an unlocked temporary object cannot promise a later hit.
Accepted Store futures and prefetch request identities span queue handoffs;
resize cannot mistake an empty queue for a fully drained controller.

The HTTP router exposes `/partition`, `/partition/resize`, and
`/partition/resize/{request_id}`. Its status reads never call the SDK. The MP
boundary excludes unconverted GPU/SHM, plugin, P2P, and trace paths. Bounded ZMQ
credits and inline tensor workspace checks constrain accepted request work
separately from the cache-backing budget.

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
failed-release retry, failed-start cleanup, repeated registration and allocation
`BUSY`, exact writer drain, and exported
Python views preventing physical handoff. They do not establish live SDK or MP
integration safety. Each fixture maps at most 20 KiB of L1 backing.
