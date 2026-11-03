# Bounded retained and temporary L1 admission

`L1Manager` optionally owns two allocation quotas in its existing physical
pool. This is the local admission primitive for private/shared partitioning.
It does not enable MP partition configuration, native shared admission,
promotion policy, physical resize, or SDK capacity control.

## Caller contract

```python
limits = L1AdmissionLimits(retained_bytes=8192, temporary_bytes=4096)
manager = L1Manager(config, memory_manager=pool, admission_limits=limits)
results = manager.reserve_write_operations(
    keys, [L1WriteRetention.PREFER_RETAIN] * len(keys), layout, tag="prefetch"
)
```

`RETAIN` needs retained quota. `TEMPORARY` needs temporary quota.
`PREFER_RETAIN` selects retained quota first and falls back to temporary
quota. Selection is final for that allocation. An existing allocation is
never reclassified when limits change or a same-key writer finishes.

The manager allocates and selects retention in one critical section. It uses
the allocation's `get_physical_size()`, including alignment, rather than the
logical tensor size. Bounded batches admit keys independently in input order.
Refused keys return `OUT_OF_MEMORY`; an occupied key/tag returns
`KEY_NOT_WRITABLE`. An allocator exception releases earlier allocations from
that call before escaping. Every successful result has an exact write handle.

Without limits, `PREFER_RETAIN` means `RETAIN` and physical allocation retains
the existing batch atomicity. The legacy boolean `reserve_write` API is
unchanged when bounded admission is disabled and rejected when it is enabled.
The two production exact-write callers use enum intents; their policy choices
are otherwise unchanged in this unit.

## Ownership and accounting

`L1ObjectState.retention` records the selected `RETAIN` or `TEMPORARY` class.
`get_admission_usage()` derives bytes from resident and staging objects under
the L1 lock. All staging tags count, including same-key contenders. A pending
read discard remains in the resident table and counts once. Charges disappear
only when the manager frees the owned allocation. The lookup is linear in
allocated objects; there is no second per-key quota ledger.

Quota bytes are subsets of existing mapped L1 backing. They are not additional
physical charges or an RSS measurement. `None` limits in an unbounded usage
snapshot explicitly mean that independent class limits are not configured.

Memory objects returned by the bounded API are borrowed. A manager-owned
memory-object pin prevents a caller's `ref_count_down()` from returning an
allocation behind the manager's ownership tables. Manager release paths free
and invalidate the object directly. Caller references and tensor views do not
replace exact read/write handles, and callers must not directly free or unpin
the manager's allocation. A surviving Python tensor view can still delay
physical segment unmapping after logical release.

## Limit updates and quiescing

`set_admission_limits()` updates quotas without freeing or moving objects.
Usage may temporarily exceed a reduced limit; that class refuses new
allocations until usage fits. First activation on a previously unbounded
manager requires empty resident and staging tables, so existing legacy
ownership cannot bypass the bounded contract. Disabling quotas is not exposed.

`set_admission_state(QUIESCING)` atomically refuses new write reservations
with `KEY_NOT_WRITABLE` and new TTL/exact readers with `KEY_NOT_READABLE`.
Read-lock conversion also refuses without consuming its TTL reservation.
Existing handles can complete, discard, and release. A held writer may publish
and acquire its completion-time serving locks. `OPEN` resumes admission.
Quiescing is not evidence that native operations have completed. Exact handles
continue to prevent force clear and close until their owners complete them.

## Eviction and temporary cleanup

`get_eviction_pressure()` reports retained allocation bytes and retained quota
when bounded, or physical used/total bytes otherwise. The existing LRU uses
this pressure and excludes temporary-only keys from its victim filter. A
zero retained quota with positive retained usage is full pressure. This lets
private data be reclaimed even when temporary backing keeps total pool usage
below its ordinary watermark.

Each L1 eviction-controller tick also calls `reclaim_temporary_objects()`.
This frees temporary residents after their serving TTLs expire and their
exact reader count reaches zero, even at zero private quota. Active unpublished
write operations never expire. The sweep does no work in disabled mode.
The existing lock-free eviction eligibility hint stays lock-free to avoid
inverting the policy-lock/L1-lock order; deletion rechecks ownership under L1.

## Evidence

`tests/v1/distributed/test_l1_admission.py` drives the public API with small
real segmented CPU mappings and native TTL locks. It covers alignment,
temporary fallback, zero private quota, concurrent admission, same-key losers,
deferred discard, borrowed-reference cleanup, quiescing, quota reduction,
retained pressure, temporary expiry, and disabled-mode batch atomicity.
One test injects an allocator exception after a real first allocation to check
rollback. These are local ownership checks, not native Store integration.
[USER_VERIFY.md](../../../../USER_VERIFY.md) records commands and dated results.
