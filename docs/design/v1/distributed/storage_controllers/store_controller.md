# Store source allocation lifetime

`StoreController` exports committed L1 allocations through L2 adapters.
It acquires an `L1ReadOperation` for every submitted source allocation and
stores those handles beside the adapter/task ID. Session TTL expiry cannot
release them. Ordinary and forced deletion, clear, and L1 shutdown must
respect them until completion. Unpublished computed writes use the staged
admission path described below.

The controller releases handles when the adapter returns a terminal result,
including a failed result. The adapter contract requires that local source
buffer access has ended before returning that completion. A successful
discard policy selects the exact handles to mark for deletion in the same
manager critical section as their release. Other readers can delay deletion;
the manager retains the original allocation identity and reclaims it when
the remaining readers finish or their legacy TTL expires. No deferred
key-only deletion can target a replacement allocation.

## Shutdown and uncertain acceptance

`stop(timeout=30.0)` rejects subsequent notifications and discards queued
notifications. A submission already being dispatched may finish after stop
begins; it remains tracked and must drain before shutdown succeeds. The
controller also drops an undispatched batch if stop is observed after its
allocation handles are acquired. If the deadline
expires, it raises `TimeoutError` while the loop continues draining and the
allocations stay pinned. The caller may retry `stop()` after completions
arrive. L1 `close()` refuses to free active operations.

If `submit_store_task` raises without returning a task ID, the controller
cannot determine whether the adapter accepted the buffers. It quarantines
their handles and stops admitting work. Other accepted tasks still drain.
Status reports `unresolved_submission_count`,
`unresolved_read_operation_count`, and `requires_process_exit=true`.
Shutdown raises `RuntimeError`; this bounded failure state requires process
exit for recovery. The controller does not guess a task ID or classify an
arbitrary exception as a pre-submission rejection. Each allocation admits at
most 128 explicit read handles, and the listener drops future notifications
after shutdown starts, bounding retained state by the existing private pool.

## Evidence and limits

`test_store_read_operations.py` pauses actual CPU buffer reads behind a
controlled Python adapter. It checks TTL expiry, forced clear, a shutdown
timeout followed by successful drain, successful/failed store retention,
and an accepted submission that raises before returning its ID. The last
case runs in a child process because process exit is its recovery boundary.
These tests use native allocations and TTL locks; their adapter is a test
double, not native SDK or network evidence.

The two-MP native Mooncake experiment also verifies both store policies on
the successful path. It does not inject native faults or prove that the SDK
honors the local-buffer completion contract after all transport failures.
Prefetch destination writes, legacy serving reads, remote DMA quiescence,
and safe backing-segment removal require separate work. This change does not
activate resizing or establish complete allocation lifetime safety.

## Unpublished computed-write admission

A caller that already owns filled `L1WriteOperation` handles can transfer them
without publishing their data to private lookup.

```python
completion = controller.submit_staged_write_operations(
    operations, adapter_id=store_descriptor.index
)
result = completion.result()
```

The adapter ID is explicit and stable. It is the descriptor index, not the
adapter's position in a list. Partition setup chooses the single shared Store
adapter. The controller does not infer a shared target from `StorePolicy`.
Each handoff contains one shape, matching the adapter's whole-task contract.
The keys must correspond to the original active handles from this controller's
L1 manager. The writer must finish all local writes before calling and must
not access or complete accepted handles afterward.

`ResidentExport` holds read capabilities for published objects.
`StagedAdmission` holds write capabilities and a
`Future[StoreAdmissionResult]`. Both use the same adapter submission path,
tracking map, and sole `pop_completed_store_tasks()` consumer. A separate
staging controller would introduce competing consumers and ambiguous cleanup
ownership, so staging extends the existing source model instead.

Terminal success and terminal failure both discard every exact staged
allocation through `finish_write_operations`. The controller never publishes
these buffers to L1, invokes resident deletion policy for them, or emits a
resident write notification. A concurrent same-key resident or staged writer
keeps its own allocation. After cleanup, the Future reports immutable
`admitted_keys` and `failed_keys` sets. The adapter currently reports whole-task
success, so a batch has either all admitted keys or all failed keys. A failure
can leave shared keys that were successfully created before the batch failed;
this path does not attempt shared-key rollback.

Cancelling the Future only stops observation. It does not cancel native work
or release handles. The completion producer resolves results after cleanup,
including if cancellation raced with terminal completion. The normal Future
callback runs on the controller thread, so callbacks must not block that
thread waiting for controller work. Callers awaiting the Future must do so
outside L1 and controller locks.

### Queue and shutdown ownership

Queue admission, target removal, and the shutdown boundary are serialized.
A rejected handoff raises before ownership transfers. The caller must discard
or reuse those handles. Accepted handoffs remain controller-owned even when
queued behind another submission or shutdown begins. Normal shutdown dispatches
and drains them; a timeout preserves their allocations and supports retry.
Status includes `pending_staged_admission_count` as well as in-flight work.
An empty handoff resolves with empty sets and submits no adapter task.

If native submission acceptance becomes uncertain, the exact submitted write
handles remain quarantined and their Future stays pending. Status includes
`unresolved_write_operation_count` and `requires_process_exit=true`. Already
tracked tasks still drain. Queued handoffs that never reached an adapter are
safely discarded and resolve as failed. Adapter removal cannot detach an
adapter with unresolved source ownership. A cancelled Future does not change
this recovery boundary.

### Evidence and limits

The staged-admission tests use real CPU allocations and native TTL locks with
controlled Python adapters. They cover delayed reads beyond TTL, success and
failure, cancellation, same-key sources, both source variants sharing a
completion drain, shutdown retry, queue rejection, uncertain acceptance, and
cleanup before Future resolution. The quota integration case checks that
terminal completion restores temporary admission capacity.

The commands and dated results are in
[USER_VERIFY.md](../../../../../USER_VERIFY.md#u4b-unpublished-computed-store-admission).
These are not live Store, SDK-failure, or network evidence. Live staged Store
integration remains a separate partition activation gate after the native
control-queue repair. This module alone does not enable partition configuration
or impose retained and temporary admission quotas.
