# Store source allocation lifetime

`StoreController` exports committed L1 allocations through L2 adapters.
It acquires an `L1ReadOperation` for every submitted source allocation and
stores those handles beside the adapter/task ID. Session TTL expiry cannot
release them. Ordinary and forced deletion, clear, and L1 shutdown must
respect them until completion.

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
