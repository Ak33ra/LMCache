# Native Mooncake CPU partition controls

The optional `LMCacheMooncakeClient(..., partition_controls=True)` mode exposes
removable shared allocations and local-only registrations for caller-owned
private CPU backing. It does not activate partitioning in StorageManager or the
MP server. The default connector behavior remains unchanged.

## Supported boundary

Construction requires the fork's `MOONCAKE_FENCED_TCP_OWNERSHIP_API=1` header and
runtime ownership snapshot version 1. The complete fork contract includes local
TCP registration generations, terminal Store futures, retryable teardown,
local-only registrations, and mmap-owned removable shared segments. A stock
SDK build still supports the default connector, but rejects partition mode.

Required setup is `protocol="tcp"`, `global_segment_size="0"`,
`MC_TCP_REQUIRE_GENERATION=1`, and `MC_NOF_BACKEND=none`. Permanent
`L1RegistrationConfig` registration, hot cache, IPC socket paths, and SSD offload
are rejected. SDK client scratch is reported separately at its actual size.
Only CPU buffers are supported. Callers supply valid, owned host address ranges.

`register_buffer` uses the SDK's local-only transport registration. It never
calls `MountSegmentAndGetId` and adds no Store allocator capacity. The strict
TCP fork also omits private buffers from remote metadata and rejects remote
access to them. A normal Store key becomes available only after an explicit
store into a separately allocated shared segment.

## Typed control surface

| Method | Meaning |
| --- | --- |
| `register_l1_buffer(base, size)` | Register caller-owned backing and return a fresh numeric ID. An identical live registration returns its existing ID. Overlap is rejected. |
| `unregister_l1_buffer(id)` | Release the exact transport registration after all accepted futures are drained. The caller still owns and must separately free its backing. |
| `allocate_shared_segment(size)` | Allocate and mount SDK-owned backing; return its segment IDs and actual accounting snapshot. |
| `release_shared_segment(id)` | Immediately unmount, fence, and release one SDK-owned segment. No grace-period admission is mistaken for release. |
| `partition_status()` | Read cached confirmed ownership and pending growth without waiting for SDK I/O. |
| `close_partition()` | Return `BUSY` while accepted futures remain undrained, then stop admission and tear down the SDK. Retry incomplete close while retaining backing. |

Results contain a `PartitionControlCode`, the original `sdk_error` (or `-1`
for a C++ exception), relevant
registration/segment IDs, `released_shared_bytes`, and a `PartitionStatus`.
Codes distinguish success, busy, invalid input/ID, SDK failure, closed admission,
and unsupported mode. Unknown or stale IDs never affect a newer allocation.
Only SDK-owned shared backing can contribute to `released_shared_bytes`.
Private deregistration always reports zero physical release.

The snapshot reports each private registration, each shared segment's ID/base/
actual size, actual scratch bytes, and `retained_failed_bytes`. Failed mounts
can retain backing without a usable segment ID. Those bytes stay charged until
SDK teardown confirms release. Registration failures likewise retain the
external registration record until cleanup confirms release.

`allocation_granularity_bytes` comes from the SDK slab constant.
`pending_shared_bytes` reports the rounded growth reservation before entering
the SDK allocation call. Other fields remain the last confirmed snapshot while
a control is in flight. On return, pending growth becomes actual shared or
retained-failure charge. Controllers must reserve that rounded budget before
issuing growth, and must not count pending bytes twice.

## Access and teardown ordering

All four native submissions share one admission path in partition mode. It
rejects closing or closed connectors before queue insertion. Get and set also
validate registered ranges before acceptance. Each accepted batch owns a pending
future entry until `drain_completions()` returns its terminal result. Callers
must retain the actual backing and its Python buffers until then. The binding
passes raw addresses and does not retain Python buffer objects for callers.

Unregister and close return `BUSY` while any pending entry remains, even when a
worker has finished but its result is undrained. This deliberately retains all
registrations for each batch, including metadata requests. A queued operation
therefore prevents unregister followed by same-address registration reuse.
Once the caller drains every accepted result, unregister can succeed and a
later registration receives a fresh ID. The old work can no longer access it.

The ownership model is `list<optional<future_id>>`. Admission reserves an empty
entry before calling the base connector, then assigns the returned ID without
allocation. An exception during base submission leaves an unresolved entry.
Its ownership cannot be inferred from a missing future, so unregister and close
remain `BUSY`; callers retain backing until process exit. Input validation
fails before reserving an entry. Admission and drain share a mutex so a fast
completion cannot be drained before its entry receives the ID.

Every worker holds shared SDK access through terminal Store completion.
Admission also holds shared access. Controls use an exclusive try-lock, then
unregister and close try the submission mutex. Drain takes only the submission
mutex. An active worker or another control produces `BUSY` without changing
ownership. The separate snapshot mutex is never held across SDK I/O. The SDK's
remote generation fence provides the separate remote-access guarantee.

Close leaves admission and charges unchanged if either lock or pending ownership
prevents quiescence. Once no accepted work remains, it marks closing, joins the
idle workers while retaining exclusive access, and invokes SDK teardown. New
submissions raise synchronously, including after a failed SDK teardown. A failed
close keeps ownership available for retry. `close()` raises when partition close
is incomplete. The destructor reports unresolved ownership; it is not a
substitute for successful explicit close.

This design keeps `ConnectorBase` and other backends unchanged. Binding exact
registration tokens into base requests was the alternative. It would allow
unrelated registrations to retire sooner, but requires a wider queue contract
change. Conservative pending ownership fits the current quiescent controls.

## Manual evidence

From the LMCache checkout, build only the optional extension against an already
built fork. This leaves the shared Python environment and other native modules
unchanged:

```bash
MAX_JOBS=2 CUDA_VISIBLE_DEVICES='' /scratch/akira/lmcache-dev/bin/python \
  csrc/storage_backends/mooncake/build_partition_connector.py \
  --sdk-root /scratch/akira/kvc-worktrees/mooncake-fenced
export CUDA_VISIBLE_DEVICES=''
export LMCACHE_TRACK_USAGE=false
export MC_TCP_REQUIRE_GENERATION=1
export MC_NOF_BACKEND=none
export PYTHONPATH="$PWD"
/scratch/akira/lmcache-dev/bin/python \
  csrc/storage_backends/mooncake/verify_partition_controls.py \
  --master-bin /scratch/akira/kvc-worktrees/mooncake-fenced/build/mooncake-store/src/mooncake_master \
  --output /tmp/kvc-native-partition-manual
```

Use a fresh output directory. Repeat with `--fail-mount` and a different output
path to terminate the isolated master before growth. That case requires the
failed allocation to remain charged. Close with unavailable metadata must also
retain the charge; the probe restarts its master and retries close to confirm
release. It keeps caller backing alive throughout that recovery.

The probe uses real native source/destination buffers and a real master. It
checks private-registration isolation from Store allocation capacity, exact
bytes after source mutation, repeated shared growth/release, actual rounded
sizes, stale IDs, worker-busy retries, visible pending growth, and final scratch
release. It also checks queued get/set behind a paused-master request,
completed-but-undrained ownership, 128 concurrent submissions with drain,
same-address re-registration, and all four submissions racing with close or
arriving after close. `/proc/self/maps` reports whether each tested shared
mapping disappeared. The requested size is 64 MiB plus one byte; the actual size
is read from the snapshot, not assumed. The probe's paused master establishes
control behavior, not transfer performance. It does not exercise cross-host,
GPU, RDMA, or MP partition activation.

Recorded on 2026-11-03 against the completed strict CPU TCP fork: normal and
failed-mount recovery probes passed. Each 64 MiB plus one byte request allocated
83,886,080 bytes (80 MiB), using the reported 16 MiB slab granularity. Both
released segment mappings were absent. The failed mount and failed close
retained 80 MiB plus 8 MiB scratch; retry after metadata recovery reached zero.
Logs and JSON are under `/tmp/kvc-native-controls-final` and
`/tmp/kvc-native-controls-final-recovery`, with matching `.log` files.

The 2026-11-03 queue repair reproduced the old post-close pending future and
unregister of a completed-but-undrained request. The extended normal and
recovery probes then passed against SDK `170e8df`. Each drained 128 concurrent
submissions and checked queued data, same-address replacement bytes, all four
post-close rejections, and close races with both empty and undrained queues.
Evidence is in `/tmp/kvc-native-queue-verified` and
`/tmp/kvc-native-queue-verified-recovery`, with matching `.log` files.
Ambiguous C++ allocation exceptions were not fault-injected. See
`USER_VERIFY.md` for the old failure logs, commands, and scope limits.

The same connector compiled against stock 0.3.14 and reported API version 0.
Partition mode rejected before setup. A separate real stock master completed
default-constructor set/get/delete with exact CPU bytes and close. This checks
compatibility only; stock mode does not provide the partition ownership contract.
Build and runtime logs are `/tmp/kvc-native-controls-stock-build.log` and
`/tmp/kvc-native-controls-stock.log`.

To reproduce the stock compilation and early rejection without replacing the
fork extension:

```bash
MAX_JOBS=2 CUDA_VISIBLE_DEVICES='' /scratch/akira/lmcache-dev/bin/python \
  csrc/storage_backends/mooncake/build_partition_connector.py \
  --sdk-root /scratch/akira/kvc-mooncake-0.3.14 \
  --output-dir .audit/stock-native
PYTHONPATH="$PWD/.audit/stock-native" /scratch/akira/lmcache-dev/bin/python - <<'PY'
import torch
import lmcache_mooncake as native
assert native.partition_control_api_version == 0
try:
    native.LMCacheMooncakeClient({}, 1, partition_controls=True)
except RuntimeError as error:
    assert "fenced CPU TCP SDK" in str(error)
else:
    raise AssertionError("Stock SDK accepted partition mode")
print("PASS: stock partition mode rejected before setup")
PY
```
