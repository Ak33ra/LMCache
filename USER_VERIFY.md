# Manual verification

Run from the LMCache checkout with the prepared environment. These commands
select this checkout without changing the shared editable installation.

```bash
source /scratch/akira/lmcache-dev/bin/activate
export PYTHONPATH="$PWD"
export NUMBA_CACHE_DIR=/tmp/kvc-numba-cache
export CUDA_VISIBLE_DEVICES=''
export LMCACHE_TRACK_USAGE=false
```

A fresh worktree needs the CPU common native extensions. Build with
`NO_GPU_EXT=1 python setup.py build_ext --inplace`, or symlink the existing
`lmcache/*.so` from the same revision and Python ABI. The recorded run used
symlinks to `/home/akira/kvc-partitioning/LMCache/lmcache/*.so`.

## U0a. Native CPU storage contracts

Check that Python source comes from this checkout, the platform is CPU,
and the native module is a compiled `.so`, not a source-only test stub.

```bash
python - <<'PY'
import sys, torch, pytest, lmcache, lmcache.lmcache_native as native
print(sys.version)
print('torch', torch.__version__, 'pytest', pytest.__version__)
print('backend', lmcache.torch_device_type, 'available', lmcache.torch_dev.is_available())
print(lmcache.__file__)
print(native.__file__)
PY
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_l1_manager.py \
  tests/v1/distributed/test_l1_memory_manager.py \
  tests/v1/distributed/test_store_controller.py \
  tests/v1/distributed/test_prefetch_controller.py \
  tests/v1/distributed/test_store_policy.py \
  tests/v1/distributed/test_prefetch_policy.py \
  tests/v1/multiprocess/test_session.py \
  tests/v1/multiprocess/test_tensor_wire.py -ra
```

`--confcutdir=tests/v1` avoids the root GPU/shared-allocator fixtures. The
existing tests now admit the CPU platform, whose `is_available()` deliberately
returns false. They still skip unavailable accelerator runtimes and the source-only native
fallback injected by the distributed conftest. The prefetch
fixture uses independent process-local pools because several tests construct
peer managers, which otherwise collide on a PID-derived SHM name.

Recorded on 2026-11-03 with Python 3.12.13, torch 2.15.0+cu132, pytest 9.2.0,
and native artifacts from base `2b7fd89a9c63e823ee9e7be8a1b9ad693fb4274c`.
The platform reported `cpu`, with availability false. Expected result is
**311 passed, zero skipped**. OpenTelemetry logging deprecation warnings
are expected in this environment. A separate gate-only check removed native
module file metadata in a child interpreter and confirmed four module skips;
that simulation is not counted as native-storage evidence.

Coverage includes real host allocations and native TTL locks, staging
visibility, duplicate writers, expiry, pinning against eviction within lock
TTLs, temporary-object release, allocation exhaustion, and concurrent access.
Controller tests use real L1 and mock L2 adapters; they establish controller
contracts, not live Store behavior. Policy, session, and codec tests are unit
checks. There is no GPU, cross-host, Mooncake, or online-resize claim.

## U0b. Separate-process typed inline TCP roundtrip

Use the same environment and native extensions as U0a. The script starts
its own MP server on an unused loopback TCP port, with no L2 adapters, no
SHM transfer, and a configured 64 MiB eager CPU pool. It shuts that server
down before exiting. Individual RPCs have deadlines, startup allows 30
seconds, and graceful shutdown allows 15 seconds before killing the child.

```bash
timeout 120s python examples/multi_process/cpu_inline_roundtrip.py
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/multiprocess/test_cpu_inline_roundtrip.py
```

Expected final output:

```text
PASS: separate-process TCP typed inline roundtrip; 2 chunks, shape=(2, 2, 8, 4), float32; pool=67108864 bytes
PASS: initial miss, source mutation, exact bytes, independent repeat read, prefix hit, lock release, clear miss, graceful server shutdown
```

Recorded passing on 2026-11-03 in the U0a environment. The client and server
are separate OS processes using the current typed codec through
`EngineDrivenContextInline` and `RequestClientFactory`, with eight tokens
per chunk, two layers, four hidden elements, K/V planes, and float32 data.
The check uses the public prepare/commit, lookup, lock-release, session,
registration, and clear RPCs. After a successful store the client overwrites
its source tensors. Both subsequent reads must still equal the originals,
even after the client overwrites its first returned copies. A three-chunk
prefix request returns the two stored chunks, and a non-forced clear must
make the key miss.

The cleanup regression test starts a real child and injects a client-factory
failure while that child is alive. It requires the child to be reaped before
the exception returns. Expected result is **one passed, zero skipped**.
Server cleanup also encloses client shutdown, so an exception from closing
the client cannot bypass child termination.

`verify_inline_roundtrip(client)` can also be imported by a verification
harness that owns an isolated server and connected request client. It assumes
an empty server with eight-token chunks and the default L1-retaining policy,
and clears the server during the check.

This establishes real loopback transport and local host-buffer independence.
It does not run an inference engine, paged gather/scatter, GPU operations,
Mooncake, multiple hosts, an L2 policy, or resizing. The 64 MiB value is cache
pool capacity; Python, torch, transport, and metadata RSS is additional.

Focused style checks use the repository-pinned tool versions:

```bash
uvx ruff@0.11.7 check examples/multi_process/cpu_inline_roundtrip.py \
  tests/v1/distributed/test_{l1_manager,l1_memory_manager,store_controller,prefetch_controller}.py
uvx ruff@0.11.7 format --check examples/multi_process/cpu_inline_roundtrip.py \
  tests/v1/distributed/test_{l1_manager,l1_memory_manager,store_controller,prefetch_controller}.py
uvx isort@6.0.1 --check-only examples/multi_process/cpu_inline_roundtrip.py \
  tests/v1/distributed/test_{l1_manager,l1_memory_manager,store_controller,prefetch_controller}.py
```

## U2. Two MP servers with native Mooncake Store

This isolated experiment requires the CPU native Mooncake binding and master
built against SDK 0.3.14 at `e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c`.
The recorded binding is
`/home/akira/kvc-partitioning/LMCache/lmcache/lmcache_mooncake.cpython-312-x86_64-linux-gnu.so`.
Its runtime path resolves the CPU SDK library built beside the master below;
the Mooncake wheel is not used. Keep the U0 CPU environment exports.

Run each command with a fresh evidence directory. The helper also works from
the outer project's experiment directory when `PYTHONPATH` names this LMCache
checkout, since it imports the reusable inline helper through `examples`.

```bash
timeout 180s python examples/multi_process/mooncake_two_server.py \
  --master-bin /scratch/akira/kvc-mooncake-0.3.14/build/mooncake-store/src/mooncake_master \
  --store-policy default --output-dir /tmp/kvc-two-mp-default
timeout 180s python examples/multi_process/mooncake_two_server.py \
  --master-bin /scratch/akira/kvc-mooncake-0.3.14/build/mooncake-store/src/mooncake_master \
  --store-policy skip_l1 --output-dir /tmp/kvc-two-mp-skip-l1
```

Each run starts its own master and two separate MP HTTP/TCP server processes.
Each MP configures a 64 MiB eager private pool, a 64 MiB shared Store segment,
and a 16 MiB SDK buffer. The resulting configured cache/SDK budget is 144 MiB
per MP, excluding process, transport, and metadata overhead. Master RPC,
metadata, and admin endpoints and both MP endpoints use explicit distinct
ports. SDK endpoint advertisement uses `MC_TCP_BIND_ADDRESS=127.0.0.1`;
the pinned SDK's TCP listener still binds `INADDR_ANY`.

A publishes two float32 chunks of shape `(2, 2, 8, 4)` before B starts, making
A's segment the only possible shared placement. The helper checks the actual
published replica endpoints before fetching through B. This sequence matters
because the current connector's default batch put supplies no placement
preference; `local_first` alone did not reliably place objects on A when both
segments were mounted. Each chunk has 512 payload bytes and occupies a
4096-byte private allocation. The shared allocator charged 1024 bytes total.

The helper waits for actual Store publication and controller quiescence,
overwrites A's original client tensors, and verifies exact bytes through B.
It clears A's private copy and verifies a second shared fetch. It then deletes
the shared objects through the public tier-specific HTTP API, waits for an
empty master namespace, and checks B's private behavior:

| Store policy | Prefetch policy | Private objects after read | B after shared deletion |
| --- | --- | --- | --- |
| `default` | `retain` | 2 per server, 8192 bytes each | exact original bytes |
| `skip_l1` | `default` | 0 per server | miss |

The master's read-only batch metadata API confirms that private reads and
private clear/refetch leave shared replica addresses unchanged. Final checks
require no private objects, no read/write locks or staging bytes, no pending
store work, no shared keys, and zero shared allocator usage. Both private
and shared backing capacities remain 64 MiB per MP throughout.

Each run ends with three `PASS` lines. The second reports retained private
objects per server, `2` for `default` and `0` for `skip_l1`. The last is:

```text
PASS: all owned service processes exited; logs and allocation counters saved
```

Evidence includes exact child commands, native/source provenance, master
version, shared replica descriptors, private status snapshots, shared segment
inventories, service logs, and exit statuses. `manifest.json` records the
executed harness SHA256 and the imported LMCache checkout's source commit.
`summary.json` contains expected and observed values and is marked successful
only after all owned children are reaped. Inspect it with
`python -m json.tool /tmp/kvc-two-mp-default/summary.json`.
The master exits `0`; the MP
HTTP servers log completed application shutdown and exit `-15` because
Uvicorn re-raises SIGTERM. No service required SIGKILL in the recorded runs.
An injected client-factory failure after MP readiness also reaped that MP
and the master, with the same exit statuses.

Recorded passing on 2026-11-03 with the U0 environment. Evidence directories
are `/tmp/kvc-two-mp-default-07`, `/tmp/kvc-two-mp-skip-l1-03`, and
`/tmp/kvc-two-mp-client-failure-01` for the failure cleanup check.

This verifies independent private/shared copies and actual cross-process
Store bytes on one host. It does not prove network isolation, cross-host
behavior, inference integration, capacity exhaustion, concurrent admission,
crash recovery, active-transfer cancellation, operation lifetime safety,
transport fencing, physical backing release, or online resizing. Inline
store acknowledgment alone is not used as shared-admission evidence.

## M3a. Explicit L1 allocation read ownership

Use the U0 CPU environment and compiled common native extension.

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_l1_read_operations.py \
  tests/v1/distributed/test_l1_manager.py
```

Expected result is **105 passed, zero skipped**. The new six cases use real
4096-byte CPU pools and native TTL locks. A held operation survives TTL
expiry, ordinary/forced delete and clear, and an unsuccessful close. The slot
cannot be reused until operation completion. Reusing that same address and
key then rejects the old handle without releasing the replacement.
Other checks cover whole-batch rejection of copied, foreign, repeated, and
already completed handles; the 128-operation admission limit; temporary
objects; and deferred discard across legacy-reader expiry and key reuse.

Recorded on 2026-11-03. Adding these six cases to the full U0a command gave
**317 passed, zero skipped**, with expected OpenTelemetry warnings. This unit
adds the ownership API. M3b below converts Store callers; this API unit alone
does not fix legacy callers, prefetch destination
writes, cancellation, native completion semantics, transport fencing, or
online removal of a backing segment.

## M3b. Store exports hold exact source allocations

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_store_read_operations.py \
  tests/v1/distributed/test_store_controller.py
```

Expected result is **28 passed, zero skipped**. A controlled adapter pauses
an actual read of a native CPU buffer past its TTL. Forced clear and a timed
out shutdown must keep the source pinned. Releasing the reader lets a second
shutdown complete, and the copied bytes must match the originals. The cases
cover retained stores, successful discard, and a failed terminal store.
Another case accepts the buffer, then raises before returning the task ID.
Its isolated child verifies that the source remains readable and cannot be
reclaimed; status requires process exit because acceptance cannot be resolved.
The adapter in these tests is controlled Python code.

Adding both M3 test files to U0a gives **321 passed, zero skipped**, recorded
on 2026-11-03. Both U2 commands also passed against the real CPU native
Mooncake binding after this wiring. Their final evidence is in
`/tmp/kvc-m3b-native-default` and `/tmp/kvc-m3b-native-skip-l1`; inspect each
`summary.json` for expected and observed counts plus process exit statuses.

Runtime adapter changes and interface compatibility also passed:

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_runtime_adapters.py \
  tests/v1/distributed/test_l1_interface.py
```

Expected result is **15 passed, zero skipped**. Eleven runtime adapter tests
use native L1 with mock L2 adapters; their module gate now admits CPU and
rejects source-only native stubs. Four structural interface tests replace
allocation helpers with ordinary torch buffers and are not native-allocation
integration evidence.

The lifetime contract and remaining limits are in
[the Store design](docs/design/v1/distributed/storage_controllers/store_controller.md).
Native success-path verification does not prove native failure fencing,
prefetch destination lifetime, serving-read lifetime, or segment removal.

The Sphinx build was attempted using the existing docs-only environment:

```bash
PYTHONPATH="$PWD" /scratch/akira/kvc-worktrees/dram-segments-docs-env/bin/sphinx-build \
  -b html -W --keep-going docs/source docs/build/html
```

It generated HTML but failed the warnings-as-errors gate with 26 diagnostics
in unchanged `docs/source` files. These include duplicate labels, short
title underlines, and an invalid YAML code-block directive in
`legacy/production/observability/chunk_statistics.rst:291`.
The log is `/tmp/kvc-cpu-docs-build.log`. No shared environment was modified.
