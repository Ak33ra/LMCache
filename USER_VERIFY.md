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

## U2a. Exact DRAM segment backing and physical release

Use the environment above on Linux. The backend creates unique memfd mappings
and needs no named SHM cleanup. It starts empty and exposes no CLI setting.
The [design contract](docs/design/v1/distributed/memory_manager/segmented-dram.md)
explains ownership and the quiescence requirement.

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_segmented_dram.py
```

Expected result: **17 passed, zero skipped**. Recorded on 2026-11-03 with
Python 3.12.13 and torch 2.15.0+cu132. Fifteen cases use actual mappings and
tensor buffers. Two construction-error cases inject failures to check cleanup;
they do not count as live integration evidence.

The check writes and reads every byte of a three-page segment. It compares
reported bytes with `/proc/self/maps`, checks the memfd descriptor, exhausts
the allocator, drains and frees objects, then verifies that both the mapping
and descriptor disappear. It also covers batch rollback, aligned allocation
charges, admission stopping during drain, removal retries, and old handles
and frees after replacement segments have been created.

To inspect the state directly, run this separate, quiescent interpreter:

```bash
python - <<'PY'
import mmap
from pathlib import Path
import torch
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    SegmentedDRAML1MemoryManager,
)

pool = SegmentedDRAML1MemoryManager()
segment = pool.add_segment(mmap.PAGESIZE)
layout = MemoryLayoutDesc([torch.Size([mmap.PAGESIZE])], [torch.uint8])
error, objects = pool.allocate(layout, 1)
view = objects[0].raw_tensor
view.fill_(73)
pool.begin_drain(segment.handle)
pool.free(objects)
print('registration range:', hex(segment.base), segment.mapped_bytes)
print('held view:', pool.remove_segment(segment.handle))
print('usage:', pool.get_memory_usage())
assert torch.all(view == 73)
del view
print('released view:', pool.remove_segment(segment.handle))
print('usage:', pool.get_memory_usage())
assert segment.mapping_name not in Path('/proc/self/maps').read_text()
pool.close()
PY
```

While the Python view survives, expect DRAINING and usage `(0, 4096)` on a
4 KiB-page host. Its base address must remain unchanged. After `del view`,
expect REMOVED, base zero, and `(0, 0)`. The caller in
this example has no outstanding native operation. A raw native pointer can
outlive both the L1 TTL and its Python tensor, so this check does not establish
transfer-completion safety. Shared Store handoff, two-host operation, GPU,
RDMA, page residency, and a total private/shared budget remain outside U2a.

## U2b. Resize backing under the same L1 manager

Run this unit after U2a:

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_segmented_dram_l1.py
```

Expected result: **10 passed, zero skipped**. Recorded on 2026-11-03 in the
U2a environment. The test constructs one `L1Manager` with a one-page segment,
fills it, adds two pages, and fills both new pages. It drains and removes the
original segment, verifies the surviving keys, adds another page, then removes
the two-page segment and verifies the new key. `/proc/self/maps` confirms each
removal. A second test grows an empty L1 to one page and returns it to zero.

The backend is injected through `L1Manager(config, memory_manager=pool)`.
Initial size and alignment must match, SHM must be disabled in the L1 config,
and the pool must have no live objects or drains. Tests reject conflicting
configuration, GDS, and a second L1 owner. `get_l1_memory_desc()` returns `None`
even when only one segment remains, because a removable segment cannot provide
a fixed registration lifetime. No MP CLI enables this backend.

The additional constructor-failure regression injects an event-bus setup
error, then constructs a working L1 with the same still-mapped pool. Failed
initialization leaves ownership available for retry. This injected case is
cleanup evidence, separate from the live resize checks.

Check unchanged L1 and Device-DAX behavior with:

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_l1_manager.py \
  tests/v1/distributed/test_devdax_l1_allocator.py
```

Recorded result: **168 passed, zero skipped**. These existing DAX tests are
regression checks; U2a and U2b provide the DRAM evidence. U2b runs within one
process using real L1 objects and native TTL locks. The caller is quiescent.
It does not establish MP service resize, transfer-completion fencing, or the
physical private/shared handoff with Mooncake Store.

Focused Ruff, isort, mypy, codespell, SPDX, and device-boundary checks passed.
The manual U2a interpreter example also passed. A strict full documentation
build was attempted with an isolated docs environment:

```bash
PYTHONPATH="$PWD" \
  /scratch/akira/kvc-worktrees/dram-segments-docs-env/bin/sphinx-build \
  -W --keep-going -b html docs/source .audit/sphinx-html
```

The build reports 26 existing diagnostics in unchanged `docs/source` pages,
including duplicate CLI and operator labels, short legacy title underlines,
and a malformed `code-block` at
`docs/source/legacy/production/observability/chunk_statistics.rst:291`.
The strict docs gate therefore remains failed. The local log is
`.audit/sphinx-build.log`. The U2 changes add Markdown design and verification
documents and do not modify any `docs/source` file.

## M3b. Store exports hold exact source allocations

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_store_read_operations.py \
  tests/v1/distributed/test_store_controller.py
```

Expected result is **29 passed, zero skipped**. A controlled adapter pauses
an actual read of a native CPU buffer past its TTL. Forced clear and a timed
out shutdown must keep the source pinned. Releasing the reader lets a second
shutdown complete, and the copied bytes must match the originals. The cases
cover retained stores, successful discard, and a failed terminal store.
Another case accepts the buffer, then raises before returning the task ID.
Its isolated child verifies that the source remains readable and cannot be
reclaimed; status requires process exit because acceptance cannot be resolved.
The adapter in these tests is controlled Python code. A further case starts
shutdown during handle acquisition and verifies that the undispatched batch
releases its handles without submitting to the adapter.
The three terminal-result cases also repeat successful shutdown, as required
when a containing storage manager retries cleanup after another tier stalls.

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

## M3c. Prefetch destinations stay private until terminal completion

```bash
CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false PYTHONPATH="$PWD" \
  /scratch/akira/lmcache-dev/bin/python -m pytest \
  --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_l1_write_operations.py \
  tests/v1/distributed/test_prefetch_write_operations.py \
  tests/v1/distributed/test_prefetch_controller.py \
  tests/v1/distributed/test_l1_manager.py
```

Recorded on 2026-11-03 at `562301cf`: **167 passed, zero skipped**.
The eight new write/prefetch lifetime cases use native CPU allocations and
controlled adapters. They hold writes past TTL, verify partial bytes remain
unreadable, reject stale/foreign completion handles, preserve an existing
same-key resident, and retain destinations during aborted or timed out
loads. Terminal failures discard their destinations. Ambiguous submissions
keep affected allocations pinned and report that process exit is required.
Known accepted tasks and unsubmitted buffers are cleaned up independently.
These cases verify LMCache's adapter contract; they do not establish native
TCP, RDMA, or GPU failure completion.

The contract is documented in
[the prefetch design](docs/design/v1/distributed/storage_controllers/prefetch_controller.md).

## M3d. Typed-inline CPU serving allocation lifetimes

Use the existing CPU native environment, with this checkout first on the import
path. The tests use real CPU allocations and native TTL locks.

```bash
export CUDA_VISIBLE_DEVICES=''
export LMCACHE_TRACK_USAGE=false
export PYTHONPATH="$PWD"
export NUMBA_CACHE_DIR=/tmp/kvc-serving-numba
/scratch/akira/lmcache-dev/bin/python -m pytest --confcutdir=tests/v1 \
  -o addopts='' -o log_cli=false -q \
  tests/v1/multiprocess/test_inline_operation_lifetimes.py \
  tests/v1/distributed/test_l1_write_operations.py
/scratch/akira/lmcache-dev/bin/python examples/multi_process/cpu_inline_roundtrip.py
```

The barrier tests pause real `Tensor.copy_` and `Tensor.clone` accesses beyond
one-second reservation TTLs. Forced clear, deletion, and another allocation must
not recycle their source/destination slot. The eventual bytes must match and the
operation counters must return to zero. Other cases cover exceptions, partial
misses, overflow allocation failure cleanup, and zero-lock store publication.
The overflow failure and listener tests use controlled failure/notification
injection; they do not prove a native transport failure contract.

The separate-process TCP smoke must report both PASS lines for exact typed
inline bytes, independent copies, prefix hits, lock release, clear, and graceful
shutdown. These checks do not validate SHM worker access, GPU/RDMA copies,
Mooncake terminal failures, or physical segment removal. The owner contract is
in [the transfer design](docs/design/v1/multiprocess/engine_driven_transfer_design.md#inline-cpu-allocation-lifetimes).

Recorded on 2026-11-03 in the serving-lifetimes worktree. The focused command
passed **12 tests, zero skipped**. The expanded inline, multi-L1, Store-source,
and prefetch-destination selection passed **144 tests, two existing device
skips** before the final two focused additions. Both smoke PASS lines were
observed. Ruff, isort, and whitespace checks passed. Sphinx generated HTML but
failed its warnings-as-errors gate with the same 26 existing source diagnostics
seen in M3b; the log is `/tmp/kvc-serving-docs.log`.

Portable trace replay also has a focused check using the existing dispatcher:

```bash
/scratch/akira/lmcache-dev/bin/python -m pytest --confcutdir=tests/v1 \
  -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_multi_l1.py::test_inline_operations_replay_portable_call_schema
```

It checks portable operation write records and legacy read records through
serialization and replay against a fresh owner, with no leftover read locks
or open replay contexts.
This test uses the existing CPU allocation fixture and establishes replay
compatibility, not native allocation lifetime evidence.

The complete `test_multi_l1.py` file plus both focused lifetime files passed
**37 tests, zero skipped** after trace compatibility was added. The separate
process smoke also passed again. Logs are `/tmp/kvc-serving-trace.log` and
`/tmp/kvc-serving-smoke-final.log`.

The serving regression selection also rejects starting a retrieve without a
current prefetch reservation. The two native cases are
`test_inline_retrieve_requires_current_prefetch_reservation`; both failed
before the correction and pass after atomic TTL-to-operation conversion.
The combined inline, read-operation, Store-source, and multi-L1 selection
passed 46 tests with no skips on 2026-11-03.

Operation trace replay has separate timeout and cleanup checks:

```bash
CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false PYTHONPATH=. \
  /scratch/akira/lmcache-dev/bin/python -m pytest --confcutdir=tests/v1 \
  -o addopts='' -o log_cli=false -q tests/v1/distributed/test_multi_l1.py \
  -k 'inline_write_outcome or operation_replay_tracks or portable_call_schema'
CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false PYTHONPATH=. \
  /scratch/akira/lmcache-dev/bin/python -m pytest --confcutdir=tests/cli \
  -o addopts='' -o log_cli=false -q \
  tests/cli/commands/trace/test_dispatch.py \
  tests/cli/commands/trace/test_driver.py
```

Successful and discarded writes retain their outcomes after the one-second
reservation timeout. Only source-side successful reservations enter the trace.
EOF and interrupted dispatch must leave zero staging allocations. Source
capabilities and owner IDs never enter the file. These CPU checks exercise
storage bookkeeping and native TTL locks, not tensor payload replay.

## M4b. Physical capacity targets

This unit validates the budget and computes targets. It does not activate MP
partitioning or claim that these bytes are allocated.

```bash
cd /home/akira/kvc-partitioning/LMCache
CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false PYTHONPATH=. \
  /scratch/akira/lmcache-dev/bin/python -m pytest --confcutdir=tests/v1 \
  -o addopts='' -o log_cli=false -q tests/v1/distributed/test_partition_config.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /scratch/akira/lmcache-dev/bin/python - <<'PYTHON'
from dataclasses import asdict
import json
from lmcache.v1.distributed.partition_config import PartitionConfig
mib = 1 << 20
config = PartitionConfig(88 * mib, 16 * mib, 8 * mib, 0.5)
for alpha in (0, 0.3, 0.5, 1):
    print(json.dumps({"alpha": alpha, **asdict(config.capacity(16*mib, alpha=alpha))}))
PYTHON
```

Expect 37 tests to pass. The printed private/shared effective MiB pairs are
0/64, 16/48, 32/32, and 64/0. Temporary and SDK reserves remain 16 and 8 MiB.
The alpha 0.3 request has effective alpha 0.25. Tests also cover rounding slack,
exact large-integer endpoints, invalid JSON field sets, booleans, nonfinite
fractions, and insufficient budgets. The root run on 2026-11-03 passed all
37 tests with no skips in `/tmp/kvc-partition-config.log`.

## M4c. Adapter capacity capability and retryable close

From `LMCache/`, use the prepared CPU environment:

```bash
CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false PYTHONPATH=. \
  NUMBA_CACHE_DIR=/tmp/kvc-capacity-adapter-numba \
  /scratch/akira/lmcache-dev/bin/python -m pytest \
  --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_native_connector_l2_adapter.py \
  tests/v1/distributed/test_mooncake_store_l2_adapter.py -k 'not Integration'
```

Expected: 119 passed, one existing optional Redis-extension test skipped, and
12 service tests deselected. The new controlled close tests refuse cleanup once
with BUSY or SDK_ERROR, then prove object completion still arrives before retry.
These are adapter contract tests, not live Store evidence. The native M4a probe
runs actual queued operations and verifies physical ownership separately.
## U4a. Bounded retained and temporary L1 admission

Run from this checkout using the environment at the top of this file. The
admission tests use real segmented CPU mappings of at most 64 KiB per manager
and native TTL locks. They do not require Mooncake or a GPU.

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/distributed/test_l1_admission.py \
  tests/v1/distributed/test_l1_write_operations.py \
  tests/v1/distributed/test_l1_read_operations.py \
  tests/v1/distributed/test_l1_interface.py
```

Expected result is **30 passed, zero skipped**. The 16 new admission cases
cover 4 KiB alignment, bounded fallback, zero private quota, concurrent quota
selection, both same-key winner classes, abort and deferred free, borrowed
reference-count cleanup, quiescing, reduced quotas, retained LRU pressure, and
temporary cleanup with zero private quota. One case injects a later allocator
exception after allocating a real first buffer. The four existing interface
tests use their existing allocation fixture; they are contract regressions,
not additional native allocator evidence.

The caller regression selection is:

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q -ra \
  tests/v1/distributed/test_l1_manager.py \
  tests/v1/distributed/test_prefetch_controller.py \
  tests/v1/distributed/test_prefetch_write_operations.py \
  tests/v1/distributed/test_multi_l1.py \
  tests/v1/distributed/test_eviction_extra_logging.py \
  tests/v1/multiprocess/test_inline_operation_lifetimes.py \
  tests/v1/multiprocess/test_engine_driven_multigroup.py
```

Recorded on 2026-11-03 with Python 3.12.13, torch 2.15.0+cu132, pytest 9.2.0,
CPU platform, and a compiled `lmcache_native` extension. The caller selection
passed **234 tests** and skipped the existing CUDA SHM multigroup roundtrip
because CUDA was unavailable. Prefetch tests use mock L2 adapters. No live
shared-store admission, MP activation, GPU, or physical-resize claim is made.
Logs are `/tmp/kvc-admission-focused-final.log` and
`/tmp/kvc-admission-regressions.log`.

Changed Python files pass Ruff 0.11.7, formatting, isort 6.0.1, SPDX, and the
direct-device checker. Production changes and the new test module pass mypy
1.17.1. Checking both migrated legacy test modules additionally reports seven
existing annotation/narrowing errors in untouched lines; see
`/tmp/kvc-admission-mypy.log`. Codespell 2.4.1 passes the changed content.

The full documentation check is reproducible without modifying the prepared
Python environment:

```bash
uvx --from sphinx --with-requirements requirements/docs.txt \
  sphinx-build -b html -W --keep-going docs/source /tmp/kvc-admission-docs
```

This check remains blocked by 26 diagnostics in existing CLI, legacy, and
operator pages, including a malformed legacy `code-block` directive. All 26
also occur when building the base revision `3744a7e0` documentation. There are
no diagnostics for the new admission page. The build logs are
`/tmp/kvc-admission-docs.log` and `/tmp/kvc-admission-docs-baseline.log`.

The implementation contract is in
[the mirrored admission design](docs/design/v1/distributed/l1-admission.md).


## U4b. Unpublished computed Store admission

Run from this checkout. The staged tests use 4 to 12 KiB native CPU pools;
existing controller regressions use at most 128 MiB per pool.

```bash
CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false \
  PYTHONPATH="$PWD" NUMBA_CACHE_DIR=/tmp/kvc-numba-staged-store-verify \
  /scratch/akira/lmcache-dev/bin/python -m pytest \
  --confcutdir=tests/v1/distributed -q -o log_cli=false \
  tests/v1/distributed/test_store_staged_admission.py \
  tests/v1/distributed/test_store_read_operations.py \
  tests/v1/distributed/test_store_controller.py \
  tests/v1/distributed/test_runtime_adapters.py
```

Recorded on 2026-11-03 with Python 3.12.13 and the compiled common native
extension. This command passed **54 tests, zero skipped**, including 14 new
staged-admission cases, after integration with U4a's typed L1 admission API.
The only pytest warnings were the existing OpenTelemetry logging-handler
deprecation.

The staged checks delay actual CPU source reads beyond the write TTL, force
clear, time out and retry shutdown, fail whole batches, cancel observation,
overlap same-key sources, and preserve a resident. A Future callback checks
that allocator usage and active write handles are already zero. Other cases
check temporary quota release, explicit adapter IDs, rejected ownership
transfer, mixed resident and staged completions, no resident notification,
and process-isolated uncertain submission. Unknown acceptance keeps its
handles and unresolved Future until process exit. Safely undispatched queued
batches are discarded and report failure.

These are controlled Python-adapter tests with native CPU allocations and
locks. They do not prove live Store admission, SDK error fencing, or network
behavior. The native staged Store roundtrip remains a separate activation
gate after the native control-queue repair. This unit does not activate MP
partition configuration.

Ruff 0.11.7, isort 6.0.1, mypy 1.17.1, codespell 2.4.1, and `git diff --check`
pass for the changed content. The contract and rationale are in
[the Store source design](docs/design/v1/distributed/storage_controllers/store_controller.md#unpublished-computed-write-admission).

The final test log is `/tmp/kvc-staged-store-final-tests.log`. The Sphinx
command from U4a was also run with output directory
`/tmp/kvc-staged-store-docs`. It fails on 26 existing diagnostics that also occur on
base `3744a7e0`, with none for this Markdown-only documentation change. The
log is `/tmp/kvc-staged-store-docs.log`.

## M4a — native CPU Mooncake ownership controls

This unit adds opt-in native controls only. It does not enable MP partitioning.
Build the optional connector against the completed strict CPU TCP fork, then
run the real-master probes from this checkout:

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
/scratch/akira/lmcache-dev/bin/python \
  csrc/storage_backends/mooncake/verify_partition_controls.py \
  --master-bin /scratch/akira/kvc-worktrees/mooncake-fenced/build/mooncake-store/src/mooncake_master \
  --output /tmp/kvc-native-partition-recovery-manual --fail-mount
```

Use fresh output paths. Each probe starts and stops its own master and writes
`summary.json`. Expect PASS, exact fetched bytes, zero shared capacity from
private registrations alone, stale-ID rejection, `BUSY` preserving ownership,
visible rounded pending growth, released shared mappings, and zero final
scratch/shared charges. Private unregister must report zero physical release;
caller-owned backing stays accessible until the caller frees it.

The failure probe terminates the master before growth. Expect retained failed
allocation bytes and unsuccessful close with zero release credit. It then
restarts the master and retries close, releasing charges only after success.
The supported boundary and control ordering are documented in
[the native controls design](docs/design/v1/distributed/l2_adapters/mooncake_partition_controls.md).
These checks make no GPU, RDMA, cross-host, or MP activation claim.

Recorded on 2026-11-03: both live probes passed. The request was 64 MiB plus one
byte; the SDK reported 80 MiB actual allocation and 16 MiB granularity. Both
normal release mappings disappeared. Failed growth and failed close retained
80 MiB plus 8 MiB scratch; recovery close reached zero. Evidence:
`/tmp/kvc-native-controls-final/summary.json` and
`/tmp/kvc-native-controls-final-recovery/summary.json`, with matching `.log`
files. Stock 0.3.14 compiled as API 0, rejected partition mode before setup,
and passed default native set/get/delete with exact bytes against its own real
master (`/tmp/kvc-native-controls-stock-build.log` and
`/tmp/kvc-native-controls-stock.log`). Stock compatibility is not ownership
safety evidence. Sphinx retained the existing 26 warning baseline.

### M4a queue ownership repair

Rebuild the native extension before repeating the two M4a commands above.
The extended probe requires unregister and close to return `BUSY` for accepted
requests until their terminal results are drained. This includes completed but
undrained requests. All get, set, exists, and delete submissions must raise
after close starts or succeeds. Retain the connector and backing after an
ambiguous submission exception; unresolved acceptance requires process exit.

Recorded on 2026-11-03 against strict CPU SDK `170e8df`. Both extended probes
passed with zero skips. Each exercised queued get/set behind a paused master,
128 concurrent submissions with a simultaneous drain, same-address replacement
with exact bytes, and four close races. Two races began empty and rejected all
four operations after close. Two began with an undrained future, returned
`BUSY`, and completed all four new operations before retrying close. The
recovery probe also rejected all four submissions after failed SDK teardown.
Final private, shared, failed, and scratch charges were zero.

The real old extension first reproduced a post-close future that never
completed. The extended probe then failed with
`Completed-but-undrained work lost its registration ownership`.
Evidence and build logs are:

- `/tmp/kvc-native-queue-baseline-postclose.log` and
  `/tmp/kvc-native-queue-baseline.log` contain the old failures.
- `/tmp/kvc-native-queue-verified/summary.json` and
  `/tmp/kvc-native-queue-verified-recovery/summary.json` contain the passing
  results, with matching `.log` files.
- `/tmp/kvc-native-queue-build-final.log` contains the isolated API 1 build.

The probe retains two 64 KiB caller mappings, at most one 80 MiB shared
allocation, and 8 MiB SDK scratch at a time. These are cache backing sizes;
Python, torch, native metadata, and process RSS are additional. It does not
inject C++ allocation exceptions or prove cross-host, GPU, or RDMA behavior.
Sphinx generated HTML but retained the same 26 existing warnings, including
the unchanged invalid code-block directive. The log is
`/tmp/kvc-native-queue-docs.log`. Ruff, isort, and whitespace checks passed.
The repaired source also built against stock 0.3.14 as API 0 and passed its
real-master default constructor set/get/delete byte check. Logs are
`/tmp/kvc-native-queue-stock-build.log` and `/tmp/kvc-native-queue-stock.log`.

## U-RPC. Bounded accepted CPU request work

From this checkout and the environment above:

```bash
python -m pytest --confcutdir=tests/v1 -o addopts='' -o log_cli=false -q \
  tests/v1/multiprocess/test_request_limits.py
```

The test runs real loopback TCP ZMQ sockets, with 1 KiB wire and 2 KiB tensor
workspace limits and two accepted requests. It uses barriers to prove running
and queued handlers share the limit, then checks refusal, handler failures,
oversized responses, credit reuse, and shutdown waiting for access to finish.
No Mooncake service, GPU, mock transport, or performance measurement is involved.

The [request-bounds design](docs/design/v1/multiprocess/transport/request-bounds.md)
explains the accounting scope and the required typed-inline workspace checks.
The feature remains opt-in through internal `MPServerConfig.request_limits`.
This unit does not activate partitioning or bound process RSS.

Recorded 2026-11-03 using Python 3.12.13 and the prepared CPU environment.
The focused suite passed **12 tests**, with no skips. A separate selected legacy
MQ regression run plus the first ten bounds cases passed **29 tests**, with
nine unrelated tests deselected. Ruff lint/format, isort, and `git diff --check`
passed for changed Python files. The documentation is internal Markdown; no
Sphinx build or live Mooncake integration is claimed for this unit.
