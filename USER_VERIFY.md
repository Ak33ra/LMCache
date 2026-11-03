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
4 KiB-page host. After `del view`, expect REMOVED and `(0, 0)`. The caller in
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
