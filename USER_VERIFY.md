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
