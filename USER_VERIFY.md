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
returns false. They still skip unavailable accelerator runtimes. The prefetch
fixture uses independent process-local pools because several tests construct
peer managers, which otherwise collide on a PID-derived SHM name.

Recorded on 2026-11-03 with Python 3.12.13, torch 2.15.0+cu132, pytest 9.2.0,
and native artifacts from base `2b7fd89a9c63e823ee9e7be8a1b9ad693fb4274c`.
The platform reported `cpu`, with availability false. Expected result is
**311 passed, zero skipped**. OpenTelemetry logging deprecation warnings
are expected in this environment.

Coverage includes real host allocations and native TTL locks, staging
visibility, duplicate writers, expiry, pinning against eviction within lock
TTLs, temporary-object release, allocation exhaustion, and concurrent access.
Controller tests use real L1 and mock L2 adapters; they establish controller
contracts, not live Store behavior. Policy, session, and codec tests are unit
checks. There is no GPU, cross-host, Mooncake, or online-resize claim.
