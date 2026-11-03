# Common-arena results, 2026-11-03

Base LMCache revision: `2b7fd89a9c63e823ee9e7be8a1b9ad693fb4274c`.
Branch: `kvc/alternative-common-arena`.
Implementation commit: `e8e7464f`.
Worktree: `/scratch/akira/kvc-worktrees/common-arena`.
Interpreter: `/scratch/akira/lmcache-dev/bin/python`, Python 3.12.13.
PyTorch: `2.15.0+cu132`.
Native common extension: worktree symlink to the existing
`lmcache_native.cpython-312-x86_64-linux-gnu.so` build.

## Observed correctness

The commands in [USER_VERIFY.md](../../USER_VERIFY.md) passed locally.

| Check | Result |
| --- | --- |
| Full native CPU test file | 21 passed, no skips, 37 upstream OpenTelemetry deprecation warnings |
| Manual runner, default 8 MiB | Passed byte comparison, staging invisibility, blocked drain, idempotent retry, stale completion, and reverse resize |
| Mapping accounting | `/dev/shm` segment size was 8,388,608 bytes; its address stayed constant through both resize directions |
| Duplicate retained copies | 1,048,576 bytes charged to private and shared separately; total allocator use 2,097,152 bytes |
| Draining held private reader | Private admission stopped at zero while both effective quotas stayed at 4,194,304 bytes |
| Teardown | Live allocation usage returned to zero; close removed the backing segment |
| Half-copy barriers | Both success and cancellation cases retained source and destination until the synchronous copier stopped |
| Cycling | Ten complete private/shared cycles used the full 32 KiB pool with zero old charges at each handoff |

The native CPU tests use `L1MemoryManager` without allocator mocks. On this
host the CUDA extension is absent, and allocation uses LMCache's real torch
platform fallback with POSIX shared memory. The imported common extension
does not imply CUDA kernels ran. OS page locking was not established.

The initial ordinary CPU allocator probe allocated a 37-byte object with a
4,096-byte charge. Its ctypes byte view required `cast("B")` before writing.
Source inspection then found that the non-SHM fallback adds one backing page
outside the reported usable pool. The final experiment uses SHM so its
mapping size can be directly checked. Mapping capacity and resident pages
remain different measurements.

## Static checks

These checks passed against the experiment files:

```bash
uvx --from ruff==0.11.7 ruff check examples/partitioning_common_arena
uvx --from ruff==0.11.7 ruff format --check examples/partitioning_common_arena
uvx --from isort==6.0.1 isort --check-only examples/partitioning_common_arena
uvx --from mypy==1.17.1 mypy --config-file=pyproject.toml \
  --python-executable=/scratch/akira/lmcache-dev/bin/python \
  examples/partitioning_common_arena
uvx --from codespell==2.4.1 codespell --toml pyproject.toml \
  examples/partitioning_common_arena USER_VERIFY.md
/scratch/akira/lmcache-dev/bin/python tools/check_spdx_header.py \
  examples/partitioning_common_arena/*.py
/scratch/akira/lmcache-dev/bin/python tools/check_direct_torch_device.py \
  examples/partitioning_common_arena/*.py
```

Tool packages ran in isolated `uvx` caches. The shared runtime environment
was not modified. No full accelerator test suite, service test, Sphinx
build, transport benchmark, or cross-host run was performed. Documentation
is confined to the standalone experiment and its manual guide.

This establishes safe local quota changes within one fixed mapping. It
does not establish the selected architecture's handoff between LMCache and
Mooncake allocations, or production shared-cache coordination.
