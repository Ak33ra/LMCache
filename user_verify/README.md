# Run the verification guide

These executable entry points reproduce the commands in the outer
[USER_VERIFY.md](../USER_VERIFY.md), including its linked engine checks.
Run a script directly from any working directory. Each script accepts
`--help`; normal execution takes no arguments.

```bash
/home/akira/kvc-partitioning/user_verify/u5c_verify.sh
/home/akira/kvc-partitioning/user_verify/u6_verify.sh
```

U5c is the shortest complete feature check. It starts a master and two MP
servers, checks endpoints and same-process resize, and stops those services.
U6 repeats that check with the byte-checked workload. For the prerequisite
sequence, run U0, U1, U2, U3a, U3b, U4a through U4e, then U5a through U6.
A1 and A2 are optional alternative-design experiments.

| Guide section | Entry point | Included work |
| --- | --- | --- |
| U0 | `u0_verify.sh` | Native CPU baseline, separate-process smoke, dependency check |
| U1 | `u1_verify.sh` | Compiled connector, API 1, exact pinned SDK revision |
| U1 optional rebuild | `u1_verify_build.sh` | DEV clean-source SDK recipe and isolated connector build |
| U2 | `u2_verify.sh` | Both default and skip_l1 live runs and summary JSON |
| U3a | `u3a_verify.sh` | Both segmented DRAM suites and held-view mapping example |
| U3b | `u3b_verify.sh` | Three persistent-L1 physical handoff cycles and JSON |
| U4a | `u4a_verify.sh` | Both TTL probes, including the expected safety failure |
| U4b | `u4b_verify.sh` | Read-operation and L1-manager suites |
| U4c | `u4c_verify.sh` | Linked M3c prefetch ownership selection |
| U4d | `u4d_verify.sh` | Sequential U4d.1 and U4d.2 |
| U4d.1 | `u4d_verify_1.sh` | Immediate and one-second grace live fenced handoffs |
| U4d.2 | `u4d_verify_2.sh` | All four native test binaries; one test waits 61 seconds |
| U4e | `u4e_verify.sh` | Inline lifetimes, smoke, portable replay and trace CLI suites |
| A1 | `a1_verify.sh` | Common-arena verifier and tests in its worktree |
| A2 | `a2_verify.sh` | Store-owned placement/lending and graceful lending, both JSON files |
| U5a | `u5a_verify.sh` | Capacity arithmetic tests and printed target loop |
| U5b | `u5b_verify.sh` | Sequential U5b.1, U5b.2 and U5b.3 |
| U5b.1 | `u5b_verify_1.sh` | Admission, callers, staged ownership, capability rejection, BUSY, RPC, activation and M4c adapters |
| U5b.2 | `u5b_verify_2.sh` | Isolated native build, normal and fail-mount probes, both JSON files |
| U5b.3 | `u5b_verify_3.sh` | Public activation probes and strict Sphinx build |
| U5c | `u5c_verify.sh` | Live endpoint/resize harness, correctness/summary JSON, charge and delay inspection |
| U6 | `u6_verify.sh` | U5c plus workload, count assertions and latency JSON |

Each invocation creates a fresh directory beneath `user_verify/results/`
and prints its absolute path. `run.log` records commands, output and exit
status. Harness JSON, process logs, mapping records, isolated native builds
and generated documentation remain in that directory. Numbered substeps
also run independently. Their parent places their result directories beneath
its own. Existing evidence is never overwritten.

Unexpected command failures propagate as nonzero script exits. U4a is the
intentional exception. Its `--require-safe` probe must exit nonzero and emit
the exact assertion diagnostic about TTL reuse before reader completion.
The script then reports `EXPECTED COUNTEREXAMPLE`. That result reproduces a
known gap; it does not establish resize safety. Pytest displays skips rather
than treating them as native evidence. The caller and adapter suites include
documented optional CUDA/Redis skips. U5b.3 requires a successful strict docs
build; old guide entries describing historical warnings do not waive it.

The live harnesses own their services and perform cleanup before returning.
They do not require long-lived services from another step. The optional
[manual server recipe](../experiments/partitioning/README.md#manual-server-launch)
remains available for interactive HTTP inspection.

## Environment overrides

Set overrides on the script invocation; no shell activation is required.
Use absolute paths for path overrides.

```bash
KVC_PROJECT_ROOT=/home/akira/kvc-partitioning \
KVC_RESULTS_ROOT=/tmp/my-verification \
  /path/to/user_verify/u3a_verify.sh
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `KVC_PROJECT_ROOT` | Parent of this directory | Outer project containing experiments and LMCache |
| `KVC_PYTHON` | `/scratch/akira/lmcache-dev/bin/python` | Prepared Python executable |
| `KVC_LMCACHE_ROOT` | `$KVC_PROJECT_ROOT/LMCache` | Engine checkout with matching compiled CPU extensions |
| `KVC_SDK_ROOT` | `/scratch/akira/kvc-worktrees/mooncake-fenced` | Existing fenced SDK checkout and build |
| `KVC_MASTER_BIN` | `$KVC_SDK_ROOT/build/mooncake-store/src/mooncake_master` | Native master for U2, U5b.2, U5c and U6 |
| `KVC_RESULTS_ROOT` | `$KVC_PROJECT_ROOT/user_verify/results` | Parent for fresh result directories |
| `KVC_ARENA_ROOT` | `/scratch/akira/kvc-worktrees/common-arena` | A1 worktree |
| `KVC_STORE_OWNED_ROOT` | `/scratch/akira/kvc-worktrees/store-owned` | A2 worktree |
| `KVC_BUILD_JOBS` | `8` | Build concurrency, integer from 1 through 8 |
| `KVC_UV_CACHE_DIR` | `/tmp/kvc-uv-cache` | uv cache, without changing the prepared environment |
| `KVC_SPHINX_BUILD` | `/scratch/akira/kvc-worktrees/dram-segments-docs-env/bin/sphinx-build` | Prepared isolated Sphinx executable for U5b.3 |

Every step sets CPU visibility and disables usage reporting. It replaces
`PYTHONPATH`, clears inherited `LD_LIBRARY_PATH`, `MC_TCP_REQUIRE_GENERATION`
and `MC_NOF_BACKEND`, and clears `PYTHONOPTIMIZE` and `PYTEST_ADDOPTS`.
The scripts set their own pytest options and per-run Numba cache directory.
They do not alter the calling shell or install into the prepared environment.
U5b.3 uses the existing separate docs environment. It runs the same strict
Sphinx command as the guide's uvx recipe, without resolving dependencies again.
Each log includes the Python version and LMCache/SDK commit IDs.

U3b and A2 use the installed SDK wheel. They select its CUDA runtime library
directory from the chosen Python environment even though execution uses CPU.
U4d.1 uses `$KVC_SDK_ROOT/build/python` with strict generation checks and
without wheel libraries. Those harnesses select their master through their
own `mooncake` package; `KVC_MASTER_BIN` does not override that selection.
Keep the installed wheel and fenced build in separate interpreters.

U5b.2 compiles with `--output-dir` into its fresh result directory. A short
Python launcher prepends that directory to `lmcache.__path__`, verifies the
loaded connector path and API version, and runs the existing native probe.
Both probes therefore exercise the newly built artifact while leaving the
engine checkout's `.so` intact. The build helper still uses its existing
LMCache `.audit/` compilation cache. Avoid simultaneous connector builds in
the same checkout. The optional U1 rebuild also leaves its SDK and connector
in the result directory; preserve that SDK because the connector links to it.

U1 expects the guide's exact SDK commit. A fresh patch reconstruction can
have a different commit ID despite an identical verified source tree; use
the optional builder's source-tree validation for that reproduction.
That optional rebuild needs network access and the development dependencies
from DEV.md, and can take substantially longer than the verification steps.

These checks establish local CPU behavior within each guide section's scope.
Controlled adapters, optional skips, and alternative designs retain their
original limitations. Workload latencies include harness overhead and do not
establish production throughput or a speedup.
