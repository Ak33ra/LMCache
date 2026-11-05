# Deployment validation, 2026-11-05

The validation target is two Ubuntu 24.04 SSH containers on **one physical
host**. Each has a control-network address and a separate experiment-network
address. This tests fresh installation, SSH orchestration, native CPU behavior,
and recovery. It is not a physical CloudLab result. No CloudLab experiment was
allocated; the user confirmed that theirs is not running yet.

## Environment and source

- LMCache revision: `3e305df16fb63abe70e0dd8b24067d6b89bcafec`.
- Ubuntu 24.04, Python 3.12.3, GCC 13.3.0, CPU Torch 2.13.0+cpu.
- Patched Mooncake tree: `1d498e94a7b1e4af1df1c0cf8005a37ad68fe980`.
- CPU partition connector API: 1. Real native extensions were built.
- [requirements.lock](requirements.lock) pins the Python environment.
- Control addresses: `172.18.0.2` and `172.18.0.3`; experiment addresses:
  `172.19.0.2` and `172.19.0.3`.

All test-owned builds and logs are under
`/scratch/akira/kvc-cloudlab-test`. The existing local LMCache checkout and
prepared native connector were not modified. The test SSH keys belong only to
the disposable fixture and are not part of the deployment package.
The fixture containers and networks were removed after collection. Historical
fixture commands below need newly created SSH targets to run again; retained
logs and builds remain available for inspection.

## Observed results

The final release is `c2f5bc563815d5e312cf`. The
[machine-readable evidence summary](validation-20261105.json) records exact
source hashes, counts, routes, negative controls, and raw artifact locations.

| Verification unit | Observed result |
| --- | --- |
| U7a setup | Both hosts built and verified the final release. |
| U7a repeated setup | Same release; completion markers and build manifests kept their contents and timestamps. |
| U7b corrected baseline | Each host passed 311 Python tests, a real CPU MP roundtrip, 87 native tests, and dependency checks. No skips. |
| U7c partition workload | 45 reads, 90 byte-verified chunks, zero errors, private/shared independence, same-PID resizing, and physical accounting passed. |
| U7d recovery | Both hosts stopped; both archives collected. Repeated cleanup passed. |
| Local lifecycle | Eight real-process checks passed, including stale-identity refusal and descendant cleanup. |
| SSH lifecycle | Six checks passed, covering host-local maps, EOF/hangup, partial launch, repeated stop, and collection. |
| Occupied service port | `BLOCKED`; no services started; unrelated listener survived; diagnostics retained. |
| Reused run ID | `BLOCKED`; all 438 existing evidence files remained unchanged. |
| Interrupted baseline | `FAILED`; node1's job never started; both hosts stopped and both archives collected. |
| Packaging | Two tests passed, including exact clean Git source, repeatable bundles, and exclusion of untracked files. |

The 87 native tests comprise TCP visibility/lifetimes (61), Store transfer
tasks (16), prefetch teardown (6), and fenced mappings (4). The corrected
baseline run is `baseline-final-002`; the final partition run is
`partition-final-002`. Both reported `PASSED` only after cleanup and collection.
The separate interrupted run intentionally failed; its negative-control check
passed because the failure and cleanup were correct.

The route records selected `eth1` and the experiment source IP in both
directions. Replica metadata identified A's `172.19.0.2:24005` endpoint before
B started. B's native maps came from node1. Native deployment used the real
patched SDK and CPU connector; lifecycle child fixtures are separate evidence.

Raw output directories:

- setup: `/scratch/akira/kvc-cloudlab-test/wrappers/u7a_verify.06TZY8vk/setup`
- baseline: `/scratch/akira/kvc-cloudlab-test/wrappers/u7b_verify.3qmGrlVF/baseline`
- partition: `/scratch/akira/kvc-cloudlab-test/wrappers/u7c_verify.b6eGeh7k/partition`
- recovery: `/scratch/akira/kvc-cloudlab-test/wrappers/u7d_verify.Ocy7IX4Z/collected`

## Commands and evidence

The manual entrypoints are the same scripts documented in
[USER_VERIFY.md](../../USER_VERIFY.md#u7a-deploy-two-cloudlab-cpu-nodes).
For the fixture, each command used these invocation-local settings:

```bash
KVC_CLOUDLAB_INVENTORY=/scratch/akira/kvc-cloudlab-test/inventory-final.json \
KVC_CLOUDLAB_SSH_CONFIG=/scratch/akira/kvc-cloudlab-test/ssh_config \
KVC_RESULTS_ROOT=/scratch/akira/kvc-cloudlab-test/wrappers \
./user_verify/u7a_verify.sh
```

Use your own inventory and ordinary SSH configuration on CloudLab. Set
`KVC_CLOUDLAB_RUN_ID` for a reproducibly named baseline or partition run and
for recovery of that run. The fixture's addresses and SSH settings are not
CloudLab defaults.

Local lifecycle verification:

```bash
python3 experiments/cloudlab/test_cli.py
python3 experiments/cloudlab/test_lifecycle.py
python3 experiments/cloudlab/test_ssh_lifecycle.py \
  --inventory /scratch/akira/kvc-cloudlab-test/inventory-final.json \
  --ssh-config /scratch/akira/kvc-cloudlab-test/ssh_config \
  --run-id ssh-lifecycle-final-001 \
  --output-dir /scratch/akira/kvc-cloudlab-test/ssh-lifecycle-final-001 \
  --scope one-host-container-emulation
```

Use a new run ID and output directory when repeating the SSH check.

## Native stress-test limitation

The first complete two-host baseline, `baseline-final-001`, remains a failed
run. Node0 passed 311 Python tests and all 87 native tests. Node1 passed the
311 Python tests, then failed
`TcpWriteVisibilityTest.ConcurrentAdmissionShutdownStressPreservesLaneOwnership`
at its initial hook predicate, before its concurrent shutdown assertions.
Both hosts stopped and their evidence was collected.

The service guardian had injected `MC_NUM_TCP_IO_THREADS=2` into this baseline.
The native source default is one worker. Isolated repetitions of the same
installed binary passed 10/10 with that default, but passed 8/20 and failed
12/20 with two workers. Each failure was the same hook-predicate timeout.

The fixture arms its connect-handler blocker after the first Busy lane. With
two workers, both callbacks can pass the unarmed blocker before either lane
becomes Busy. Direct observation of the test process captured a failing run
with two Busy lanes and an unentered hook throughout the timeout. This proves
that interleaving in the observed reproduction; it does not retrospectively
observe the original failure or establish all two-worker native behavior.
Build contention was not established as the cause.

`baseline.sh` now clears inherited I/O-thread and bind-address overrides along
with the live strict-transfer settings. No native source, test timeout, or
assertion was changed. No retry converts failures into passes. The live
partition workload still uses two I/O workers. Its byte checks do not replace
the native fixture's concurrent admission/shutdown assertions; that fixture
needs deterministic synchronization before it reliably tests two workers.

The investigation, repeat logs, source/binary hashes, and process-state probe
are under `/scratch/akira/kvc-cloudlab-test/review-native-flake/`. To reproduce
the diagnostic on an installed node, substitute its workspace below:

```bash
kvc_workspace=/mydata/kvc
kvc_release=$(cat "$kvc_workspace/current-release")
env -u MC_TCP_REQUIRE_GENERATION -u MC_NOF_BACKEND -u LD_LIBRARY_PATH \
  -u MC_TCP_BIND_ADDRESS MC_NUM_TCP_IO_THREADS=2 \
  "$kvc_workspace/releases/$kvc_release/sdk/build/mooncake-transfer-engine/tests/tcp_write_visibility_test" \
  --gtest_filter=TcpWriteVisibilityTest.ConcurrentAdmissionShutdownStressPreservesLaneOwnership \
  --gtest_repeat=20
```

This diagnostic can fail with the known fixture issue. It is not a passing
acceptance result or an alternative to U7b.

## Scope of the result

The workload uses small CPU buffers, two MP servers, and serial synthetic
requests. Its timings include client and orchestration overhead. They do not
establish network bandwidth, a speedup, or production inference performance.
Container networking cannot establish physical cross-host behavior.

After CloudLab allocation, run U7a, U7b, U7c, then U7d against the real
inventory. Retain the source/build manifests, both hosts' archives, preflight
routes, runtime listeners, replica metadata, byte-check results, and cleanup
records. A successful physical U7c run establishes the CPU TCP experiment on
those hosts. GPU inference and RDMA require separate environments and tests.
