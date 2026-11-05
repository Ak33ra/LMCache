# Two-host experiment ownership

The deployment CLI runs on the workstation. The byte-checking client runs on
node0, alongside the master and MP A. MP B runs on node1. The inventory keeps
SSH destinations separate from cache addresses so the client uses the
experiment LAN without needing workstation access to that LAN.

`cloudlab.py` owns source packaging, fresh run directories, SSH invocation,
collection, and the final result. `node.py` owns host preflight, the coordinator,
and recovery. `process_host.py` owns service groups, their identity records,
and `/proc` observations. `partition_roundtrip.py` retains the existing
correctness assertions and publication order. It starts B only after proving
that A owns the source replicas.

The Model the Domain principle led to a validated inventory with two distinct
host identities. Separate Before Serializing Shared State led to isolated
release and run directories, with one guardian writing each service record.
Build the Lever led to executable setup, baseline, workload, and recovery
commands. Prove It Works led to fresh CPU builds, real-process interruption
tests, and SSH tests in addition to source review.

## Source and process boundaries

Setup sends the exact clean engine revision in a Git bundle and hashes an
explicit list of outer experiment files. Each release has a pinned Python
dependency set, the expected patched SDK tree, installed package records,
native binary hashes, and a completion marker. A verification command checks
the selected release before starting a job. The remote SDK stays in place
because native libraries refer to its build paths.

The workstation reserves a new run directory on each host before starting
anything. A coordinator watches its SSH input for EOF. Each service has a
guardian that owns its process group and records the boot ID, PID, group ID,
start time, and command hash. Cleanup checks these identities before signaling.
If a group survives without a provable owner, status reports it unresolved
and recovery refuses to signal it. A required SIGKILL fails the experiment.

Baseline jobs use the same ownership and recovery path as partition runs.
Their natural exit code must be zero, and the baseline script must record its
own passing result. A daemon service's permitted SIGTERM shutdown cannot stand
in for successful baseline completion. Both host results, cleanup, and artifact
collection must succeed before the workstation reports `PASSED`.

## Selected alternative

Four design candidates compared node0 coordination with workstation sequencing.
The Astra candidate was selected by both the root review and an independent
Sol review. Its narrow process and mapping interface preserved the existing
assertion owner. The Sol candidate contributed explicit identity records,
recovery states, and an executable SSH lifecycle check. The Terra candidate
contributed the host-local bootstrap boundary. The Luna candidate clarified
the separation between workstation invocation and the node0 data client.

A workstation-managed publication barrier would split assertion sequencing
across processes and add a control protocol. A copied cross-host assertion
suite would create two versions of the same checks. A general deployment
framework would add dependencies without removing the experiment's ordering
and cleanup requirements. Node0 coordination accepts one extra SSH prerequisite
in exchange for keeping the existing verifier intact.

## Reconciled implementation decisions

Independent review exposed surviving descendants after their group leader
exited. Cleanup now observes the whole group and requires a still-matching
observed member before escalation. Status also reports leaderless groups.
Review brought baseline jobs under the same recovery mechanism and added their
explicit completion check. Real Python 3.12 probes found that a buffered stdin
watcher aborted at interpreter shutdown. The watcher now reads the raw file
descriptor. These changes preserve the selected ownership boundaries.

Repeated setup exposed nondeterministic bytes from multithreaded Git packing.
Packaging now fixes the pack thread count; repeated bundles from the same
checkout produce the same release ID. Workstation preflight failures retain
their remote diagnostics, and a failed baseline host stops host sequencing.

The baseline guardian initially injected the live-service I/O-thread count
into native tests. A reproduced test-hook race exposed that boundary error.
The baseline now clears service tuning and lets native fixtures select their
own environment. The unresolved two-worker stress-fixture limitation remains
visible in the [validation report](VALIDATION.md#native-stress-test-limitation).

The TCP connectivity probe uses an ephemeral port so it does not leave a
future service port in TIME_WAIT. The pinned SDK also chooses dynamic data
ports and can bind wildcard listeners. The scripts record routes and listeners
and use experiment IPs for cache traffic; they do not claim NIC isolation.

The test uses small CPU buffers and synthetic requests. Container evidence
establishes orchestration on one physical host. Physical CloudLab correctness,
link performance, GPU behavior, and RDMA remain separate experiments.
