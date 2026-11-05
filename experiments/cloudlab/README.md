# Two-node CPU deployment

Run the existing private/shared verifier on two Ubuntu 24.04 x86-64 hosts.
Node 0 owns the client, Mooncake master and MP A. Node 1 owns MP B. The
workstation needs Python 3.12 or newer, Git and OpenSSH; it needs no Torch,
CUDA or Mooncake installation. The host recipe follows [DEV.md](../../DEV.md).

Use CloudLab's [small-lan profile](https://www.cloudlab.us/example-profiles.php)
with two bare-metal CPU nodes, Ubuntu 24.04
(`UBUNTU24-64-STD`), and one experiment LAN. The proposed Utah `c6525-25g`
pair is suitable for this CPU test. Give each node about 100 GB of local
workspace at `/mydata`. The scripts do not allocate or terminate experiments.
Set their experiment IPv4 addresses explicitly. Use control-network SSH
destinations. Node0 must reach node1 noninteractively using the same `ssh`
value as the workstation. The scripts honor your SSH configuration without
enabling agent forwarding or copying private keys. Verify host keys normally.

Using `YOUR_USER@CONTROL_HOST` as each `ssh` value avoids needing the same
alias on both machines. Copy these destinations from CloudLab's **List View**.
Use `ip -br address` on each node to identify its experiment IP. Keep the
control-network hostname out of the `experiment_ip` fields.

If node0 needs your workstation's SSH key to reach node1, load that key into
your local agent and explicitly enable `ForwardAgent yes` for node0 in your
workstation SSH config. For example, `~/.ssh/config` can contain:

```text
Host NODE0_CONTROL_HOST
    ForwardAgent yes
```

Verify the peer host key in an interactive session:

```bash
ssh-add ~/.ssh/id_ed25519
ssh -A YOUR_USER@NODE0_CONTROL_HOST
```

Then run on node0, using the same node1 destination as the inventory:

```bash
ssh YOUR_USER@NODE1_CONTROL_HOST true
exit
```

Before deployment, confirm that the ordinary workstation SSH command has the
access the CLI will use. Substitute the filled inventory's destinations:

```bash
ssh YOUR_USER@NODE0_CONTROL_HOST \
  'ssh -o BatchMode=yes YOUR_USER@NODE1_CONTROL_HOST true'
```

Both hosts need passwordless `sudo` for package installation, Internet access,
Python 3.12, and enough writable disk for the source, venv and SDK build.
`setup` creates the configured workspace if missing, using `sudo install -d`
only if ordinary creation fails. It does not change ownership of an existing
directory. `/mydata/kvc` is the default; mount your experiment storage first.

```bash
python3 experiments/cloudlab/cloudlab.py inventory > cluster.json
# Edit SSH aliases, experiment IPs, remote_root and port_base in cluster.json.
python3 experiments/cloudlab/cloudlab.py --inventory cluster.json setup
python3 experiments/cloudlab/cloudlab.py --inventory cluster.json \
  baseline --run-id baseline-001 --output-dir results/baseline-001
python3 experiments/cloudlab/cloudlab.py --inventory cluster.json \
  run --run-id partition-001 --output-dir results/partition-001 --workload
```

`--ssh-config PATH` selects a workstation
config; node 0 uses its own `~/.ssh/config` for its peer connection. When using
an isolated writer checkout, set `--project-root` to the outer project and
`--lmcache-root` to its clean engine checkout. Normal outer-project usage
needs neither override.

`setup` packages clean tracked LMCache source in a Git bundle, plus an
explicit allowlist of outer experiment/build files and hashes. It downloads
and builds hosts sequentially. Tail the printed output directory's
`node0-setup.log` or `node1-setup.log` to watch progress. Each immutable release
has a source manifest, package lock and freeze, OS package list, SDK tree ID,
native artifact hashes, build manifest and completion marker. A partial build
cannot pass preflight. Repeating setup reuses only a verified completed build.

`baseline` runs 311 CPU tests with no skips, a real local MP roundtrip, and the
four compiled SDK suites. It clears live-service transfer, bind-address, and
I/O-thread settings so the native suites choose their own configuration. The
native default uses one I/O worker. The live services use two I/O workers and
set strict generation checks, disable NOF and CUDA, and remove inherited
`LD_LIBRARY_PATH`. The CPU native extension reports partition control API 1.
The [validation report](VALIDATION.md#native-stress-test-limitation) records
a synchronization limitation in one native stress fixture with two workers.

`run` reserves fresh directories on both nodes, checks address ownership and
ports, and tests the node0-to-node1 SSH and experiment-LAN connection. It starts
master and A, publishes and proves exclusively-A replicas, then starts B.
The existing byte, private/shared independence, alpha, idempotence, same-PID
and physical-accounting assertions remain the acceptance checks. `--workload`
adds the existing 45-read, 90-chunk workload. No workload request is recorded
as `SKIPPED_BY_REQUEST` for that optional step.

The nine ports starting at `port_base` are reserved for this experiment.
Offsets 0–2 are the master RPC, metadata and admin ports; 3–5 are A's RPC,
HTTP and transfer handshake ports; 6–8 are B's. The SDK also opens dynamic
TCP data ports, so a firewall allowance for only those nine ports is insufficient.
All advertised, configured and dialed cache addresses use the experiment IPs.
The pinned SDK's TCP data acceptor can bind wildcard despite advertising an
experiment IP. Its listener binding does not provide NIC isolation.
Preflight and live service evidence record actual
listeners and routes. These scripts do not change NICs, routes or firewalls.

```bash
python3 experiments/cloudlab/cloudlab.py --inventory cluster.json \
  status --run-id partition-001
python3 experiments/cloudlab/cloudlab.py --inventory cluster.json \
  stop --run-id partition-001
python3 experiments/cloudlab/cloudlab.py --inventory cluster.json \
  collect --run-id partition-001 --output-dir results/recovered-001
```

For `run` and `baseline`, use a new run ID and local output directory.
`status` distinguishes
absent, running, stopped and unreachable hosts. `stop` is repeatable and only
signals recorded run-owned identities. Guardians record host, role, PID,
process group, boot ID, start time and command hash; they clean up on control
EOF, SSH hangup and termination. Recovery refuses uncertain or reused process
identities. Required SIGKILL is a failed cleanup, never a passing experiment.

To interrupt an active command, press Ctrl-C in the workstation terminal
that runs it. Then use `stop` and `collect` to recover any remaining services
and evidence. A failed baseline host stops the sequence before the next host.

`run` and `baseline` use the same recorded guardians and always attempt
cleanup and artifact collection after a failure. Remote
artifacts remain available for a later `collect`. Top-level `outcome.json`
reports `PASSED` only when assertions, cleanup and collection all succeed;
`FAILED` means the attempted experiment failed, and `BLOCKED` means a required
prerequisite was unavailable. A zero exit code from setup proves a completed
build; the separate baseline and workload commands establish their own scope.
Failed host checks preserve their diagnostic output in `nodeN-preflight.log`.
Node archives and extracted `node0/<run-id>` and `node1/<run-id>` directories
include commands, logs, identity ledgers, manifests and exit records. Native
process maps for B are read on node 1 and returned to the existing verifier.

The manual entrypoints are [U7a](../../user_verify/u7a_verify.sh),
[U7b](../../user_verify/u7b_verify.sh), [U7c](../../user_verify/u7c_verify.sh),
and [U7d](../../user_verify/u7d_verify.sh). Set `KVC_CLOUDLAB_INVENTORY` and,
optionally, `KVC_CLOUDLAB_SSH_CONFIG`. U7d also requires the existing
`KVC_CLOUDLAB_RUN_ID`; the other units generate fresh IDs by default. These
wrappers use system Python, not the prepared local native environment.

## Executable lifecycle checks

```bash
python3 experiments/cloudlab/test_lifecycle.py
python3 experiments/cloudlab/test_ssh_lifecycle.py \
  --inventory containers.json --ssh-config containers.ssh_config \
  --run-id ssh-lifecycle-001 --output-dir results/ssh-lifecycle-001 \
  --scope one-host-container-emulation
```

The first command starts real local child processes and checks EOF/hangup,
partial launch, stale PID refusal, descendant cleanup and repeated recovery.
The second requires two already-installed SSH targets and exercises host-local
maps, control interruption, partial startup, repeated stop and collection after
failure. Run `setup`, `baseline` and `run --workload` against the same inventory
to test the complete deployment and experiment route. Two SSH containers on
separate control and experiment bridges constitute **one-host container
emulation**. They establish no physical cross-host or CloudLab result. The
serial synthetic workload makes no bandwidth or performance claim.

[The design note](DESIGN.md) explains ownership and the selected alternative.
[The validation report](VALIDATION.md) records tested versions, commands,
results, and the remaining physical CloudLab check.
