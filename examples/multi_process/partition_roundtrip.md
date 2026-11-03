# Live private/shared partition verification

`partition_roundtrip.py` starts an isolated Mooncake master and two CPU MP
servers. It uses the native adapter, typed inline ZMQ transfer, and public HTTP
management APIs. Missing native dependencies or unavailable services fail the
run. No service checks are skipped.

The canonical project location is `experiments/partitioning/partition_roundtrip.py`.
This isolated development branch keeps the candidate under LMCache examples.
Run with the LMCache checkout's CPU native extensions and the fenced
Mooncake adapter built. The master must match the SDK linked into that adapter.
Use a new evidence directory for every run. `--lmcache-root` must match the
checkout imported through `PYTHONPATH`; a mismatch fails before starting services.
The command below starts from the outer project root.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/LMCache"
unset LD_LIBRARY_PATH
python experiments/partitioning/partition_roundtrip.py \
  --lmcache-root "$PWD/LMCache" \
  --master-bin /path/to/mooncake_master \
  --output-dir /tmp/partition-correctness
```

The first server publishes at alpha zero before the second server starts.
The second server must read exact bytes after source tensor mutation and keep
zero private objects. The harness then checks promotion at alpha 0.5, deletion
independence in both directions, private-only writes at alpha one, and disabled
peer fetch at alpha one. Repeated resize cycles keep the second MP PID alive.
Every completed resize is replayed under the same ID and challenged with a
conflicting target. An out-of-range alpha must return HTTP 422.

Each server has 88 MiB of managed backing. Staging reserves 16 MiB and SDK
scratch reserves 8 MiB, leaving 64 MiB for private/shared retained capacity.
The native allocation granularity is 16 MiB. Requests contain two 512-byte
float32 chunks and run serially. Worker source, expected, and retrieved tensors
remain well below the 32 MiB workspace ceiling. This is managed backing
accounting, not a bound on Python, PyTorch, or whole-process RSS.

JSON snapshots preserve the reported charge components. `/proc/PID/maps`
independently checks private memfd backing when the controller is settled.
Released shared segment addresses are compared with current mappings.
`range_absent` proves an unmapped range at that sample; a recorded overlap can
be address reuse and does not prove that the old allocation survived. Native
release/unmap verification remains a separate acceptance unit.

The output contains source and binary hashes, exact subprocess commands,
service PIDs, logs, exit statuses, replica metadata, partition snapshots,
process mapping snapshots, and `summary.json`. Successful teardown requires
all owned services to exit without SIGKILL.

For the bounded workload, repeat with `--workload` and another output directory.
Correctness runs first. Five interleaved rounds exercise alpha 0, 0.5, and 1
with the same two chunks and three serial reads per configuration. Alpha one
seeds local bytes because remote fetch is disabled. Other configurations start
without private copies and read A's published bytes.

`workload.json` records completed hit counts, private/shared counts established
by controlled pre-read residency, exact verified byte counts, and latency per
request. Timing includes lookup, completed prefetch, inline transfer, exact
comparison, and session release. Resize and seeding are outside that interval.
These tiny payloads exercise orchestration and copy correctness. The harness
does not profile a limiting resource or claim a speedup, bandwidth result,
production TTFT improvement, or optimal alpha.
