# Verify the Store-owned alternative

This branch contains a throwaway architecture probe. It makes no changes to
LMCache's cache engine, MP protocol, or configuration. The probe compares a
Store placement shortcut with a host-owned RAM arena that lends disjoint
regions to Store and LMCache's existing `TensorMemoryAllocator`.

## Run the probe

Use the prepared Python 3.12 environment on h200b. It contains
`mooncake-transfer-engine==0.3.14` and
`nvidia-cuda-runtime-cu12==12.9.79`. The runtime library is required to import
this wheel even though the probe uses CPU memory and TCP.

```bash
cd /scratch/akira/kvc-worktrees/store-owned
source /scratch/akira/lmcache-dev/bin/activate
export CUDA_VISIBLE_DEVICES=''
export NUMBA_CACHE_DIR=/tmp/kvc-store-owned-numba
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH=/scratch/akira/lmcache-dev/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
python tools/partitioning/probe_lent_segments.py \
    --mode all --cycles 3 --output /tmp/kvc-store-owned-check
python tools/partitioning/probe_lent_segments.py \
    --mode lend --cycles 1 --grace-seconds 1 \
    --output /tmp/kvc-store-owned-graceful-check
```

Each command starts its own master and a separate peer process, then shuts
them down. Each output directory contains `observations.json` and master
logs. Save stdout and stderr to retain client logs too. A failed observation
raises an exception and exits nonzero. The expected placement failure is an
observed counterexample, so that probe exits zero after confirming it.

Master addresses and advertised client addresses use loopback. The probe
allocates free master ports and confines client transfer ports to
26000 through 26999. Mooncake's transfer listeners bind all interfaces in
this version, even when `MC_TCP_BIND_ADDRESS=127.0.0.1` selects loopback for
traffic. The probe does not establish loopback-only listener binding.

## What the observations establish

The placement probe gives the owner no Store capacity and the peer 16 MiB.
The owner writes a `private/owner-a/...` key with its own preferred segment.
The write succeeds in the peer's capacity, and the peer reads the bytes.
Preferred placement and a key prefix therefore fail the private contract.
Strict tenant namespaces are a separate mechanism and are not exercised
here. Tenant quotas do not establish owner-local physical placement.

The lending probe allocates one anonymous, RAM-backed 32 MiB memfd. It
keeps a 16 MiB region private and lends the other 16 MiB to Store through
`mount_segment`. Both clients start with zero Store contribution. Each
client also has an 8 MiB SDK scratch buffer. Runtime and metadata memory
are outside these configured buffer sizes.

| Region | Initial owner | After Store unmount | After private release |
| --- | --- | --- | --- |
| First 16 MiB | LMCache allocator | Same allocator | Same allocator |
| Second 16 MiB | Store | New LMCache allocator | Available to remount |

The host maps the memfd once. Store maps an alias of only the second region.
The probe confirms that Store's bytes appear in the host's original mapping
and that a separate process retrieves identical bytes. It then unmounts the
segment, verifies that Store's mapping disappears, and checks that the peer
can no longer discover the old key. Only then does an LMCache allocator
allocate and overwrite the full withdrawn 16 MiB. After release, another
cycle mounts the same backing again. A private sentinel in the first region
must survive every transition.

The graceful variant also confirms that a successful unmount submission
retains Store's mapping. It waits for that mapping to disappear before
private allocation. This checks one successful cleanup path. It is not a
production completion API or an unregister-failure test.

## Recorded result

The 2026-11-03 run used Python 3.12.13, Torch 2.15.0, and Mooncake 0.3.14.
The LMCache worktree started at
`2b7fd89a9c63e823ee9e7be8a1b9ad693fb4274c`. The probe records its exact
SHA-256 digest and imported LMCache path in each result.

- [Immediate observations](tools/partitioning/results/2026-11-03-immediate.json)
  record the placement counterexample and three successful lending cycles.
- [Graceful observations](tools/partitioning/results/2026-11-03-graceful.json)
  record one successful cycle after asynchronous cleanup.

Every lending cycle consumed the full 16 MiB with the actual LMCache
allocator. The memfd backing stayed at 33,554,432 bytes. Store had zero
remaining mappings of the withdrawn region before private reuse.

The first instrument run encountered a lease-blocked explicit object delete.
The final probe removes that unnecessary operation and uses client teardown
for the placement experiment. An intermediate run also found that this
Python build omits `os.memfd_create`; the probe calls glibc's equivalent.
Those instrument failures do not count as architecture validation.

## Source contract and remaining work

The reviewed Mooncake tag is `v0.3.14`, revision
`e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c`.

- [Allocation fallback][allocation] tries preferred segments, then other
  serving segments. This is not a hard owner-local constraint.
- [RealClient mounting][real-client] opens and maps a caller's file range.
  The memfd supplies RAM backing without a new C++ binding.
- [Immediate and graceful cleanup][client] have different error semantics.
  Immediate unmount propagates non-benign transfer-unregister failures.
  Graceful cleanup logs such failures and still invokes the cleanup
  callback. The observed successful mapping removal does not resolve that
  failure path.
- [TensorMemoryAllocator][allocator] accepts a preallocated tensor view.
  The current L1 memory manager does not expose region lending through
  this interface.

This prototype establishes quiescent reuse of one physical backing through
two existing allocators. It does not pass the selected architecture's M2
or M3 gates. It does not grow an existing L1Manager, run MP requests, test
active transfers, inject unregister failures, protect stale completions,
or prove cross-host operation. The local private key check uses the Store
lookup API. It is not a security isolation claim.

The next implementation would need a region-aware private memory manager,
exclusive region ownership with generations, a trustworthy Store cleanup
completion API, and operation-bound lifetime tests. Keep this candidate if
avoiding OS release and reallocation justifies those changes. It preserves
Store's shared coordination, unlike the separate common-arena peer-cache
alternative.

[allocation]: https://github.com/kvcache-ai/Mooncake/blob/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/include/allocation_strategy.h#L352
[real-client]: https://github.com/kvcache-ai/Mooncake/blob/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/src/real_client.cpp#L2186
[client]: https://github.com/kvcache-ai/Mooncake/blob/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/src/client_service.cpp#L3972
[allocator]: lmcache/v1/memory_allocators/tensor_memory_allocator.py

## SDK-owned backing and concurrent TCP reads

The second runner uses real SDK allocation and native LMCache L1 locks.
Run it in the environment above. It requires the ignored native extension
in this worktree, built with `NO_GPU_EXT=1 python setup.py build_ext --inplace`
or copied from the identical baseline build.

```bash
python tools/partitioning/probe_sdk_handoff.py \
    --mode handoff --cycles 3 --output /tmp/kvc-sdk-handoff
python tools/partitioning/probe_sdk_handoff.py \
    --mode tcp-read --tcp-policy drain-first --output /tmp/kvc-tcp-safe
python tools/partitioning/probe_sdk_handoff.py \
    --mode tcp-read --tcp-policy repro --output /tmp/kvc-tcp-repro
python tools/partitioning/probe_sdk_handoff.py \
    --mode stale-header --output /tmp/kvc-tcp-stale
```

A zero exit means the expected observation occurred. Inspect `gate` before
using a result as acceptance evidence. The repro and stale-header modes
intentionally record blocked gates. Both retain source allocations until
completion and never free an active source. The largest source is 64 MiB;
SDK scratch is 8 MiB per client. The owner runs a real raw loopback TCP
socket against native SDK server threads in the same process. The lending
and handoff modes additionally use the separate SDK peer process.

Measured on 2026-11-03 with wheel 0.3.14, Python 3.12.13 and the baseline
native LMCache extension:

| Observation | Result |
| --- | --- |
| SDK-owned shared/private/shared backing, three cycles | A 16 MiB + 1 request rounds to 32 MiB. The Store range is unmapped before private allocation; the private range is unmapped before the next Store allocation. |
| Real L1 native read reservation | Deletion fails while reserved, succeeds after completion, and usage returns to zero. |
| Drain-first TCP read | All 64 MiB arrives before unregister. No capacity is granted while incomplete. |
| Unregister during an accepted TCP read | Unregister returns success after 4 KiB is consumed. The read later returns a changed source tail, proving access continues. |
| Identical old v2 header after same-address registration | The old header is accepted and returns new registration bytes. |

The SDK-owned check creates and closes an empty L1Manager each cycle. It
does not resize an existing L1Manager. `/proc/self/maps` establishes mapping
presence in this run, not RSS or a general allocator release guarantee.
A remaining mapping blocks the next grant conservatively. The lending
check retains its host-owned memfd throughout and changes exclusive region
ownership instead. Neither result proves concurrent lifetime safety.

Retained evidence is in [SDK handoff](tools/partitioning/results/2026-11-03-sdk-handoff.json),
[drain-first](tools/partitioning/results/2026-11-03-tcp-safe.json),
[accepted-read counterexample](tools/partitioning/results/2026-11-03-tcp-repro.json),
and [stale-header counterexample](tools/partitioning/results/2026-11-03-stale-header.json).
The final shared service cleanup was rerun with stale-header after those
observations. Cleanup checks child exit statuses, escalates terminate to
kill when needed, and attempts all cleanup even if one close fails.

The next safe M3 mechanism is an opt-in TCP registration fence. Each
accepted operation holds a region lease through terminal completion.
Unregister seals admission, returns busy while leases remain, and allows
reuse only after successful removal. A v3 header must carry the registration
generation published in metadata; strict regions reject legacy headers and
old generations. Local outgoing work needs the same lease, including work
waiting in client queues. Store must retain backing on unregister errors,
retry unmount idempotently, and suppress graceful cleanup callbacks until
the transport drain succeeds. Disable Store's local memcpy bypass with
`MC_STORE_MEMCPY=0` until it participates in that lifetime contract.

A cooperative cluster barrier could avoid the wire change only if every
client stops submissions, drains accepted and queued work, discards stale
requests, and stays fenced through reuse. A missing client blocks progress.
This is plausible for a closed harness but weaker as a reusable allocator
contract. Registration generations also do not protect object-slot reuse
within one still-registered segment. Object lifetime must remain tied to
actual transfer completion rather than TTL or an API timeout.
