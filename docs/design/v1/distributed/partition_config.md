# Private/shared capacity targets

`PartitionConfig` holds one host's managed cache-backing budget D, temporary
L1 reserve T, SDK scratch allowance S, and requested private fraction alpha.
It is a validated arithmetic contract. This unit does not allocate memory or
enable MP partitioning.

For a native-reported segment granule G:

```text
C = floor((D - T - S) / G) * G
P_requested = floor(alpha * C)
P_effective = floor(P_requested / G) * G
H_effective = C - P_effective
unassigned = D - T - S - C
```

Private and shared segments use the same granule. Rounding C before splitting
it prevents rounding a shared allocation beyond the remaining budget. Alpha
zero and one produce exact shared-only and private-only retained targets.
At intermediate alphas, at least one shared segment remains. Integer-ratio
arithmetic preserves the endpoint even for byte budgets beyond float precision.

An 88 MiB budget with 16 MiB temporary reserve and 8 MiB SDK reserve leaves
64 MiB retained capacity. With the measured 16 MiB SDK granule, alpha 0.3
requests about 19.2 MiB private but gives effective capacities of 16 MiB private
and 48 MiB shared. A larger budget permits finer effective alpha increments.
An 89 MiB budget leaves one additional MiB unassigned.

Configuration requires explicit integer byte counts, page-aligned temporary
backing, and a finite alpha in [0, 1]. Capacity calculation rejects a budget
that cannot fit one native segment. Startup must separately compare actual
SDK scratch with its allowance and validate the supported native control API.

Targets are not physical credit. The resize controller retains
charges for live, draining, failed, and pending allocations until release is
confirmed. Registration bytes, used slots, and pinned subsets must not be added
to their already-counted backing. Worker tensors, RPC payloads, metadata, and
process RSS are outside D and require separate bounds or observations. The
opt-in MP integration is described in [partition ownership](partition_manager.md).

[Manual checks](../../../../USER_VERIFY.md#m4b-physical-capacity-targets) print the
requested and effective splits and test numeric validation. They are arithmetic
checks, not evidence of a live partition controller.
