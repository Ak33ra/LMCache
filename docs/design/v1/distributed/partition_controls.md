# Physical capacity controls

`L2AdapterInterface.get_capacity_controls()` optionally exposes immutable
ownership snapshots and exact registration/segment operations over the same
backend client as object I/O. `MooncakeCapacityControls` translates native
API-v1 records at the adapter boundary. Core partition code does not import
SDK types.

Registration metadata describes caller-owned backing and is not an additional
allocation. Shared segments, failed retained allocations, SDK scratch, and
pending growth are separate charges. Only confirmed unregistration permits a
caller to attempt unmap. Only successful physical shared release grants capacity
back to the other owner.

The native adapter attempts controlled cleanup while completion polling remains
active. A busy or failed close raises and leaves polling alive for retry.
Stopping polling first would prevent queued native ownership from ever draining.
The default path retains its existing constructor and cleanup behavior.

The internal Mooncake config capability is selected by partition startup.
Passing `partition_controls` as an arbitrary SDK JSON field is rejected.

Manual adapter contract checks are in [USER_VERIFY.md](../../../../USER_VERIFY.md).
Controlled tests establish Python routing and retry behavior. The native live
probe separately establishes real ownership and physical release.
