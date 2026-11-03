# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral ownership controls for physical partition changes."""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class CapacityControlCode(Enum):
    """A control attempt's outcome; failure never implies physical release."""

    SUCCESS = "success"
    BUSY = "busy"
    INVALID = "invalid"
    FAILED = "failed"
    CLOSED = "closed"
    UNSUPPORTED = "unsupported"


class LocalRegistrationState(Enum):
    """A caller-owned range is usable or retained pending confirmed cleanup."""

    REGISTERED = "registered"
    RETAINED = "retained"


@dataclass(frozen=True)
class LocalRegistration:
    """Registration metadata; these bytes belong to the caller's allocation."""

    registration_id: int
    base: int
    size: int
    state: LocalRegistrationState


@dataclass(frozen=True)
class SharedAllocation:
    """An exact live backend-owned shared segment, including draining bytes."""

    segment_id: str
    base: int
    size: int


@dataclass(frozen=True)
class SharedCapacitySnapshot:
    """Last confirmed ownership plus reserved growth, without blocking on I/O.

    Shared allocations, retained failures, scratch, and pending growth are
    distinct charges. Local registrations are metadata for caller-owned backing
    and must not be charged a second time. This snapshot grants no access lease.
    """

    enabled: bool
    closing: bool
    closed: bool
    scratch_bytes: int
    granularity_bytes: int
    pending_shared_bytes: int
    shared_allocated_bytes: int
    retained_failed_bytes: int
    local_buffers: tuple[LocalRegistration, ...]
    shared_segments: tuple[SharedAllocation, ...]


@dataclass(frozen=True)
class CapacityControlResult:
    """Actual ownership after one control attempt, including failed mutations."""

    code: CapacityControlCode
    sdk_error: int
    snapshot: SharedCapacitySnapshot
    local_buffer_id: int = 0
    segment_ids: tuple[str, ...] = ()
    released_shared_bytes: int = 0


class SharedCapacityControls(Protocol):
    """Controls on the same backend client used for object I/O.

    The topology owner serializes mutations, keeps caller allocations alive,
    and drains exact L1 operations before unregistering or removing mappings.
    BUSY and FAILED retain the ownership reported in the result. Methods may
    perform blocking SDK work; ``snapshot`` must remain independently readable.
    """

    def snapshot(self) -> SharedCapacitySnapshot:
        """Return the last confirmed state without waiting for SDK I/O."""
        ...

    def register_local(self, base: int, size: int) -> CapacityControlResult:
        """Register caller-owned backing for local transfer access only."""
        ...

    def unregister_local(self, registration_id: int) -> CapacityControlResult:
        """Fence an exact registration; success does not free caller backing."""
        ...

    def allocate_shared(self, size: int) -> CapacityControlResult:
        """Allocate backend-owned shared backing; report actual rounded bytes."""
        ...

    def release_shared(self, segment_id: str) -> CapacityControlResult:
        """Withdraw and release an exact shared allocation after access drains."""
        ...

    def close(self) -> CapacityControlResult:
        """Attempt terminal cleanup; retain all reported ownership on failure."""
        ...
