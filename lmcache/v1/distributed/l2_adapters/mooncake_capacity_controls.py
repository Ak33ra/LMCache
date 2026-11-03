# SPDX-License-Identifier: Apache-2.0
"""Translate the fenced Mooncake binding into physical ownership contracts."""

# Standard
from typing import Any

# First Party
from lmcache.v1.distributed.partition_controls import (
    CapacityControlCode,
    CapacityControlResult,
    LocalRegistration,
    LocalRegistrationState,
    SharedAllocation,
    SharedCapacitySnapshot,
)

_CODES = {
    "SUCCESS": CapacityControlCode.SUCCESS,
    "BUSY": CapacityControlCode.BUSY,
    "INVALID": CapacityControlCode.INVALID,
    "SDK_ERROR": CapacityControlCode.FAILED,
    "CLOSED": CapacityControlCode.CLOSED,
    "UNSUPPORTED": CapacityControlCode.UNSUPPORTED,
}


def _snapshot(value: Any) -> SharedCapacitySnapshot:
    """Copy a native API-v1 snapshot into immutable backend-neutral records."""
    return SharedCapacitySnapshot(
        enabled=value.enabled,
        closing=value.closing,
        closed=value.closed,
        scratch_bytes=value.scratch_bytes,
        granularity_bytes=value.allocation_granularity_bytes,
        pending_shared_bytes=value.pending_shared_bytes,
        shared_allocated_bytes=value.shared_allocated_bytes,
        retained_failed_bytes=value.retained_failed_bytes,
        local_buffers=tuple(
            LocalRegistration(
                registration_id=item.id,
                base=item.base,
                size=item.size,
                state=LocalRegistrationState[item.state.name],
            )
            for item in value.local_buffers
        ),
        shared_segments=tuple(
            SharedAllocation(item.id, item.base, item.size)
            for item in value.shared_segments
        ),
    )


def _result(value: Any) -> CapacityControlResult:
    """Preserve errors and actual retained charge across the native boundary."""
    return CapacityControlResult(
        code=_CODES[value.code.name],
        sdk_error=value.sdk_error,
        snapshot=_snapshot(value.snapshot),
        local_buffer_id=value.local_buffer_id,
        segment_ids=tuple(value.segment_ids),
        released_shared_bytes=value.released_shared_bytes,
    )


class MooncakeCapacityControls:
    """Ownership capability over an adapter's existing fenced native client.

    Args:
        native_client: The API-v1 Mooncake client used by the same L2 adapter.

    The wrapper owns no buffers, polling thread, or separate SDK connection.
    """

    def __init__(self, native_client: Any) -> None:
        self._client = native_client

    def snapshot(self) -> SharedCapacitySnapshot:
        """Read the native client's independently cached ownership snapshot."""
        return _snapshot(self._client.partition_status())

    def register_local(self, base: int, size: int) -> CapacityControlResult:
        """Register a local-only range; preserve retained failures in the result."""
        return _result(self._client.register_l1_buffer(base, size))

    def unregister_local(self, registration_id: int) -> CapacityControlResult:
        """Fence an exact local registration without granting backing credit."""
        return _result(self._client.unregister_l1_buffer(registration_id))

    def allocate_shared(self, size: int) -> CapacityControlResult:
        """Request shared backing and return its actual rounded allocation."""
        return _result(self._client.allocate_shared_segment(size))

    def release_shared(self, segment_id: str) -> CapacityControlResult:
        """Return physical credit only after the native mapping is released."""
        return _result(self._client.release_shared_segment(segment_id))

    def close(self) -> CapacityControlResult:
        """Attempt native cleanup; callers retain backing while it is unresolved."""
        return _result(self._client.close_partition())
