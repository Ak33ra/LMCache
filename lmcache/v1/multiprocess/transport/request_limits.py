# SPDX-License-Identifier: Apache-2.0
"""Opt-in auxiliary memory limits for the CPU partition request path."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestLimits:
    """Bound accepted requests separately from managed cache backing.

    Args:
        max_payload_bytes: Maximum aggregate wire bytes per request/response,
            excluding explicitly accounted tensor workspace.
        max_workspace_bytes: Maximum sum of simultaneously live tensor buffers
            per handler. Callers count decoded tensors and clones before allocation.
        max_inflight_requests: Maximum queued, running, or unsent requests.

    Raises:
        ValueError: If limits are not positive integers.
    """

    max_payload_bytes: int
    max_inflight_requests: int
    max_workspace_bytes: int

    @classmethod
    def for_partition(cls, staging_bytes: int) -> "RequestLimits":
        """Derive bounded inline admission from a partition's staging reserve.

        Args:
            staging_bytes: Positive configured temporary-cache reserve in bytes.

        Returns:
            Four request credits, a wire ceiling of half the reserve clamped
            to 8--64 MiB, and a per-handler workspace ceiling four times that.
            These auxiliary limits are separate from managed cache backing.

        Raises:
            ValueError: If staging_bytes is not a positive integer.
        """
        if type(staging_bytes) is not int or staging_bytes <= 0:
            raise ValueError("Partition staging bytes must be a positive integer")
        payload_bytes = max(8 << 20, min(64 << 20, staging_bytes // 2))
        return cls(payload_bytes, 4, 4 * payload_bytes)

    def __post_init__(self) -> None:
        if (
            type(self.max_payload_bytes) is not int
            or self.max_payload_bytes < 1
            or type(self.max_inflight_requests) is not int
            or self.max_inflight_requests < 1
            or type(self.max_workspace_bytes) is not int
            or self.max_workspace_bytes < 1
        ):
            raise ValueError("Request limits must be positive integers")

    def check_payload_bytes(self, total_bytes: int) -> None:
        """Reject an encoded payload outside the explicit wire-byte bound."""
        if total_bytes < 0 or total_bytes > self.max_payload_bytes:
            raise ValueError("Inline payload exceeds request limit")

    def check_workspace_bytes(self, total_bytes: int) -> None:
        """Reject an aggregate tensor workspace before allocating its copies.

        Args:
            total_bytes: Sum of simultaneously live handler tensor buffers.

        Raises:
            ValueError: If the workspace is negative or exceeds the limit.
        """
        if total_bytes < 0 or total_bytes > self.max_workspace_bytes:
            raise ValueError("Inline tensor workspace exceeds request limit")
