# SPDX-License-Identifier: Apache-2.0
"""Managed cache backing budgets for private/shared partitioning."""

# Standard
from dataclasses import dataclass
import math
import mmap
import sys


@dataclass(frozen=True)
class PartitionCapacity:
    """Requested split and usable whole-segment capacity within one budget.

    Reserves and rounding slack are outside the retained capacity. None of
    these targets measures allocated memory, native access, or process RSS.
    """

    granularity_bytes: int
    retained_bytes: int
    private_requested_bytes: int
    shared_requested_bytes: int
    private_effective_bytes: int
    shared_effective_bytes: int
    unassigned_bytes: int
    effective_alpha: float


@dataclass(frozen=True)
class PartitionConfig:
    """One host's managed backing envelope and requested private fraction.

    Args:
        total_bytes: Bound on L1 mappings, shared mappings, and SDK scratch.
        staging_bytes: Permanent L1 backing reserved for temporary operations.
        sdk_scratch_bytes: Explicit SDK scratch allowance, checked at activation.
        alpha: Requested private fraction of the retained capacity, in [0, 1].

    Raises:
        ValueError: Values are invalid, unaligned, or leave no retained budget.

    RPC payloads, worker tensors, and runtime metadata need separate bounds.
    A target is not allocation credit; resize must first confirm actual release.
    """

    total_bytes: int
    staging_bytes: int
    sdk_scratch_bytes: int
    alpha: float

    def __post_init__(self) -> None:
        for name in ("total_bytes", "staging_bytes", "sdk_scratch_bytes"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= sys.maxsize:
                raise ValueError(f"{name} must be a positive native-sized integer")
        if self.staging_bytes % mmap.PAGESIZE:
            raise ValueError("staging_bytes must be a multiple of the page size")
        if self.total_bytes <= self.staging_bytes + self.sdk_scratch_bytes:
            raise ValueError("total_bytes must exceed staging and SDK reserves")
        self._validate_alpha(self.alpha)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "PartitionConfig":
        """Parse exactly the four partition fields from a configuration object.

        Args:
            values: JSON-compatible mapping with explicit byte counts and alpha.

        Raises:
            ValueError: Fields are missing, unknown, or invalid.
        """
        fields = {"total_bytes", "staging_bytes", "sdk_scratch_bytes", "alpha"}
        if set(values) != fields:
            raise ValueError(
                "Partition fields differ: "
                f"missing={sorted(fields - set(values))}, "
                f"unknown={sorted(set(values) - fields)}"
            )
        return cls(**values)  # type: ignore[arg-type]

    def capacity(
        self, granularity_bytes: int, *, alpha: float | None = None
    ) -> PartitionCapacity:
        """Compute a split without exceeding the managed backing envelope.

        Args:
            granularity_bytes: Native-reported segment granule; a page-aligned
                power of two. Private and shared use the same granule.
            alpha: Optional resize target. Omission uses the configured alpha.

        Returns:
            Requested and effective byte targets, including unused slack.

        Raises:
            ValueError: Granularity or alpha is invalid, or no segment fits.
        """
        if (
            type(granularity_bytes) is not int
            or granularity_bytes < mmap.PAGESIZE
            or granularity_bytes > sys.maxsize
            or granularity_bytes & (granularity_bytes - 1)
        ):
            raise ValueError("granularity_bytes must be a page-aligned power of two")
        fraction = self.alpha if alpha is None else alpha
        self._validate_alpha(fraction)
        available = self.total_bytes - self.staging_bytes - self.sdk_scratch_bytes
        retained = available // granularity_bytes * granularity_bytes
        if not retained:
            raise ValueError("Retained budget must fit at least one native segment")
        numerator, denominator = fraction.as_integer_ratio()
        private_requested = retained * numerator // denominator
        private = private_requested // granularity_bytes * granularity_bytes
        return PartitionCapacity(
            granularity_bytes=granularity_bytes,
            retained_bytes=retained,
            private_requested_bytes=private_requested,
            shared_requested_bytes=retained - private_requested,
            private_effective_bytes=private,
            shared_effective_bytes=retained - private,
            unassigned_bytes=available - retained,
            effective_alpha=private / retained,
        )

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not 0 <= alpha <= 1
            or not math.isfinite(alpha)
        ):
            raise ValueError("alpha must be a finite number between zero and one")
