# SPDX-License-Identifier: Apache-2.0
"""Physical split arithmetic must never grant the same backing twice."""

# Standard
from dataclasses import replace
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.partition_config import PartitionConfig

MIB = 1 << 20
GRANULE = 16 * MIB
CONFIG = PartitionConfig(88 * MIB, 16 * MIB, 8 * MIB, 0.5)


@pytest.mark.parametrize(
    "alpha,private_mib,shared_mib", [(0, 0, 64), (0.5, 32, 32), (1, 64, 0)]
)
def test_endpoints_and_balanced_split(
    alpha: float, private_mib: int, shared_mib: int
) -> None:
    """Both endpoints and a balanced split reserve staging and SDK first."""
    capacity = CONFIG.capacity(GRANULE, alpha=alpha)
    assert capacity.private_effective_bytes == private_mib * MIB
    assert capacity.shared_effective_bytes == shared_mib * MIB
    assert capacity.effective_alpha == alpha
    assert capacity.unassigned_bytes == 0


def test_rounding_exposes_slack_without_shared_overshoot() -> None:
    """Round retained capacity before its split so the complement always fits."""
    config = replace(CONFIG, total_bytes=89 * MIB, alpha=0.3)
    capacity = config.capacity(GRANULE)
    assert capacity.private_effective_bytes == 16 * MIB
    assert capacity.shared_effective_bytes == 48 * MIB
    assert capacity.effective_alpha == 0.25
    assert capacity.unassigned_bytes == MIB
    assert (
        capacity.private_effective_bytes
        + capacity.shared_effective_bytes
        + capacity.unassigned_bytes
        + config.staging_bytes
        + config.sdk_scratch_bytes
    ) == config.total_bytes
    assert capacity.private_requested_bytes + capacity.shared_requested_bytes == (
        capacity.retained_bytes
    )


def test_large_integer_budget_preserves_exact_endpoint() -> None:
    """Float conversion must not grant an extra segment near the native limit."""
    config = replace(CONFIG, total_bytes=sys.maxsize, alpha=1.0)
    capacity = config.capacity(GRANULE)
    assert capacity.private_requested_bytes == capacity.retained_bytes
    assert capacity.private_effective_bytes == capacity.retained_bytes
    assert capacity.shared_effective_bytes == 0


@pytest.mark.parametrize("alpha", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_invalid_fraction_rejected_at_startup_and_resize(alpha: object) -> None:
    """Configuration and management requests share the same numeric boundary."""
    with pytest.raises(ValueError, match="alpha"):
        replace(CONFIG, alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        CONFIG.capacity(GRANULE, alpha=alpha)


@pytest.mark.parametrize("field", ["total_bytes", "staging_bytes", "sdk_scratch_bytes"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "4096", sys.maxsize + 1])
def test_byte_counts_require_bounded_integers(field: str, value: object) -> None:
    """JSON coercion cannot turn booleans or fractional bytes into capacity."""
    with pytest.raises(ValueError, match=field):
        replace(CONFIG, **{field: value})


def test_unaligned_or_exhausted_reserves_are_rejected() -> None:
    with pytest.raises(ValueError, match="page size"):
        replace(CONFIG, staging_bytes=4097)
    with pytest.raises(ValueError, match="exceed"):
        replace(CONFIG, total_bytes=24 * MIB)
    with pytest.raises(ValueError, match="one native segment"):
        replace(CONFIG, total_bytes=25 * MIB).capacity(GRANULE)


@pytest.mark.parametrize("granule", [True, 0, 2048, 4097, 3 * MIB, 1 << 64])
def test_invalid_native_granularity_is_rejected(granule: int) -> None:
    with pytest.raises(ValueError, match="granularity"):
        CONFIG.capacity(granule)


def test_explicit_configuration_fields() -> None:
    values = {
        "total_bytes": CONFIG.total_bytes,
        "staging_bytes": CONFIG.staging_bytes,
        "sdk_scratch_bytes": CONFIG.sdk_scratch_bytes,
        "alpha": CONFIG.alpha,
    }
    assert PartitionConfig.from_dict(values) == CONFIG
    with pytest.raises(ValueError, match="unknown"):
        PartitionConfig.from_dict({**values, "private_bytes": MIB})
    del values["alpha"]
    with pytest.raises(ValueError, match="missing"):
        PartitionConfig.from_dict(values)
