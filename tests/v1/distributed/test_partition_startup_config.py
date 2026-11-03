# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import replace
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    L1MemoryManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter import (
    MooncakeStoreL2AdapterConfig,
)

PARTITION = {
    "total_bytes": 88 << 20,
    "staging_bytes": 16 << 20,
    "sdk_scratch_bytes": 8 << 20,
    "alpha": 0.5,
}
ADAPTER = {
    "type": "mooncake_store",
    "protocol": "tcp",
    "global_segment_size": "0",
    "local_buffer_size": str(8 << 20),
}


def arguments() -> list[str]:
    """Return explicit partition and shared-owner configuration."""
    return [
        "--kv-partition",
        json.dumps(PARTITION),
        "--l2-adapter",
        json.dumps(ADAPTER),
    ]


def test_partition_cli_owns_staging_backing_and_does_not_mutate_adapter() -> None:
    config = parse_args(arguments())
    assert config.partition is not None
    assert config.partition.total_bytes == 88 << 20
    memory = config.l1_manager_config.memory_config
    assert memory.size_in_bytes == 16 << 20
    assert not memory.use_lazy and not memory.shm_name
    adapter = config.l2_adapter_config.adapters[0]
    assert isinstance(adapter, MooncakeStoreL2AdapterConfig)
    activated = adapter.for_partition(8 << 20)
    assert activated.partition_controls
    assert not adapter.partition_controls
    activated.setup_config["local_hostname"] = "changed"
    assert "local_hostname" not in adapter.setup_config


@pytest.mark.parametrize(
    "extra",
    [
        ["--l1-size-gb", "1"],
        ["--l1-use-hugepages"],
        ["--l1-align-bytes", "8192"],
        ["--l2-store-policy", "skip_l1"],
    ],
)
def test_partition_cli_rejects_conflicting_legacy_options(extra: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_args(arguments() + extra)


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "rdma"),
        ("global_segment_size", "16777216"),
        ("local_buffer_size", "0"),
        ("local_buffer_size", "8MB"),
    ],
)
def test_native_setup_cannot_bypass_the_budget(field: str, value: str) -> None:
    values = {**ADAPTER, field: value}
    with pytest.raises(ValueError):
        parse_args(
            [
                "--kv-partition",
                json.dumps(PARTITION),
                "--l2-adapter",
                json.dumps(values),
            ]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_lazy", True),
        ("shm_name", "unsafe"),
        ("mlock", True),
        ("size_in_bytes", 32 << 20),
        ("align_bytes", 8192),
    ],
)
def test_programmatic_config_rejects_incompatible_backing(
    field: str, value: object
) -> None:
    config = parse_args(arguments())
    memory = replace(config.l1_manager_config.memory_config)
    setattr(memory, field, value)
    l1 = replace(config.l1_manager_config, memory_config=memory)
    with pytest.raises(ValueError, match="Partition L1"):
        replace(config, l1_manager_config=l1, l1_manager_configs=[l1])


def test_unbounded_legacy_configuration_remains_available() -> None:
    config = parse_args(["--l1-size-gb", "0.01", "--eviction-policy", "LRU"])
    assert config.partition is None
    assert isinstance(config.l1_manager_config.memory_config, L1MemoryManagerConfig)
