# SPDX-License-Identifier: Apache-2.0
"""Inject real segmented DRAM into one L1 manager and resize without restart."""

# Standard
from pathlib import Path
import mmap

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    DRAMSegmentState,
    SegmentedDRAML1MemoryManager,
)
from lmcache.v1.memory_management import MemoryObj

PAGE = mmap.PAGESIZE
pytestmark = pytest.mark.no_shared_allocator


def _layout(size: int = PAGE) -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([size])], [torch.uint8])


def _mapped_bytes(name: str) -> int:
    total = 0
    for line in Path("/proc/self/maps").read_text().splitlines():
        if f"/memfd:{name} " in line:
            lower, upper = line.split()[0].split("-")
            total += int(upper, 16) - int(lower, 16)
    return total


def _config(size: int) -> L1ManagerConfig:
    return L1ManagerConfig(
        L1MemoryManagerConfig(size, False, shm_name="", align_bytes=PAGE),
        write_ttl_seconds=600,
        read_ttl_seconds=600,
    )


def _key(index: int) -> ObjectKey:
    return ObjectKey(index.to_bytes(32, "big"), "segmented-dram-test", 0)


def _write(l1: L1Manager, index: int, value: int) -> None:
    key = _key(index)
    error, obj = l1.reserve_write([key], [False], _layout())[key]
    assert error is L1Error.SUCCESS and obj is not None
    assert obj.raw_tensor is not None
    obj.raw_tensor.fill_(value)
    assert l1.finish_write([key])[key] is L1Error.SUCCESS


def _read(l1: L1Manager, index: int, value: int) -> None:
    key = _key(index)
    error, obj = l1.reserve_read([key])[key]
    assert error is L1Error.SUCCESS and obj is not None
    assert obj.raw_tensor is not None
    assert torch.all(obj.raw_tensor == value)
    assert l1.finish_read([key])[key] is L1Error.SUCCESS


def test_same_l1_manager_grows_and_releases_backing_in_both_directions() -> None:
    pool = SegmentedDRAML1MemoryManager()
    original = pool.add_segment(PAGE)
    l1 = L1Manager(_config(PAGE), memory_manager=pool)
    try:
        _write(l1, 0, 0x11)
        assert (
            l1.reserve_write([_key(1)], [False], _layout())[_key(1)][0]
            is L1Error.OUT_OF_MEMORY
        )
        added = pool.add_segment(2 * PAGE)
        _write(l1, 1, 0x22)
        _write(l1, 2, 0x33)
        assert l1.get_memory_usage() == (3 * PAGE, 3 * PAGE)
        assert l1.get_capacity_bytes_by_backend() == {L1BackendType.DRAM: 3 * PAGE}
        assert l1.memory_region_count() == 2
        assert l1.get_l1_memory_desc() is None
        pool.begin_drain(original.handle)
        assert pool.remove_segment(original.handle).state is DRAMSegmentState.DRAINING
        assert l1.get_memory_usage() == (3 * PAGE, 3 * PAGE)
        assert l1.get_capacity_bytes_by_backend() == {L1BackendType.DRAM: 2 * PAGE}
        assert l1.delete([_key(0)])[_key(0)] is L1Error.SUCCESS
        assert pool.remove_segment(original.handle).state is DRAMSegmentState.REMOVED
        assert _mapped_bytes(original.mapping_name) == 0
        _read(l1, 1, 0x22)
        _read(l1, 2, 0x33)
        returned = pool.add_segment(PAGE)
        _write(l1, 3, 0x44)
        pool.begin_drain(added.handle)
        assert l1.delete([_key(1), _key(2)]) == {
            _key(1): L1Error.SUCCESS,
            _key(2): L1Error.SUCCESS,
        }
        assert pool.remove_segment(added.handle).state is DRAMSegmentState.REMOVED
        assert _mapped_bytes(added.mapping_name) == 0
        assert l1.get_memory_usage() == (PAGE, PAGE)
        assert l1.get_capacity_bytes_by_backend() == {L1BackendType.DRAM: PAGE}
        assert _mapped_bytes(returned.mapping_name) == PAGE
        _read(l1, 3, 0x44)
        assert l1.memcheck()
    finally:
        l1.close()
    assert pool.get_memory_usage() == (0, 0)


@pytest.mark.parametrize("conflict", ["size", "alignment", "shm", "live", "drain"])
def test_l1_injection_rejects_conflicting_configuration(conflict: str) -> None:
    pool = SegmentedDRAML1MemoryManager()
    segment = pool.add_segment(PAGE)
    config = _config(PAGE)
    objects: list[MemoryObj] = []
    if conflict == "size":
        config.memory_config.size_in_bytes = 2 * PAGE
    elif conflict == "alignment":
        config.memory_config.align_bytes = PAGE // 2
    elif conflict == "shm":
        config.memory_config.shm_name = "another_pool"
    elif conflict == "live":
        _, objects = pool.allocate(_layout(), 1)
    else:
        pool.begin_drain(segment.handle)
    try:
        with pytest.raises(ValueError, match="segmented DRAM"):
            L1Manager(config, memory_manager=pool)
    finally:
        pool.free(objects)
        pool.close()


def test_l1_ownership_cannot_be_transferred_twice() -> None:
    pool = SegmentedDRAML1MemoryManager()
    pool.add_segment(PAGE)
    first = L1Manager(_config(PAGE), memory_manager=pool)
    try:
        with pytest.raises(ValueError, match="already attached"):
            L1Manager(_config(PAGE), memory_manager=pool)
        _write(first, 0, 42)
        _read(first, 0, 42)
    finally:
        first.close()


def test_failed_l1_construction_leaves_pool_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SegmentedDRAML1MemoryManager()
    segment = pool.add_segment(PAGE)

    def fail_event_bus():
        raise RuntimeError("injected event bus failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                "lmcache.v1.distributed.l1_manager.get_event_bus", fail_event_bus
            )
            with pytest.raises(RuntimeError, match="injected event bus"):
                L1Manager(_config(PAGE), memory_manager=pool)
        assert pool.get_memory_usage() == (0, PAGE)
        assert _mapped_bytes(segment.mapping_name) == PAGE
        manager = L1Manager(_config(PAGE), memory_manager=pool)
        try:
            _write(manager, 0, 81)
            _read(manager, 0, 81)
        finally:
            manager.close()
        assert _mapped_bytes(segment.mapping_name) == 0
    finally:
        pool.close()


def test_zero_capacity_l1_can_grow_and_return_to_zero() -> None:
    pool = SegmentedDRAML1MemoryManager()
    l1 = L1Manager(_config(0), memory_manager=pool)
    try:
        assert l1.get_memory_usage() == (0, 0)
        assert l1.get_capacity_bytes_by_backend() == {}
        segment = pool.add_segment(PAGE)
        _write(l1, 1, 57)
        _read(l1, 1, 57)
        pool.begin_drain(segment.handle)
        assert l1.delete([_key(1)])[_key(1)] is L1Error.SUCCESS
        pool.remove_segment(segment.handle)
        assert l1.get_memory_usage() == (0, 0)
        assert l1.get_capacity_bytes_by_backend() == {}
        assert l1.memory_region_count() == 0
        assert _mapped_bytes(segment.mapping_name) == 0
    finally:
        l1.close()


def test_gds_config_cannot_claim_segmented_dram() -> None:
    pool = SegmentedDRAML1MemoryManager()
    config = _config(0)
    config.gds_l1_config = GdsL1Config("/unused-gds-test-path", PAGE)
    with pytest.raises(ValueError, match="GDS"):
        L1Manager(config, memory_manager=pool)
    l1 = L1Manager(_config(0), memory_manager=pool)
    l1.close()
