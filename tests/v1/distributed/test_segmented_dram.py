# SPDX-License-Identifier: Apache-2.0
"""Real Linux DRAM mappings and quiescent L1 resize; no allocator mocks."""

# Standard
from pathlib import Path
from uuid import uuid4
import mmap
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    DRAMSegmentHandle,
    DRAMSegmentState,
    SegmentedDRAML1MemoryManager,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator

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


def _descriptor_count(name: str) -> int:
    count = 0
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        count += target == f"/memfd:{name} (deleted)"
    return count


def test_exact_mapping_and_full_capacity_roundtrip() -> None:
    pool = SegmentedDRAML1MemoryManager()
    segment = pool.add_segment(3 * PAGE)
    try:
        assert _mapped_bytes(segment.mapping_name) == 3 * PAGE
        assert _descriptor_count(segment.mapping_name) == 1
        assert pool.get_memory_usage() == (0, 3 * PAGE)
        error, objects = pool.allocate(_layout(3 * PAGE), 1)
        assert error is L1Error.SUCCESS
        assert segment.base != 0 and segment.base % PAGE == 0
        assert objects[0].data_ptr == segment.base
        assert objects[0].get_size() == segment.mapped_bytes
        tensor = objects[0].raw_tensor
        assert tensor is not None
        expected = torch.arange(3 * PAGE, dtype=torch.int64).to(torch.uint8)
        tensor.copy_(expected)
        assert torch.equal(tensor, expected)
        assert pool.get_memory_usage() == (3 * PAGE, 3 * PAGE)
        assert pool.allocate(_layout(), 1) == (L1Error.OUT_OF_MEMORY, [])
        pool.begin_drain(segment.handle)
        assert pool.remove_segment(segment.handle).state is DRAMSegmentState.DRAINING
        pool.free(objects)
        del tensor
        removed = pool.remove_segment(segment.handle)
        assert removed.state is DRAMSegmentState.REMOVED
        assert removed.base == 0
        assert pool.get_memory_usage() == (0, 0)
        assert _mapped_bytes(segment.mapping_name) == 0
        assert _descriptor_count(segment.mapping_name) == 0
        assert pool.remove_segment(segment.handle) == removed
    finally:
        pool.close()


def test_python_view_blocks_release_after_logical_free() -> None:
    pool = SegmentedDRAML1MemoryManager()
    segment = pool.add_segment(PAGE)
    _, objects = pool.allocate(_layout(), 1)
    tensor = objects[0].raw_tensor
    assert tensor is not None
    tensor.fill_(77)
    pool.begin_drain(segment.handle)
    pool.free(objects)
    assert pool.get_memory_usage() == (0, PAGE)
    retained = pool.remove_segment(segment.handle)
    assert retained.state is DRAMSegmentState.DRAINING
    assert retained.base == segment.base == tensor.data_ptr()
    assert retained.mapped_bytes == PAGE
    assert _mapped_bytes(segment.mapping_name) == PAGE
    assert torch.all(tensor == 77)
    del tensor
    assert pool.remove_segment(segment.handle).state is DRAMSegmentState.REMOVED
    pool.close()


def test_batch_spans_segments_and_rolls_back_without_enough_capacity() -> None:
    pool = SegmentedDRAML1MemoryManager()
    first = pool.add_segment(PAGE)
    second = pool.add_segment(2 * PAGE)
    try:
        assert pool.allocate(_layout(), 4) == (L1Error.OUT_OF_MEMORY, [])
        assert pool.get_memory_usage() == (0, 3 * PAGE)
        assert pool.memcheck()
        error, objects = pool.allocate(_layout(), 3)
        assert error is L1Error.SUCCESS
        assert pool.get_segment_status(first.handle).used_bytes == PAGE
        assert pool.get_segment_status(second.handle).used_bytes == 2 * PAGE
        for i, obj in enumerate(objects):
            assert obj.raw_tensor is not None
            obj.raw_tensor.fill_(i + 1)
        pool.free(objects)
        pool.free(objects)
        assert pool.memcheck()
        assert pool.get_memory_usage() == (0, 3 * PAGE)
    finally:
        pool.close()


def test_drain_stops_admission_and_close_retries_busy_segments() -> None:
    pool = SegmentedDRAML1MemoryManager()
    first = pool.add_segment(2 * PAGE)
    _, objects = pool.allocate(_layout(), 1)
    pool.begin_drain(first.handle)
    assert pool.get_active_capacity() == 0
    assert pool.allocate(_layout(), 1) == (L1Error.OUT_OF_MEMORY, [])
    assert pool.get_memory_usage() == (PAGE, 2 * PAGE)
    with pytest.raises(RuntimeError, match="still have"):
        pool.close()
    assert _mapped_bytes(first.mapping_name) == 2 * PAGE
    with pytest.raises(RuntimeError, match="closed"):
        pool.add_segment(PAGE)
    pool.free(objects)
    pool.close()
    pool.close()
    assert pool.get_memory_usage() == (0, 0)
    assert _mapped_bytes(first.mapping_name) == 0


def test_stale_handles_and_object_frees_never_affect_replacement() -> None:
    pool = SegmentedDRAML1MemoryManager()
    first = pool.add_segment(PAGE)
    _, old_objects = pool.allocate(_layout(), 1)
    pool.free(old_objects)
    pool.begin_drain(first.handle)
    pool.remove_segment(first.handle)
    try:
        for _ in range(10):
            current = pool.add_segment(PAGE)
            assert current.handle.sequence > first.handle.sequence
            _, current_objects = pool.allocate(_layout(), 1)
            assert current_objects[0].raw_tensor is not None
            current_objects[0].raw_tensor.fill_(91)
            pool.begin_drain(first.handle)
            pool.remove_segment(first.handle)
            pool.free(old_objects)
            assert pool.get_memory_usage() == (PAGE, PAGE)
            assert torch.all(current_objects[0].raw_tensor == 91)
            pool.free(current_objects)
            pool.begin_drain(current.handle)
            pool.remove_segment(current.handle)
        assert pool.get_segment_statuses() == []
        assert pool.get_segment_status(first.handle).state is DRAMSegmentState.REMOVED
        for invalid in (
            DRAMSegmentHandle(uuid4(), 1),
            DRAMSegmentHandle(first.handle.owner, 0),
            DRAMSegmentHandle(first.handle.owner, 100),
        ):
            with pytest.raises(KeyError):
                pool.remove_segment(invalid)
    finally:
        pool.close()


@pytest.mark.parametrize("size", [0, -PAGE, PAGE - 1, PAGE + 1])
def test_invalid_segment_size_never_adds_capacity(size: int) -> None:
    pool = SegmentedDRAML1MemoryManager()
    with pytest.raises(ValueError, match="page size"):
        pool.add_segment(size)
    assert pool.get_memory_usage() == (0, 0)
    pool.close()


@pytest.mark.parametrize("alignment", [0, -1, 3, PAGE * 2])
def test_invalid_alignment(alignment: int) -> None:
    with pytest.raises(ValueError, match="alignment"):
        SegmentedDRAML1MemoryManager(align_bytes=alignment)


def test_remove_requires_drain_and_small_allocations_count_alignment() -> None:
    pool = SegmentedDRAML1MemoryManager()
    segment = pool.add_segment(PAGE)
    with pytest.raises(ValueError, match="begin_drain"):
        pool.remove_segment(segment.handle)
    _, objects = pool.allocate(_layout(1), 1)
    assert pool.get_memory_usage() == (PAGE, PAGE)
    assert objects[0].get_size() == 1
    pool.free(objects)
    pool.close()


@pytest.mark.parametrize("failure", ["allocator", "mmap"])
def test_failed_add_retains_no_mapping_or_descriptor(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    pool = SegmentedDRAML1MemoryManager()
    before_fds = set(os.listdir("/proc/self/fd"))
    before_maps = [
        line
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/memfd:lmcache_dram_" in line
    ]

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected construction failure")

    with monkeypatch.context() as patch:
        if failure == "allocator":
            patch.setattr(TensorMemoryAllocator, "__init__", fail)
        else:
            patch.setattr(mmap, "mmap", fail)
        with pytest.raises(RuntimeError, match="injected"):
            pool.add_segment(PAGE)
    assert set(os.listdir("/proc/self/fd")) == before_fds
    assert [
        line
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/memfd:lmcache_dram_" in line
    ] == before_maps
    assert pool.get_memory_usage() == (0, 0)
    first = pool.add_segment(PAGE)
    assert first.handle.sequence == 1
    pool.close()


def test_reference_count_release_preserves_pool_accounting() -> None:
    pool = SegmentedDRAML1MemoryManager()
    segment = pool.add_segment(PAGE)
    _, objects = pool.allocate(_layout(), 1)
    pool.begin_drain(segment.handle)
    objects[0].ref_count_down()
    assert pool.get_memory_usage() == (0, PAGE)
    assert pool.remove_segment(segment.handle).state is DRAMSegmentState.REMOVED
    pool.close()
