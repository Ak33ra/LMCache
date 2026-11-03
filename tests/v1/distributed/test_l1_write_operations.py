# SPDX-License-Identifier: Apache-2.0
"""Unpublished write ownership with real CPU allocations and native TTL locks."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1WriteOperation
from lmcache.v1.distributed.l1_manager import L1Manager

LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
KEY = ObjectKey(ObjectKey.IntHash2Bytes(1), "write-lifetime", 0)


@pytest.fixture
def manager() -> Iterator[L1Manager]:
    """Use one slot so premature release immediately permits reuse."""
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False, shm_name=""),
            write_ttl_seconds=1,
        )
    )
    yield manager
    manager.close()


def reserve(manager: L1Manager, tag: str = "load") -> L1WriteOperation:
    """Reserve a destination and require its capability."""
    error, handle = manager.reserve_write_operations([KEY], [False], LAYOUT, tag)[KEY]
    assert error == L1Error.SUCCESS and handle is not None
    return handle


def test_paused_writer_survives_ttl_and_all_free_paths(manager: L1Manager) -> None:
    """Late writes cannot target reused or prematurely published storage."""
    handle = reserve(manager)
    tensor = handle.memory_obj.get_tensor(0)
    assert tensor is not None
    tensor.fill_(3)
    release = threading.Event()
    finished = threading.Event()

    def write() -> None:
        release.wait()
        tensor.fill_(7)
        finished.set()

    worker = threading.Thread(target=write)
    worker.start()
    try:
        time.sleep(1.1)
        assert manager.reserve_read([KEY])[KEY][0] == L1Error.KEY_NOT_EXIST
        assert not manager.is_key_evictable(KEY)
        for force in (False, True):
            manager.clear(force=force)
            assert manager.delete([KEY], force=force)[KEY] == L1Error.KEY_IS_LOCKED
            assert manager.get_memory_usage()[0] == 4096
        for finish in (
            manager.finish_write,
            manager.finish_write_and_delete,
            manager.finish_write_and_reserve_read,
        ):
            result = finish([KEY], tag="load")[KEY]
            assert (result[0] if isinstance(result, tuple) else result) == (
                L1Error.KEY_IN_WRONG_STATE
            )
        with pytest.raises(RuntimeError, match="active write operations"):
            manager.close()
        assert manager.reserve_write([KEY], [False], LAYOUT, "load")[KEY][0] == (
            L1Error.KEY_NOT_WRITABLE
        )
        assert manager.reserve_write([KEY], [False], LAYOUT, "other")[KEY][0] == (
            L1Error.OUT_OF_MEMORY
        )
    finally:
        release.set()
        worker.join(timeout=5)
    assert finished.is_set()
    result = manager.finish_write_operations([handle])[handle]
    assert result[0] == L1Error.SUCCESS and result[1] is handle.memory_obj
    assert torch.all(tensor == 7)
    manager.finish_read([KEY])
    manager.delete([KEY])
    newer = reserve(manager)
    with pytest.raises(ValueError, match="not active"):
        manager.finish_write_operations([handle])
    assert manager.get_staging_memory_usage() == 4096
    manager.finish_write_operations([newer], discard=frozenset([newer]))
    assert manager.get_memory_usage()[0] == 0


def test_invalid_completion_is_atomic(manager: L1Manager) -> None:
    """Copied, foreign, repeated and stale capabilities cannot release a writer."""
    handle = reserve(manager)
    other = L1Manager(manager.config)
    foreign = reserve(other)
    try:
        for invalid in (replace(handle), replace(handle, manager_id=-1), foreign):
            with pytest.raises(ValueError):
                manager.finish_write_operations([handle, invalid])
            assert manager.report_status()["active_write_operation_count"] == 1
        with pytest.raises(ValueError, match="Repeated"):
            manager.finish_write_operations([handle, handle])
        with pytest.raises(ValueError, match="outside"):
            manager.finish_write_operations([handle], discard=frozenset([foreign]))
        assert manager.get_memory_usage()[0] == 4096
    finally:
        other.finish_write_operations([foreign], discard=frozenset([foreign]))
        other.close()
        manager.finish_write_operations([handle], discard=frozenset([handle]))


def test_concurrent_same_key_admission_keeps_resident() -> None:
    """A competing writer can win admission without being changed by completion."""
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=8192, use_lazy=False, shm_name="")
        )
    )
    first, second = reserve(manager, "first"), reserve(manager, "second")
    first.memory_obj.get_tensor(0).fill_(3)
    second.memory_obj.get_tensor(0).fill_(9)
    manager.finish_write_operations([first])
    manager.finish_read([KEY])
    assert not manager.is_key_evictable(KEY)
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    result = manager.finish_write_operations([second])[second]
    assert result[1] is first.memory_obj
    assert torch.all(result[1].get_tensor(0) == 3)
    assert manager.get_memory_usage()[0] == 4096
    assert manager.unsafe_read([KEY])[KEY][1] is first.memory_obj
    assert not manager.is_key_evictable(KEY)
    manager.finish_read([KEY])
    assert manager.is_key_evictable(KEY)
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS
    assert manager.get_memory_usage()[0] == 0
    manager.close()


def test_zero_read_lock_admission_and_negative_rejection(manager: L1Manager) -> None:
    """Normal stores publish without a TTL reader; invalid counts retain ownership."""
    handle = reserve(manager)
    with pytest.raises(ValueError, match="nonnegative"):
        manager.finish_write_operations([handle], read_locks=-1)
    assert manager.get_staging_memory_usage() == 4096
    result = manager.finish_write_operations([handle], read_locks=0)
    assert result[handle][0] == L1Error.SUCCESS
    assert manager.is_key_evictable(KEY)
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS
