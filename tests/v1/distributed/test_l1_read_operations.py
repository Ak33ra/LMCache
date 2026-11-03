# SPDX-License-Identifier: Apache-2.0
"""Allocation-bound read lifetimes using real CPU buffers and native TTL locks."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import lmcache_native, torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ReadOperation
from lmcache.v1.distributed.l1_manager import MAX_READ_LOCK_COUNT, L1Manager

if not getattr(lmcache_native, "__file__", None):
    pytest.skip("Requires compiled lmcache_native extension", allow_module_level=True)
if torch_device_type != "cpu" and not torch_dev.is_available():
    pytest.skip("Requires an available allocation backend", allow_module_level=True)

LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
KEY = ObjectKey(ObjectKey.IntHash2Bytes(1), "operation-lifetime", 0)


@pytest.fixture
def manager() -> Iterator[L1Manager]:
    """Own one physical slot so any unsafe release allows immediate reuse."""
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False, shm_name=""),
            read_ttl_seconds=1,
        )
    )
    yield manager
    manager.close()


def populate(manager: L1Manager, temporary: bool = False) -> None:
    """Fill and publish the single slot with recognizable bytes."""
    error, obj = manager.reserve_write([KEY], [temporary], LAYOUT)[KEY]
    assert error == L1Error.SUCCESS and obj is not None
    tensor = obj.get_tensor(0)
    assert tensor is not None
    tensor.fill_(3)
    assert manager.finish_write([KEY])[KEY] == L1Error.SUCCESS


def acquire(manager: L1Manager) -> L1ReadOperation:
    """Require one allocation-specific read capability."""
    error, operation = manager.acquire_read_operations([KEY])[KEY]
    assert error == L1Error.SUCCESS and operation is not None
    return operation


def test_operation_survives_ttl_force_clear_and_failed_close(
    manager: L1Manager,
) -> None:
    """A paused operation owns its bytes even after an ordinary reader expires."""
    populate(manager)
    manager.reserve_read([KEY])
    operation = acquire(manager)
    tensor = operation.memory_obj.get_tensor(0)
    assert tensor is not None
    pointer = tensor.data_ptr()
    time.sleep(1.1)
    assert not manager.is_key_evictable(KEY)
    for force in (False, True):
        assert manager.delete([KEY], force=force)[KEY] == L1Error.KEY_IS_LOCKED
        manager.clear(force=force)
        assert manager.get_memory_usage()[0] == 4096
    with pytest.raises(RuntimeError, match="active read operations"):
        manager.close()
    replacement = ObjectKey(ObjectKey.IntHash2Bytes(2), "operation-lifetime", 0)
    assert manager.reserve_write([replacement], [False], LAYOUT)[replacement][0] == (
        L1Error.OUT_OF_MEMORY
    )
    assert torch.all(tensor == 3)
    manager.finish_read_operations([operation])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS
    populate(manager)
    current = acquire(manager)
    assert current.memory_obj.data_ptr == pointer
    with pytest.raises(ValueError, match="not active"):
        manager.finish_read_operations([operation])
    assert manager.delete([KEY], force=True)[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_read_operations([current])


def test_invalid_completion_batch_releases_nothing(manager: L1Manager) -> None:
    """Copied, repeated and foreign capabilities cannot partially release a batch."""
    populate(manager)
    first, second = acquire(manager), acquire(manager)
    for invalid in (replace(first), replace(first, manager_id=-1)):
        with pytest.raises(ValueError):
            manager.finish_read_operations([second, invalid])
        assert manager.report_status()["active_read_operation_count"] == 2
    with pytest.raises(ValueError, match="Repeated"):
        manager.finish_read_operations([first, first])
    with pytest.raises(ValueError, match="outside"):
        manager.finish_read_operations([first], discard=frozenset([second]))
    other = L1Manager(manager.config)
    populate(other)
    foreign = acquire(other)
    try:
        with pytest.raises(ValueError, match="not active"):
            manager.finish_read_operations([second, foreign])
        assert manager.report_status()["active_read_operation_count"] == 2
    finally:
        other.finish_read_operations([foreign])
        other.close()
    manager.finish_read_operations([first, second])
    assert manager.report_status()["active_read_operation_count"] == 0


def test_operation_admission_is_bounded(manager: L1Manager) -> None:
    """Outstanding handles are capped per live allocation and leave no tombstones."""
    populate(manager)
    operations = [acquire(manager) for _ in range(MAX_READ_LOCK_COUNT)]
    assert manager.acquire_read_operations([KEY])[KEY] == (L1Error.KEY_IS_LOCKED, None)
    manager.finish_read_operations(operations)
    operation = acquire(manager)
    manager.finish_read_operations([operation])
    assert manager.report_status()["active_read_operation_count"] == 0


def test_temporary_reader_waits_for_operation(manager: L1Manager) -> None:
    """Ending the session's last TTL read does not free an active operation."""
    populate(manager, temporary=True)
    manager.reserve_read([KEY])
    operation = acquire(manager)
    manager.finish_read([KEY])
    assert manager.get_memory_usage()[0] == 4096
    manager.finish_read_operations([operation])
    assert manager.get_memory_usage()[0] == 0


def test_discard_never_follows_a_reused_key(manager: L1Manager) -> None:
    """Deferred discard belongs to the old allocation even if its key is reused."""
    populate(manager)
    manager.reserve_read([KEY])
    operation = acquire(manager)
    manager.finish_read_operations([operation], discard=frozenset([operation]))
    assert manager.reclaim_discarded_read_operations() == 1
    manager.clear(force=True)
    populate(manager)
    assert manager.reclaim_discarded_read_operations() == 0
    assert manager.get_memory_usage()[0] == 4096


def test_discard_reclaims_after_legacy_reader_expiry(manager: L1Manager) -> None:
    """An abandoned TTL reader eventually releases a completed discard operation."""
    populate(manager)
    manager.reserve_read([KEY])
    operation = acquire(manager)
    manager.finish_read_operations([operation], discard=frozenset([operation]))
    assert manager.reclaim_discarded_read_operations() == 1
    time.sleep(1.1)
    assert manager.reclaim_discarded_read_operations() == 0
    assert manager.get_memory_usage()[0] == 0
