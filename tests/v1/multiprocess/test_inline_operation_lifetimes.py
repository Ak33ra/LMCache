# SPDX-License-Identifier: Apache-2.0
"""Real CPU inline copies retain their allocations beyond reservation TTLs."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1WriteOperationResult,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.server_transfer import InlineTransferStrategy
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
    decode_tensors,
    encode_tensors,
)

LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
KEY = ObjectKey(ObjectKey.IntHash2Bytes(1), "inline-lifetime", 0)
OTHER = ObjectKey(ObjectKey.IntHash2Bytes(2), "inline-lifetime", 0)
IPC_KEY = IPCCacheServerKey.from_token_ids(
    "inline-lifetime", 1, 0, [1], end=1, request_id="r"
)
CONTEXT = EngineDrivenContextMetadata(LAYOUT, 1, False)


@pytest.fixture
def storage() -> Iterator[tuple[StorageManager, L1Manager]]:
    """A one-slot native CPU allocator makes premature reuse observable."""
    config = L1ManagerConfig(
        L1MemoryManagerConfig(4096, False, shm_name=""),
        write_ttl_seconds=1,
        read_ttl_seconds=1,
    )
    manager = L1Manager(config)
    store = StorageManager(
        StorageManagerConfig(config, EvictionConfig("noop")),
        _l1_managers=(manager,),
    )
    try:
        yield store, manager
    finally:
        store.close()


@pytest.mark.parametrize("direction", ["store", "retrieve"])
def test_inline_copy_survives_expiry_and_forced_clear(
    storage: tuple[StorageManager, L1Manager],
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
) -> None:
    """Force-clear and cancellation-style unlock cannot recycle an active copy."""
    store, manager = storage
    strategy = InlineTransferStrategy(store)
    expected = torch.arange(1024, dtype=torch.float32)
    payload = encode_tensors([expected])
    if direction == "retrieve":
        assert strategy.commit_store(IPC_KEY, 1, payload, CONTEXT, lambda _: [KEY])
        manager.reserve_read([KEY])
    entered, release = threading.Event(), threading.Event()
    method = "copy_" if direction == "store" else "clone"
    original = getattr(torch.Tensor, method)

    def paused(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        entered.set()
        assert release.wait(10)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, method, paused)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            strategy.commit_store
            if direction == "store"
            else strategy.prepare_retrieve,
            *(
                [IPC_KEY, 1, payload, CONTEXT, lambda _: [KEY]]
                if direction == "store"
                else [IPC_KEY, 1, lambda _: [KEY]]
            ),
        )
        try:
            assert entered.wait(5)
            time.sleep(1.1)
            assert not future.cancel()
            manager.finish_read([KEY])
            store.clear(force=True)
            assert store.delete_l1_keys([KEY], force=True) == (0, 1)
            assert store.get_l1_usage() == (4096, 4096)
            assert store.reserve_write_operations([OTHER], LAYOUT) == {}
        finally:
            release.set()
        result = future.result(timeout=5)
    monkeypatch.setattr(torch.Tensor, method, original)
    if direction == "store":
        assert result is True
        manager.reserve_read([KEY])
        result = strategy.prepare_retrieve(IPC_KEY, 1, lambda _: [KEY])
    assert result.success
    (actual,) = decode_tensors(
        result.data,
        expected_shape=LAYOUT.shapes[0],
        expected_dtype=torch.float32,
        expected_count=1,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    status = manager.report_status()
    assert status["active_write_operation_count"] == 0
    assert status["active_read_operation_count"] == 0
    store.clear(force=True)
    assert store.get_l1_usage()[0] == 0
    replacement = store.reserve_write_operations([OTHER], LAYOUT)
    assert OTHER in replacement
    store.finish_write_operations(replacement, written_keys=frozenset())


@pytest.mark.parametrize("direction", ["store", "retrieve"])
def test_inline_copy_failure_releases_exact_operations(
    storage: tuple[StorageManager, L1Manager],
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
) -> None:
    """An exception after pointer acquisition leaves no live operation."""
    store, manager = storage
    strategy = InlineTransferStrategy(store)
    payload = encode_tensors([torch.ones(1024)])
    if direction == "retrieve":
        assert strategy.commit_store(IPC_KEY, 1, payload, CONTEXT, lambda _: [KEY])
        manager.reserve_read([KEY])

    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("copy failed")

    monkeypatch.setattr(
        torch.Tensor, "copy_" if direction == "store" else "clone", fail
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        if direction == "store":
            strategy.commit_store(IPC_KEY, 1, payload, CONTEXT, lambda _: [KEY])
        else:
            strategy.prepare_retrieve(IPC_KEY, 1, lambda _: [KEY])
    status = manager.report_status()
    assert status["active_write_operation_count"] == 0
    assert status["active_read_operation_count"] == 0
    store.clear()
    assert store.get_l1_usage()[0] == 0


def test_partial_read_miss_releases_acquired_operations(
    storage: tuple[StorageManager, L1Manager],
) -> None:
    """A missing key does not strand other source operations or TTL locks."""
    store, manager = storage
    strategy = InlineTransferStrategy(store)
    assert strategy.commit_store(
        IPC_KEY, 1, encode_tensors([torch.ones(1024)]), CONTEXT, lambda _: [KEY]
    )
    manager.reserve_read([KEY])
    assert not strategy.prepare_retrieve(IPC_KEY, 1, lambda _: [KEY, OTHER]).success
    store.clear()
    assert store.get_l1_usage()[0] == 0


def test_overflow_allocation_failure_discards_earlier_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exceptional overflow target cannot strand the primary allocation."""
    configs = [
        L1ManagerConfig(L1MemoryManagerConfig(4096, False, shm_name=""), tag=tag)
        for tag in ("_default", "overflow")
    ]
    managers = tuple(L1Manager(config) for config in configs)
    store = StorageManager(
        StorageManagerConfig(configs[0], EvictionConfig("noop")),
        _l1_managers=managers,
    )
    primary, overflow = managers
    reserve = primary.reserve_write_operations

    def partial(
        keys: list[ObjectKey],
        temporary: list[bool],
        layout: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1WriteOperationResult]:
        result = reserve(keys[:1], temporary[:1], layout, tag)
        result.update({key: (L1Error.OUT_OF_MEMORY, None) for key in keys[1:]})
        return result

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("allocator failed")

    monkeypatch.setattr(primary, "reserve_write_operations", partial)
    monkeypatch.setattr(overflow, "reserve_write_operations", fail)
    try:
        with pytest.raises(RuntimeError, match="allocator failed"):
            store.reserve_write_operations([KEY, OTHER], LAYOUT)
        assert store.get_l1_usage()[0] == 0
        assert primary.report_status()["active_write_operation_count"] == 0
    finally:
        store.close()


def test_zero_lock_store_admission_notifies_store_listeners(
    storage: tuple[StorageManager, L1Manager],
) -> None:
    """Operation-backed inline stores keep ordinary store publication events."""
    store, manager = storage
    listener = MagicMock(spec=L1ManagerListener)
    manager.register_listener(listener)
    operations = store.reserve_write_operations([KEY], LAYOUT)
    store.finish_write_operations(operations, written_keys=frozenset([KEY]))
    assert manager.report_status()["read_locked_count"] == 0
    listener.on_l1_keys_write_finished.assert_called_once_with([KEY])
    listener.on_l1_keys_finish_write_and_reserve_read.assert_not_called()


def test_inline_reads_follow_exact_multi_l1_owners() -> None:
    """Overflow writes and inline clones address the same two real CPU pools."""
    configs = [
        L1ManagerConfig(L1MemoryManagerConfig(4096, False, shm_name=""), tag=tag)
        for tag in ("_default", "overflow")
    ]
    managers = tuple(L1Manager(config) for config in configs)
    store = StorageManager(
        StorageManagerConfig(configs[0], EvictionConfig("noop")),
        _l1_managers=managers,
    )
    owners = {}
    strategy = InlineTransferStrategy(store, lambda _: owners)
    try:
        for index, (key, manager) in enumerate(
            zip((KEY, OTHER), managers, strict=True)
        ):
            assert strategy.commit_store(
                IPC_KEY,
                1,
                encode_tensors([torch.full((1024,), float(index))]),
                CONTEXT,
                lambda _, key=key: [key],
            )
            manager.reserve_read([key])
            owners[key] = manager.l1_manager_id
        result = strategy.prepare_retrieve(IPC_KEY, 1, lambda _: [KEY, OTHER])
        assert result.success
        tensors = decode_tensors(
            result.data,
            expected_shape=LAYOUT.shapes[0],
            expected_dtype=torch.float32,
            expected_count=2,
        )
        for index, tensor in enumerate(tensors):
            assert torch.all(tensor == index)
        store.clear()
        assert store.get_l1_usage()[0] == 0
    finally:
        store.close()
