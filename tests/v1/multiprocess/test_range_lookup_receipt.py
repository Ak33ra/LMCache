# SPDX-License-Identifier: Apache-2.0
"""Generation-bound receipts over local range lookup and its typed transport."""

# Standard
from dataclasses import replace
from unittest.mock import MagicMock
import socket

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchResult
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import LookupObservation, SessionManager
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

# Test helpers
from tests.v1.multiprocess.transport_test_utils import start_lookup_request_server


@pytest.fixture
def context() -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = 4
    ctx.session_manager = SessionManager(
        TokenHasher(4, "blake3"), cleanup_interval=None
    )
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc([-1], 1)
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.storage_manager.wait_prefetch_status.return_value = True
    ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
        hit_cells=[Bitmap(2, 1)],
        l1_hit_cells=[Bitmap(2, 1)],
        l2_hit_cells=[Bitmap(2)],
    )
    return ctx


@pytest.fixture
def key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "model",
        1,
        None,
        list(range(24)),
        start=16,
        end=24,
        request_id="range",
        num_kv_readers=1,
    )


def test_receipt_survives_later_lookup_after_consumption(
    context: MagicMock, key: IPCCacheServerKey
) -> None:
    module = LookupModule(context)
    replacement = replace(key, token_bytes=pack_token_ids([99] * 24))

    def replace_after_consume(event) -> None:
        if event.event_type == EventType.MP_LOOKUP_PREFETCH_END:
            module.lookup(replacement, 1)

    context.event_bus.publish.side_effect = replace_after_consume
    receipt = module.lookup_range(key, 1, 1.0)
    assert receipt == LookupObservation(5, (1,), (0,))
    assert context.session_manager.get("range").get_lookup_observation() is None


def test_replaced_lookup_cannot_return_replacement_receipt(
    context: MagicMock, key: IPCCacheServerKey
) -> None:
    module = LookupModule(context)
    replacement = replace(key, token_bytes=pack_token_ids([99] * 24))

    def replace_before_consume(handle, timeout) -> bool:
        module.lookup(replacement, 1)
        return True

    context.storage_manager.wait_prefetch_status.side_effect = replace_before_consume
    with pytest.raises(ValueError, match="replaced or consumed"):
        module.lookup_range(key, 1, 1.0)
    assert context.session_manager.get("range").get_lookup_observation() is None
    assert module.query_prefetch_status("range") == 5


def test_timeout_preserves_job_for_existing_wait_endpoint(
    context: MagicMock, key: IPCCacheServerKey
) -> None:
    module = LookupModule(context)
    context.storage_manager.wait_prefetch_status.return_value = False
    assert module.lookup_range(key, 1, 0.0) is None
    assert context.session_manager.get("range").get_lookup_observation() is None
    context.storage_manager.wait_prefetch_status.return_value = True
    assert module.wait_prefetch_status("range", 1.0) == 5
    assert context.session_manager.get("range").get_lookup_observation() == (
        LookupObservation(5, (1,), (0,))
    )


def test_zmq_returns_typed_receipt_and_timeout(
    context: MagicMock, key: IPCCacheServerKey
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"tcp://127.0.0.1:{port}"
    server = start_lookup_request_server("zmq", url, LookupModule(context))
    client = RequestClientFactory.create(url)
    try:
        receipt = client.lookup_range(key, 1, 1.0).result(5)
        assert receipt == LookupObservation(5, (1,), (0,))
        assert context.session_manager.get("range").get_lookup_observation() == receipt
        context.storage_manager.wait_prefetch_status.return_value = False
        assert client.lookup_range(key, 1, 0.0).result(5) is None
    finally:
        client.close()
        server.close()
