# SPDX-License-Identifier: Apache-2.0
"""Lookup completions cannot publish into a replaced or removed session."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle, PrefetchResult
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.http_apis.lookup_api import router
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def _result(hits: int) -> PrefetchResult:
    return PrefetchResult(
        hit_cells=[Bitmap(2, hits)],
        l1_hit_cells=[Bitmap(2, hits)],
        l2_hit_cells=[Bitmap(2)],
    )


@pytest.fixture
def lookup_case():
    ctx = MagicMock()
    ctx.chunk_size = 4
    ctx.token_hasher = TokenHasher(4, "blake3")
    ctx.session_manager = SessionManager(ctx.token_hasher, cleanup_interval=None)
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.storage_manager.submit_prefetch_task.side_effect = [
        PrefetchHandle(i, "r", 2, 0.0) for i in range(3)
    ]
    ctx.storage_manager.query_prefetch_status.side_effect = lambda handle: _result(
        1 if handle.prefetch_request_id == 0 else 2
    )
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, None, list(range(8)), end=8, request_id="r", num_kv_readers=1
    )
    module = LookupModule(ctx)
    app = FastAPI()
    app.include_router(router)
    app.state.context = SimpleNamespace(engine=SimpleNamespace(context=ctx))
    try:
        with TestClient(app) as client:
            yield ctx, key, module, client
    finally:
        ctx.session_manager.close()


@pytest.mark.parametrize("removal", ["end", "expire"])
def test_late_completion_does_not_recreate_session(lookup_case, removal):
    ctx, key, module, client = lookup_case
    module.lookup(key, 1)
    if removal == "end":
        module.end_session("r")
    else:
        ctx.session_manager.get("r").created_at = 0
        assert ctx.session_manager.cleanup_expired() == 1

    assert client.get("/cache/lookups/r").status_code == 404
    assert module.query_prefetch_status("r") == 0
    assert client.get("/cache/lookups/r").status_code == 404
    assert ctx.session_manager.active_count() == 0
    assert module.report_status()["active_prefetch_jobs"] == 0
    ctx.storage_manager.finish_read_prefetched.assert_not_called()


@pytest.mark.parametrize("poll", ["query", "wait"])
@pytest.mark.parametrize("replacement", ["lookup", "end", "expire"])
def test_delayed_completion_preserves_replacement(lookup_case, poll, replacement):
    ctx, key, module, client = lookup_case
    entered, release = Event(), Event()

    def delayed(handle, *args):
        if handle.prefetch_request_id == 0:
            entered.set()
            assert release.wait(5)
        return True if poll == "wait" else _result(1)

    if poll == "wait":
        ctx.storage_manager.wait_prefetch_status.side_effect = delayed
    else:
        ctx.storage_manager.query_prefetch_status.side_effect = delayed
    module.lookup(key, 1)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(
                module.wait_prefetch_status
                if poll == "wait"
                else module.query_prefetch_status,
                *(["r", 5] if poll == "wait" else ["r"]),
            )
            assert entered.wait(5)
            if replacement == "end":
                module.end_session("r")
            elif replacement == "expire":
                ctx.session_manager.get("r").created_at = 0
                assert ctx.session_manager.cleanup_expired() == 1
            module.lookup(key, 1)
            assert client.get("/cache/lookups/r").status_code == 409
            release.set()
            assert pending.result(5) == 0
    finally:
        release.set()

    assert client.get("/cache/lookups/r").status_code == 409
    assert module.report_status()["active_prefetch_jobs"] == 1
    ctx.storage_manager.query_prefetch_status.side_effect = lambda handle: _result(2)
    assert module.query_prefetch_status("r") == 2
    assert client.get("/cache/lookups/r").json()["hit_chunks"] == 2
    assert module.report_status()["active_prefetch_jobs"] == 0
    ctx.storage_manager.finish_read_prefetched.assert_not_called()


@pytest.mark.parametrize("early_exit", ["empty", "no_layout", "no_group_layouts"])
def test_early_exit_clears_previous_receipt(lookup_case, early_exit):
    ctx, key, module, client = lookup_case
    module.lookup(key, 1)
    assert module.query_prefetch_status("r") == 1
    assert client.get("/cache/lookups/r").status_code == 200
    if early_exit == "empty":
        key = replace(key, token_bytes=b"", end=0)
    elif early_exit == "no_layout":
        ctx.layout_desc_registry.find.return_value = None
        ctx.layout_desc_registry.find_attn_desc.side_effect = ValueError("no layout")
    else:
        ctx.layout_desc_registry.find_group_layout_descs.return_value = {}

    module.lookup(key, 1)

    assert client.get("/cache/lookups/r").status_code == 409
    ctx.storage_manager.query_prefetch_status.side_effect = lambda handle: _result(0)
    assert module.query_prefetch_status("r") == 0
    assert client.get("/cache/lookups/r").json()["hit_chunks"] == 0
    module.end_session("r")
    assert client.get("/cache/lookups/r").status_code == 404


def test_delayed_submission_does_not_replace_new_job(lookup_case):
    ctx, key, module, client = lookup_case
    entered, release = Event(), Event()

    def delayed_submit(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return PrefetchHandle(0, "r", 2, 0.0)

    ctx.storage_manager.submit_prefetch_task.side_effect = delayed_submit
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(module.lookup, key, 1)
            assert entered.wait(5)
            ctx.storage_manager.submit_prefetch_task.side_effect = None
            ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
                1, "r", 2, 0.0
            )
            module.lookup(key, 1)
            release.set()
            pending.result(5)
    finally:
        release.set()

    assert module.query_prefetch_status("r") == 2
    assert client.get("/cache/lookups/r").json()["hit_chunks"] == 2


@pytest.mark.parametrize("reuse_session_id", [False, True])
def test_completion_during_replacement_submission(lookup_case, reuse_session_id):
    ctx, key, module, client = lookup_case
    query_entered, query_release = Event(), Event()
    submit_entered, submit_release = Event(), Event()
    module.lookup(key, 1)

    def delayed_query(handle):
        if handle.prefetch_request_id == 0:
            query_entered.set()
            assert query_release.wait(5)
        return _result(1 if handle.prefetch_request_id == 0 else 2)

    def delayed_submit(*args, **kwargs):
        submit_entered.set()
        assert submit_release.wait(5)
        return PrefetchHandle(1, "r", 2, 0.0)

    ctx.storage_manager.query_prefetch_status.side_effect = delayed_query
    ctx.storage_manager.submit_prefetch_task.side_effect = delayed_submit
    try:
        with ThreadPoolExecutor(2) as pool:
            old_query = pool.submit(module.query_prefetch_status, "r")
            assert query_entered.wait(5)
            if reuse_session_id:
                module.end_session("r")
            new_lookup = pool.submit(module.lookup, key, 1)
            assert submit_entered.wait(5)
            assert client.get("/cache/lookups/r").status_code == 409
            query_release.set()
            assert old_query.result(5) == 0
            assert client.get("/cache/lookups/r").status_code == 409
            submit_release.set()
            new_lookup.result(5)
    finally:
        query_release.set()
        submit_release.set()

    assert module.query_prefetch_status("r") == 2
    assert client.get("/cache/lookups/r").json()["hit_chunks"] == 2
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
