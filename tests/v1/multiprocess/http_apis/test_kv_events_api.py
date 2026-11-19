# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmcache.v1.multiprocess.custom_types import KVEventBatch, KVEventRecord
from lmcache.v1.multiprocess.http_apis.kv_events_api import router


class _Engine:
    def read_kv_events(self, model_name, cursor, max_events):
        assert (model_name, cursor, max_events) == ("model/a", 7, 2)
        return KVEventBatch(
            enabled=True,
            incarnation=11,
            next_cursor=8,
            lost=False,
            events=[
                KVEventRecord(8, "stored", "cpu", "model/a", [b"\x00\xff"],
                              b"\x01", [3, 4])
            ],
        )


def _client(engine=None):
    app = FastAPI()
    app.include_router(router)
    if engine is not None:
        app.state.engine = engine
    return TestClient(app)


def test_kv_events_returns_bounded_json_page():
    response = _client(_Engine()).get(
        "/kv-events", params={"model_name": "model/a", "cursor": 7,
                               "max_events": 2}
    )
    assert response.status_code == 200
    assert response.json() == {
        "enabled": True,
        "incarnation": 11,
        "next_cursor": 8,
        "lost": False,
        "events": [{"seq": 8, "kind": "stored", "medium": "cpu",
                    "model_name": "model/a", "block_hashes": ["00ff"],
                    "parent_block_hash": "01", "token_ids": [3, 4]}],
    }


def test_kv_events_rejects_unbounded_page_and_missing_engine():
    assert _client(_Engine()).get(
        "/kv-events", params={"model_name": "m", "cursor": 0,
                               "max_events": 1025}
    ).status_code == 422
    assert _client().get(
        "/kv-events", params={"model_name": "m", "cursor": 0}
    ).status_code == 503
