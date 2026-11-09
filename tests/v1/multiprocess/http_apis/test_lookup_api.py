# SPDX-License-Identifier: Apache-2.0
"""Exercise receipt lifetime through HTTP and the real session manager."""

# Standard
from types import SimpleNamespace

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.http_apis.lookup_api import router
from lmcache.v1.multiprocess.session import LookupObservation, SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def test_observation_lifetime() -> None:
    sessions = SessionManager(TokenHasher(4, "blake3"), cleanup_interval=None)
    app = FastAPI()
    app.include_router(router)
    app.state.context = SimpleNamespace(
        engine=SimpleNamespace(context=SimpleNamespace(session_manager=sessions))
    )
    try:
        with TestClient(app) as client:
            assert client.get("/cache/lookups/r").status_code == 404
            session = sessions.get_or_create("r")
            assert client.get("/cache/lookups/r").status_code == 409
            session.record_prefetch_result(
                4, (0,), observation=LookupObservation(4, (2,), (2,))
            )
            assert client.get("/cache/lookups/r").json() == {
                "request_id": "r",
                "hit_chunks": 4,
                "l1_resident_keys": [2],
                "l2_loaded_keys": [2],
            }
            key = IPCCacheServerKey.from_token_ids(
                "m", 1, 0, [1, 2, 3, 4], end=4, request_id="r"
            )
            session.begin_lookup(key, (-1,))
            assert client.get("/cache/lookups/r").status_code == 409
            sessions.remove("r")
            assert client.get("/cache/lookups/r").status_code == 404
    finally:
        sessions.close()
