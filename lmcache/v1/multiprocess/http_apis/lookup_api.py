# SPDX-License-Identifier: Apache-2.0
"""Read a consumed lookup result without extending its session lifetime."""

# Standard
from dataclasses import asdict

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.v1.multiprocess.http_apis.dependencies import get_context

router = APIRouter()


@router.get("/cache/lookups/{request_id}")
async def lookup_observation(request_id: str, request: Request) -> dict:
    """Return interface-origin counts for a live session's consumed lookup.

    Returns HTTP 404 for an absent session and 409 before result consumption.
    L1 resident keys can be temporary; L2 keys have unspecified physical locality.
    """
    sessions = get_context(request).engine.context.session_manager
    session = sessions.get(request_id)
    if session is None:
        raise HTTPException(404, "Unknown lookup session")
    observation = session.get_lookup_observation()
    if observation is None:
        raise HTTPException(409, "Lookup result has not been consumed")
    return {"request_id": request_id, **asdict(observation)}
