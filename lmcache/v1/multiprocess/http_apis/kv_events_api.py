# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP access to the MP server's authoritative CPU KV journal."""

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


@router.get("/kv-events")
async def kv_events(
    request: Request,
    model_name: str,
    cursor: int = Query(ge=0),
    max_events: int = Query(default=1024, ge=1, le=1024),
) -> dict:
    """Return records after ``cursor`` without waiting for future events."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    batch = engine.read_kv_events(model_name, cursor, max_events)
    return {
        "enabled": batch.enabled,
        "incarnation": batch.incarnation,
        "next_cursor": batch.next_cursor,
        "lost": batch.lost,
        "events": [
            {
                "seq": event.seq,
                "kind": event.kind,
                "medium": event.medium,
                "model_name": event.model_name,
                "block_hashes": [value.hex() for value in event.block_hashes],
                "parent_block_hash": (
                    event.parent_block_hash.hex()
                    if event.parent_block_hash is not None
                    else None
                ),
                "token_ids": event.token_ids,
            }
            for event in batch.events
        ],
    }
