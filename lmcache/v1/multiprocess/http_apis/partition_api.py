# SPDX-License-Identifier: Apache-2.0
"""Inspect and asynchronously resize the opt-in private/shared partition."""

# Standard
from typing import cast

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# First Party
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.http_apis.dependencies import get_context

router = APIRouter()


class ResizeRequest(BaseModel):
    """One idempotency identity and a finite requested private fraction."""

    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    alpha: float = Field(ge=0, le=1, allow_inf_nan=False)


def _storage(request: Request) -> StorageManager:
    """Resolve an activated partition, or report disabled/not-ready to HTTP."""
    manager = cast(StorageManager, get_context(request).engine.storage_manager)
    if not manager.partition_enabled:
        raise HTTPException(404, "Partitioning is disabled")
    return manager


@router.get("/partition")
async def partition_status(request: Request) -> JSONResponse:
    """Read cached actual backing, pending growth, and completed/target splits."""
    return JSONResponse(jsonable_encoder(_storage(request).get_partition_status()))


@router.post("/partition/resize", status_code=202)
async def resize_partition(body: ResizeRequest, request: Request) -> JSONResponse:
    """Enqueue a resize; repeated IDs keep the same target and outcome."""
    manager = _storage(request)
    try:
        status = manager.request_partition_resize(body.request_id, body.alpha)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    return JSONResponse(jsonable_encoder(status), status_code=202)


@router.get("/partition/resize/{request_id}")
async def resize_status(request_id: str, request: Request) -> JSONResponse:
    """Read a known request without waiting for native control work."""
    try:
        status = _storage(request).get_partition_resize(request_id)
    except KeyError as error:
        raise HTTPException(404, "Unknown resize request") from error
    return JSONResponse(jsonable_encoder(status))
