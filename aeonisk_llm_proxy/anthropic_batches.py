"""
Anthropic-compatible Message Batches API proxy.

Forwards batch API requests to Anthropic's actual batch API, adding value
through API key management, unified monitoring, and results_url rewriting
so SDK clients fetch results through the proxy.
"""

import os
import logging
from typing import Optional, List

import aiohttp
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BETA = "message-batches-2024-09-24"


# --- Pydantic Models ---

class MessageBatchRequestParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    max_tokens: int
    messages: list


class MessageBatchRequestItem(BaseModel):
    custom_id: str
    params: MessageBatchRequestParams


class CreateMessageBatchRequest(BaseModel):
    requests: List[MessageBatchRequestItem]


class MessageBatchRequestCounts(BaseModel):
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


class MessageBatch(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: str = "message_batch"
    processing_status: str
    request_counts: MessageBatchRequestCounts
    ended_at: Optional[str] = None
    created_at: str
    expires_at: str
    cancel_initiated_at: Optional[str] = None
    results_url: Optional[str] = None


class MessageBatchList(BaseModel):
    data: List[MessageBatch]
    has_more: bool
    first_id: Optional[str] = None
    last_id: Optional[str] = None


# --- Service Class ---

class AnthropicBatchProxy:
    """Proxies Anthropic Message Batches API requests."""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — batch proxy will return 500s")

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": ANTHROPIC_BETA,
            "Content-Type": "application/json",
        }

    def _rewrite_results_url(self, batch_data: dict, base_url: str) -> dict:
        """Rewrite results_url to point through the proxy."""
        if batch_data.get("results_url"):
            batch_id = batch_data["id"]
            batch_data["results_url"] = f"{base_url}/v1/messages/batches/{batch_id}/results"
        return batch_data

    async def create_batch(self, payload: dict) -> tuple[int, dict]:
        """Forward batch creation to Anthropic. Returns (status_code, body)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                ANTHROPIC_BASE_URL,
                headers=self._headers(),
                json=payload,
            ) as resp:
                body = await resp.json()
                return resp.status, body

    async def get_batch(self, batch_id: str) -> tuple[int, dict]:
        """Forward batch status check. Returns (status_code, body)."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ANTHROPIC_BASE_URL}/{batch_id}",
                headers=self._headers(),
            ) as resp:
                body = await resp.json()
                return resp.status, body

    async def list_batches(
        self,
        limit: Optional[int] = None,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> tuple[int, dict]:
        """Forward list batches request. Returns (status_code, body)."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if before_id is not None:
            params["before_id"] = before_id
        if after_id is not None:
            params["after_id"] = after_id

        async with aiohttp.ClientSession() as session:
            async with session.get(
                ANTHROPIC_BASE_URL,
                headers=self._headers(),
                params=params,
            ) as resp:
                body = await resp.json()
                return resp.status, body

    async def cancel_batch(self, batch_id: str) -> tuple[int, dict]:
        """Forward cancel request. Returns (status_code, body)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ANTHROPIC_BASE_URL}/{batch_id}/cancel",
                headers=self._headers(),
            ) as resp:
                body = await resp.json()
                return resp.status, body

    async def stream_results(self, batch_id: str):
        """
        Stream JSONL results from Anthropic.

        First checks batch status; if not ended, returns None.
        Otherwise yields 8KB chunks from the results URL.
        """
        # Check batch status first
        status_code, batch_data = await self.get_batch(batch_id)
        if status_code != 200:
            return None, status_code, batch_data

        if batch_data.get("processing_status") != "ended":
            return None, 404, {
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Batch {batch_id} has not ended yet (status: {batch_data.get('processing_status')})",
                },
            }

        results_url = batch_data.get("results_url")
        if not results_url:
            return None, 404, {
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Batch {batch_id} has no results_url",
                },
            }

        async def _generate():
            timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(results_url, headers=self._headers()) as resp:
                    if resp.status != 200:
                        return
                    async for chunk in resp.content.iter_chunked(8192):
                        yield chunk

        return _generate(), 200, None


# --- Router ---

batch_proxy: Optional[AnthropicBatchProxy] = None

router = APIRouter(tags=["Anthropic Batches"])


def _get_base_url(request: Request) -> str:
    """Derive the proxy's external base URL from the incoming request."""
    return str(request.base_url).rstrip("/")


def _ensure_proxy():
    """Return the proxy instance or a 500 error response."""
    if batch_proxy is None:
        return None, JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "server_error", "message": "Batch proxy not initialized"}},
        )
    if not batch_proxy.api_key:
        return None, JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "server_error", "message": "ANTHROPIC_API_KEY not configured"}},
        )
    return batch_proxy, None


@router.post("/v1/messages/batches")
async def create_message_batch(body: CreateMessageBatchRequest, request: Request):
    """Create a message batch (forwarded to Anthropic)."""
    proxy, err = _ensure_proxy()
    if err:
        return err

    try:
        status, data = await proxy.create_batch(body.model_dump())
    except aiohttp.ClientError as e:
        logger.error(f"Network error creating batch: {e}")
        return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": str(e)}})
    except TimeoutError:
        return JSONResponse(status_code=504, content={"type": "error", "error": {"type": "timeout_error", "message": "Upstream timeout"}})

    if status >= 400:
        return JSONResponse(status_code=status, content=data)

    data = proxy._rewrite_results_url(data, _get_base_url(request))
    return JSONResponse(status_code=status, content=data)


@router.get("/v1/messages/batches/{batch_id}")
async def get_message_batch(batch_id: str, request: Request):
    """Get message batch status (forwarded to Anthropic)."""
    proxy, err = _ensure_proxy()
    if err:
        return err

    try:
        status, data = await proxy.get_batch(batch_id)
    except aiohttp.ClientError as e:
        logger.error(f"Network error getting batch {batch_id}: {e}")
        return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": str(e)}})
    except TimeoutError:
        return JSONResponse(status_code=504, content={"type": "error", "error": {"type": "timeout_error", "message": "Upstream timeout"}})

    if status >= 400:
        return JSONResponse(status_code=status, content=data)

    data = proxy._rewrite_results_url(data, _get_base_url(request))
    return JSONResponse(status_code=status, content=data)


@router.get("/v1/messages/batches")
async def list_message_batches(
    request: Request,
    limit: Optional[int] = Query(None),
    before_id: Optional[str] = Query(None),
    after_id: Optional[str] = Query(None),
):
    """List message batches (forwarded to Anthropic)."""
    proxy, err = _ensure_proxy()
    if err:
        return err

    try:
        status, data = await proxy.list_batches(limit=limit, before_id=before_id, after_id=after_id)
    except aiohttp.ClientError as e:
        logger.error(f"Network error listing batches: {e}")
        return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": str(e)}})
    except TimeoutError:
        return JSONResponse(status_code=504, content={"type": "error", "error": {"type": "timeout_error", "message": "Upstream timeout"}})

    if status >= 400:
        return JSONResponse(status_code=status, content=data)

    # Rewrite results_url in each batch
    base_url = _get_base_url(request)
    if "data" in data:
        for batch in data["data"]:
            proxy._rewrite_results_url(batch, base_url)

    return JSONResponse(status_code=status, content=data)


@router.post("/v1/messages/batches/{batch_id}/cancel")
async def cancel_message_batch(batch_id: str, request: Request):
    """Cancel a message batch (forwarded to Anthropic)."""
    proxy, err = _ensure_proxy()
    if err:
        return err

    try:
        status, data = await proxy.cancel_batch(batch_id)
    except aiohttp.ClientError as e:
        logger.error(f"Network error canceling batch {batch_id}: {e}")
        return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": str(e)}})
    except TimeoutError:
        return JSONResponse(status_code=504, content={"type": "error", "error": {"type": "timeout_error", "message": "Upstream timeout"}})

    if status >= 400:
        return JSONResponse(status_code=status, content=data)

    data = proxy._rewrite_results_url(data, _get_base_url(request))
    return JSONResponse(status_code=status, content=data)


@router.get("/v1/messages/batches/{batch_id}/results")
async def get_message_batch_results(batch_id: str):
    """Stream message batch results as JSONL (forwarded from Anthropic)."""
    proxy, err = _ensure_proxy()
    if err:
        return err

    try:
        generator, status, error_body = await proxy.stream_results(batch_id)
    except aiohttp.ClientError as e:
        logger.error(f"Network error streaming results for {batch_id}: {e}")
        return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": str(e)}})
    except TimeoutError:
        return JSONResponse(status_code=504, content={"type": "error", "error": {"type": "timeout_error", "message": "Upstream timeout"}})

    if generator is None:
        return JSONResponse(status_code=status, content=error_body)

    return StreamingResponse(
        generator,
        media_type="application/x-jsonlines",
        headers={"Transfer-Encoding": "chunked"},
    )
