"""
FastAPI server for LLM batching proxy.

Provides HTTP endpoints for submitting requests and getting responses.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .proxy_server import LLMProxyServer
from .models import LLMRequest, LLMResponse, ProxyConfig, ProxyStats, HealthCheck

logger = logging.getLogger(__name__)

# Global proxy instance
proxy: Optional[LLMProxyServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global proxy

    # Startup
    logger.info("Starting LLM proxy server...")

    # Load config from environment variables
    config = ProxyConfig.from_env()
    logger.info(
        f"Proxy config: threshold={config.batch_threshold}, "
        f"timeout={config.max_wait_seconds}s, "
        f"max_idle={config.max_idle_seconds}s"
    )

    proxy = LLMProxyServer(config=config)
    await proxy.start()

    logger.info("LLM proxy server started")

    yield

    # Shutdown
    logger.info("Shutting down LLM proxy server...")

    if proxy:
        await proxy.stop()

    logger.info("LLM proxy server stopped")


# Create FastAPI app
app = FastAPI(
    title="LLM Batching Proxy",
    description="Smart proxy for batching LLM requests with cost optimization",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Batching Proxy",
        "version": "1.0.0",
        "status": "running" if proxy and proxy.is_running else "stopped",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    health = proxy.get_health()

    return HealthCheck(
        status=health["status"],
        uptime_seconds=health["uptime_seconds"],
        queue_size=health["queue_size"],
        active_batches=health["active_batches"],
        providers_healthy=health["providers_healthy"],
    )


@app.post("/v1/chat/completions", response_model=LLMResponse)
async def chat_completions(request: LLMRequest, timeout: Optional[float] = None):
    """
    Submit LLM request (OpenAI-compatible endpoint).

    Args:
        request: LLM request
        timeout: Timeout in seconds (default: None - no timeout for batch requests)

    Returns:
        LLM response
    """
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    try:
        response = await proxy.submit_request(request, timeout=timeout)

        if not response:
            raise HTTPException(status_code=500, detail="No response received")

        return response

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit", response_model=LLMResponse)
async def submit_request(request: LLMRequest, timeout: Optional[float] = None):
    """
    Submit LLM request (simplified endpoint).

    Args:
        request: LLM request
        timeout: Timeout in seconds (default: None - no timeout)

    Returns:
        LLM response
    """
    return await chat_completions(request, timeout)


@app.get("/stats", response_model=ProxyStats)
async def get_stats():
    """Get proxy statistics."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    return proxy.get_stats()


@app.get("/queue/stats")
async def get_queue_stats():
    """Get queue statistics."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    return proxy.queue.get_stats()


@app.get("/batches/stats")
async def get_batch_stats():
    """Get batch handler statistics."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    return proxy.batch_handler.get_stats()


@app.post("/queue/flush/{provider}")
async def flush_queue(provider: str):
    """Manually flush queue for a provider."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    from .models import LLMProvider

    try:
        provider_enum = LLMProvider(provider)
        submission = await proxy.queue.flush_provider(provider_enum)

        if submission:
            return {"message": f"Flushed queue for {provider}", "batch_id": submission.batch_id}
        else:
            return {"message": f"No requests in queue for {provider}"}

    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")


@app.get("/batches/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of a specific batch."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    # Check active batches
    batch = proxy.batch_handler.active_batches.get(batch_id)

    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    return {
        "batch_id": batch.batch_id,
        "provider": batch.provider.value,
        "provider_batch_id": batch.provider_batch_id,
        "status": batch.status,
        "total_requests": batch.total_requests,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "submitted_at": batch.submitted_at.isoformat() if batch.submitted_at else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
        "output_file_path": batch.output_file_path,
        "age_seconds": (
            (datetime.utcnow() - batch.created_at).total_seconds()
            if batch.created_at
            else 0
        ),
    }


@app.get("/requests/{request_id}")
async def get_request_status(request_id: str):
    """Get status of a specific request."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    # Check if response exists
    response = proxy.response_tracker.responses.get(request_id)

    if response:
        return {
            "request_id": request_id,
            "status": "completed",
            "response": response.dict(),
        }

    # Check if still pending
    event = proxy.response_tracker.events.get(request_id)

    if event:
        return {
            "request_id": request_id,
            "status": "pending",
            "message": "Request is still processing",
        }

    # Not found
    raise HTTPException(status_code=404, detail=f"Request {request_id} not found")


@app.get("/batches")
async def list_batches():
    """List all active batches."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    return proxy.batch_handler.get_stats()


@app.delete("/batches/{batch_id}")
async def delete_batch(batch_id: str, delete_files: bool = True):
    """Delete a specific batch by ID."""
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    if batch_id not in proxy.batch_handler.active_batches:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    result = proxy.batch_handler.purge_batches(
        batch_ids=[batch_id],
        delete_files=delete_files,
    )

    return result


@app.post("/batches/purge")
async def purge_batches(
    older_than_seconds: Optional[int] = None,
    status: Optional[str] = None,
    delete_files: bool = True,
):
    """
    Purge completed batches.

    Args:
        older_than_seconds: Purge batches older than this many seconds.
        status: Only purge batches with this status (comma-separated for multiple).
                Valid: completed, failed, expired, cancelled, ended
        delete_files: Also delete temp files (default: true).

    Returns:
        Count of purged batches and their IDs.
    """
    if not proxy:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    status_filter = None
    if status:
        status_filter = [s.strip() for s in status.split(",")]

    result = proxy.batch_handler.purge_batches(
        older_than_seconds=older_than_seconds,
        status_filter=status_filter,
        delete_files=delete_files,
    )

    return result


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
