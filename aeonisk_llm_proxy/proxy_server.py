"""
Main LLM batching proxy server.

Coordinates queue, router, batch handler, and response tracker.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict

from .models import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    RoutingStrategy,
    RequestStatus,
    ProxyConfig,
    ProxyStats,
)
from .batch_queue import BatchQueue
from .router import RequestRouter
from .batch_handler import BatchAPIHandler
from .response_tracker import ResponseTracker
from .direct_executor import DirectExecutor

logger = logging.getLogger(__name__)


class LLMProxyServer:
    """
    Main LLM batching proxy server.

    Receives LLM requests, routes them to batch or direct processing,
    and returns responses to callers.
    """

    def __init__(self, config: Optional[ProxyConfig] = None):
        """
        Initialize proxy server.

        Args:
            config: Proxy configuration
        """
        self.config = config or ProxyConfig()

        # Core components
        self.queue = BatchQueue(
            batch_threshold=self.config.batch_threshold,
            max_wait_seconds=self.config.max_wait_seconds,
            max_idle_seconds=self.config.max_idle_seconds,
            flush_callback=self._on_batch_flush,
        )
        self.router = RequestRouter(self.config)
        self.batch_handler = BatchAPIHandler(
            poll_interval_seconds=self.config.poll_interval_seconds
        )
        self.response_tracker = ResponseTracker()
        self.direct_executor = DirectExecutor()

        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None

        # Statistics
        self.stats = ProxyStats()

    async def start(self):
        """Start proxy server."""
        if self.is_running:
            return

        logger.info("Starting LLM proxy server...")
        self.is_running = True
        self.start_time = datetime.utcnow()

        # Start queue
        await self.queue.start()

        # Start batch handler and resume in-progress batches
        await self.batch_handler.start()

        logger.info("LLM proxy server started")

    async def stop(self):
        """Stop proxy server."""
        if not self.is_running:
            return

        logger.info("Stopping LLM proxy server...")
        self.is_running = False

        # Stop components
        await self.queue.stop()
        await self.batch_handler.stop()

        logger.info("LLM proxy server stopped")

    async def submit_request(
        self,
        request: LLMRequest,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """
        Submit LLM request and wait for response.

        This is the main entry point for clients.

        Args:
            request: LLM request
            timeout: Optional timeout in seconds

        Returns:
            LLM response
        """
        # Register request for tracking
        await self.response_tracker.register_request(request)

        # Update stats
        self.stats.total_requests += 1

        try:
            # Route request
            queue_size = self.queue.get_queue_size(request.provider)
            queue_age = self.queue.get_oldest_request_age(request.provider) or 0

            strategy = self.router.route(request, queue_size, queue_age)

            if strategy == RoutingStrategy.DIRECT:
                # Send directly
                await self._execute_direct(request)
                self.stats.routed_direct += 1

            else:  # BATCH
                # Add to queue
                queued = await self.queue.enqueue(request)

                if not queued:
                    # Queue rejected, send direct
                    await self._execute_direct(request)
                    self.stats.routed_direct += 1
                else:
                    self.stats.routed_batch += 1
                    self.stats.queued_requests += 1

            # Wait for response
            response = await self.response_tracker.wait_for_response(
                request.request_id,
                timeout=timeout,
            )

            # Update stats
            if response:
                if response.status == RequestStatus.COMPLETED:
                    self.stats.completed_requests += 1
                elif response.status == RequestStatus.FAILED:
                    self.stats.failed_requests += 1

            return response

        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}", exc_info=True)

            # Set error response
            await self.response_tracker.set_error(request.request_id, str(e))

            # Try to get the error response
            return await self.response_tracker.wait_for_response(request.request_id, timeout=1.0)

    async def _execute_direct(self, request: LLMRequest):
        """Execute request directly via API."""
        try:
            result = await self.direct_executor.execute(request)

            # Create response
            response = LLMResponse(
                request_id=request.request_id,
                status=RequestStatus.COMPLETED,
                content=result.get("content"),
                usage=result.get("usage"),
                provider=request.provider,
                model=request.model,
                completed_at=datetime.utcnow(),
                routed_via="direct",
            )

            await self.response_tracker.set_response(response)

        except Exception as e:
            logger.error(f"Direct execution failed: {e}", exc_info=True)
            await self.response_tracker.set_error(request.request_id, str(e))

    async def _on_batch_flush(self, submission, requests):
        """
        Callback when batch queue is flushed.

        Args:
            submission: BatchSubmission
            requests: List of LLMRequest
        """
        try:
            logger.info(
                f"Flushing batch {submission.batch_id} with {len(requests)} requests"
            )

            # Submit to batch API
            provider_batch_id = await self.batch_handler.submit_batch(
                submission,
                requests,
            )

            # Update stats
            self.stats.active_batches += 1
            self.stats.queued_requests -= len(requests)
            self.stats.processing_requests += len(requests)

            # Wait for completion in background
            asyncio.create_task(
                self._wait_for_batch_completion(submission.batch_id, requests)
            )

        except Exception as e:
            logger.error(f"Batch flush error: {e}", exc_info=True)

            # Set errors for all requests
            for request in requests:
                await self.response_tracker.set_error(
                    request.request_id,
                    f"Batch submission failed: {str(e)}",
                )

    async def _wait_for_batch_completion(self, batch_id: str, requests):
        """
        Wait for batch to complete and process results.

        Args:
            batch_id: Batch ID
            requests: List of requests in batch
        """
        # Wait for batch handler to complete (it polls automatically)
        # We check periodically if the batch is done
        submission = None
        completed_status = None

        while True:
            # Check status FIRST, then sleep
            submission = self.batch_handler.active_batches.get(batch_id)

            if not submission:
                # Batch completed and was removed from active_batches
                # We need to check if we already processed it
                if completed_status:
                    # Already processed, exit
                    logger.info(f"Batch {batch_id} fully processed and cleaned up")
                    break
                else:
                    # Batch was removed without us seeing completion status
                    # This shouldn't happen, but handle it gracefully
                    logger.warning(f"Batch {batch_id} removed from active_batches before status check")
                    break

            if submission.status in ["completed", "failed", "expired", "cancelled", "ended"]:
                # Store status and output path before it gets removed
                completed_status = submission.status
                output_file_path = submission.output_file_path
                provider = submission.provider

                logger.info(f"Batch {batch_id} finished with status: {completed_status}")

                # Process results
                if completed_status in ["completed", "ended"] and output_file_path:
                    await self._process_batch_results(batch_id, submission, requests)
                else:
                    # Set errors - provide more helpful error message
                    if completed_status in ["completed", "ended"] and not output_file_path:
                        # Batch completed but output file missing - this is a proxy bug
                        error_msg = (
                            f"Batch {batch_id} completed (status={completed_status}) but "
                            f"output_file_path is missing. This is a proxy-side bug. "
                            f"Check if batch was created correctly and output was written."
                        )
                        logger.error(error_msg)
                    else:
                        error_msg = f"Batch ended with status: {completed_status}"

                    for request in requests:
                        await self.response_tracker.set_error(
                            request.request_id,
                            error_msg,
                        )

                # Update stats
                self.stats.active_batches -= 1
                self.stats.completed_batches += 1
                self.stats.processing_requests -= len(requests)

                # Remove from active_batches now that we're done processing
                self.batch_handler.active_batches.pop(batch_id, None)
                logger.info(f"Batch {batch_id} fully processed and cleaned up")
                break

            # Sleep before next check
            await asyncio.sleep(10)

    async def _process_batch_results(self, batch_id, submission, requests):
        """Process batch results and set responses."""
        import json
        from pathlib import Path

        output_file = Path(submission.output_file_path)

        if not output_file.exists():
            logger.error(f"Output file not found: {output_file}")
            # Set errors for all requests
            for request in requests:
                await self.response_tracker.set_error(
                    request.request_id,
                    "Batch output file not found",
                )
            return

        try:
            # Parse results
            results = {}
            truncated_lines = []
            line_num = 0

            with open(output_file) as f:
                for line in f:
                    line_num += 1
                    if not line.strip():
                        continue

                    custom_id = None  # Reset for each line
                    try:
                        result = json.loads(line)
                        custom_id = result.get("custom_id")

                        if not custom_id:
                            continue

                        # Check for error in result
                        if result.get("error"):
                            error_msg = result["error"].get("message", "Unknown error")
                            logger.error(f"Request {custom_id} failed: {error_msg}")
                            await self.response_tracker.set_error(custom_id, error_msg)
                            continue

                        # Extract response data
                        if submission.provider.value == "openai":
                            response_data = result.get("response", {}).get("body", {})
                            content = response_data.get("choices", [{}])[0].get("message", {}).get("content")
                            raw_usage = response_data.get("usage", {})

                            # Normalize usage - flatten nested details
                            usage = {
                                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                                "completion_tokens": raw_usage.get("completion_tokens", 0),
                                "total_tokens": raw_usage.get("total_tokens", 0),
                            }

                        else:  # anthropic
                            response_data = result.get("result", {})
                            content = response_data.get("content", [{}])[0].get("text")
                            raw_usage = response_data.get("usage", {})

                            # Normalize usage
                            usage = {
                                "input_tokens": raw_usage.get("input_tokens", 0),
                                "output_tokens": raw_usage.get("output_tokens", 0),
                            }

                        results[custom_id] = {
                            "content": content,
                            "usage": usage,
                            "provider": submission.provider,
                            "model": requests[0].model if requests else None,
                        }

                    except json.JSONDecodeError as e:
                        # Truncated/malformed JSON line - likely from incomplete download
                        truncated_lines.append((line_num, line[:100]))
                        logger.error(f"Batch {batch_id} line {line_num}: Truncated/invalid JSON: {line[:100]}...")

                    except Exception as e:
                        logger.error(f"Error processing result line {line_num}: {e}", exc_info=True)
                        if custom_id:
                            await self.response_tracker.set_error(custom_id, f"Result parsing error: {str(e)}")

            # Log summary of truncation issues
            if truncated_lines:
                logger.error(
                    f"Batch {batch_id}: {len(truncated_lines)} truncated/invalid lines detected. "
                    f"This may indicate incomplete download or network issues."
                )

            # Set responses
            if results:
                await self.response_tracker.set_batch_responses(
                    batch_id,
                    results,
                    routed_via="batch",
                )

            logger.info(f"Processed {len(results)} results from batch {batch_id}")

        except Exception as e:
            logger.error(f"Error processing batch results: {e}", exc_info=True)
            # Set errors for all requests that weren't processed
            for request in requests:
                if not self.response_tracker.get_response(request.request_id):
                    await self.response_tracker.set_error(
                        request.request_id,
                        f"Batch processing error: {str(e)}",
                    )

    def get_stats(self) -> ProxyStats:
        """Get proxy statistics."""
        # Update current stats
        self.stats.queued_requests = self.queue.get_queue_size()
        self.stats.active_batches = len(self.batch_handler.active_batches)

        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            self.stats.avg_response_time_seconds = (
                uptime / self.stats.completed_requests
                if self.stats.completed_requests > 0
                else 0
            )

        return self.stats

    def get_health(self) -> Dict:
        """Get health check info."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "status": "healthy" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "queue_size": self.queue.get_queue_size(),
            "active_batches": len(self.batch_handler.active_batches),
            "providers_healthy": {
                provider.value: bool(self.direct_executor.get_api_key(provider))
                for provider in LLMProvider
            },
        }
