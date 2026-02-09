"""
Response tracker for correlating batch responses to original requests.

Handles request/response matching and provides a way for callers to
wait for their responses.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime

from .models import LLMRequest, LLMResponse, RequestStatus

logger = logging.getLogger(__name__)


class ResponseTracker:
    """
    Tracks pending requests and their responses.

    Provides async wait mechanism for callers to get their responses.
    """

    def __init__(self):
        """Initialize response tracker."""
        # Pending requests
        self.pending: Dict[str, LLMRequest] = {}

        # Completed responses
        self.responses: Dict[str, LLMResponse] = {}

        # Events for waiting on responses
        self.events: Dict[str, asyncio.Event] = {}

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def register_request(self, request: LLMRequest):
        """
        Register a new request for tracking.

        Args:
            request: LLM request to track
        """
        async with self.lock:
            self.pending[request.request_id] = request
            self.events[request.request_id] = asyncio.Event()

            logger.debug(f"Registered request {request.request_id}")

    async def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[LLMResponse]:
        """
        Wait for response to a request.

        Args:
            request_id: Request ID to wait for
            timeout: Optional timeout in seconds

        Returns:
            LLMResponse or None if timeout

        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        event = self.events.get(request_id)

        if not event:
            logger.warning(f"Request {request_id} not found in tracker")
            return None

        # Wait for event to be set
        try:
            if timeout:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response {request_id}")
            raise

        # Return response
        async with self.lock:
            return self.responses.get(request_id)

    async def set_response(self, response: LLMResponse):
        """
        Set response for a request.

        Args:
            response: LLM response to set
        """
        async with self.lock:
            request_id = response.request_id

            # Store response
            self.responses[request_id] = response

            # Remove from pending
            self.pending.pop(request_id, None)

            # Signal event
            event = self.events.get(request_id)
            if event:
                event.set()

            logger.debug(
                f"Set response for {request_id} (status: {response.status})"
            )

    async def set_error(
        self,
        request_id: str,
        error: str,
    ):
        """
        Set error for a request.

        Args:
            request_id: Request ID
            error: Error message
        """
        response = LLMResponse(
            request_id=request_id,
            status=RequestStatus.FAILED,
            error=error,
        )

        await self.set_response(response)

    async def set_batch_responses(
        self,
        batch_id: str,
        responses: Dict[str, Dict],
        routed_via: str = "batch",
    ):
        """
        Set responses for multiple requests from a batch.

        Args:
            batch_id: Batch ID
            responses: Dict mapping request_id to response data
            routed_via: How the request was routed
        """
        for request_id, response_data in responses.items():
            # Create response
            response = LLMResponse(
                request_id=request_id,
                status=RequestStatus.COMPLETED,
                content=response_data.get("content"),
                usage=response_data.get("usage"),
                provider=response_data.get("provider"),
                model=response_data.get("model"),
                completed_at=datetime.utcnow(),
                routed_via=routed_via,
                batch_id=batch_id,
            )

            await self.set_response(response)

    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self.pending)

    def get_response(self, request_id: str) -> Optional[LLMResponse]:
        """Get response synchronously (non-blocking)."""
        return self.responses.get(request_id)

    async def cleanup_old_responses(self, max_age_seconds: int = 3600):
        """
        Clean up old completed responses to prevent memory leak.

        Args:
            max_age_seconds: Max age to keep responses
        """
        async with self.lock:
            now = datetime.utcnow()
            to_remove = []

            for request_id, response in self.responses.items():
                if response.completed_at:
                    age = (now - response.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(request_id)

            for request_id in to_remove:
                self.responses.pop(request_id, None)
                self.events.pop(request_id, None)

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old responses")

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            "pending_requests": len(self.pending),
            "completed_responses": len(self.responses),
            "waiting_callers": len(self.events),
        }
