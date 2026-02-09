"""
Batch queue for collecting and flushing LLM requests.

Queues requests until threshold or timeout, then flushes to batch API.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from collections import defaultdict

from .models import (
    LLMRequest,
    LLMProvider,
    RequestPriority,
    BatchSubmission,
)

logger = logging.getLogger(__name__)


class BatchQueue:
    """
    Manages queuing and flushing of LLM requests for batch processing.

    Requests are queued by (provider, model) and flushed when:
    1. Queue reaches batch_threshold size
    2. Oldest request exceeds max_wait_seconds
    3. No new requests added for max_idle_seconds (for overnight batches)
    4. Manual flush is triggered
    """

    # Type alias for queue key: (provider, model)
    QueueKey = tuple  # (LLMProvider, str)

    def __init__(
        self,
        batch_threshold: int = 100,
        max_wait_seconds: int = 300,
        max_idle_seconds: int = 3600,
        flush_callback: Optional[Callable[[BatchSubmission, List[LLMRequest]], Awaitable[None]]] = None,
    ):
        """
        Initialize batch queue.

        Args:
            batch_threshold: Flush when queue reaches this size
            max_wait_seconds: Flush after this many seconds
            max_idle_seconds: Flush if no new requests for this long
            flush_callback: Async function called when batch is flushed
        """
        self.batch_threshold = batch_threshold
        self.max_wait_seconds = max_wait_seconds
        self.max_idle_seconds = max_idle_seconds
        self.flush_callback = flush_callback

        # Queues by (provider, model) - OpenAI Batch API requires same model per batch
        self.queues: Dict[tuple, List[LLMRequest]] = defaultdict(list)

        # Pending requests lookup by ID
        self.pending: Dict[str, LLMRequest] = {}

        # Track last request time per queue key for idle detection
        self.last_request_time: Dict[tuple, datetime] = {}

        # Lock for thread safety
        self.lock = asyncio.Lock()

        # Auto-flush task
        self.auto_flush_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def start(self):
        """Start auto-flush background task."""
        if self.is_running:
            return

        self.is_running = True
        self.auto_flush_task = asyncio.create_task(self._auto_flush_loop())
        logger.info("Batch queue started")

    async def stop(self):
        """Stop auto-flush and flush remaining requests."""
        if not self.is_running:
            return

        self.is_running = False

        if self.auto_flush_task:
            self.auto_flush_task.cancel()
            try:
                await self.auto_flush_task
            except asyncio.CancelledError:
                pass

        # Flush all remaining requests
        await self.flush_all()
        logger.info("Batch queue stopped")

    def _queue_key(self, request: LLMRequest) -> tuple:
        """Get queue key for a request: (provider, model)."""
        return (request.provider, request.model)

    async def enqueue(self, request: LLMRequest) -> bool:
        """
        Add request to queue.

        Args:
            request: LLM request to queue

        Returns:
            True if queued, False if should be sent directly
        """
        async with self.lock:
            # Check if we should batch this request
            if not self._should_batch(request):
                return False

            # Add to queue by (provider, model)
            key = self._queue_key(request)
            self.queues[key].append(request)
            self.pending[request.request_id] = request

            # Update last request time for idle detection
            self.last_request_time[key] = datetime.utcnow()

            logger.debug(
                f"Queued request {request.request_id} for {request.provider}:{request.model} "
                f"(queue size: {len(self.queues[key])})"
            )

            # Check if we should flush immediately
            if self._should_flush(key):
                asyncio.create_task(self._flush_queue(key))

            return True

    async def dequeue(self, request_id: str) -> Optional[LLMRequest]:
        """
        Remove request from queue (e.g., for cancellation).

        Args:
            request_id: Request ID to remove

        Returns:
            Removed request or None
        """
        async with self.lock:
            request = self.pending.pop(request_id, None)

            if request:
                # Remove from queue
                key = self._queue_key(request)
                self.queues[key] = [
                    r for r in self.queues[key] if r.request_id != request_id
                ]
                logger.debug(f"Dequeued request {request_id}")

            return request

    async def _flush_queue(self, key: tuple) -> Optional[BatchSubmission]:
        """
        Flush all requests for a (provider, model) queue.

        Args:
            key: Queue key (provider, model)

        Returns:
            BatchSubmission if flushed, None if queue empty
        """
        async with self.lock:
            queue = self.queues[key]
            provider, model = key

            if not queue:
                return None

            # Create batch submission
            submission = BatchSubmission(
                provider=provider,
                request_ids=[r.request_id for r in queue],
                total_requests=len(queue),
                status="pending",
            )

            # Remove from queue
            requests = queue.copy()
            self.queues[key] = []

            # Remove from pending
            for request in requests:
                self.pending.pop(request.request_id, None)

            logger.info(
                f"Flushing {len(requests)} requests for {provider}:{model} "
                f"(batch_id: {submission.batch_id})"
            )

        # Call flush callback (outside lock)
        if self.flush_callback:
            try:
                await self.flush_callback(submission, requests)
            except Exception as e:
                logger.error(f"Flush callback error: {e}", exc_info=True)

        return submission

    async def flush_provider(self, provider: LLMProvider) -> List[BatchSubmission]:
        """
        Flush all queues for a provider (all models).

        Args:
            provider: Provider to flush

        Returns:
            List of BatchSubmissions
        """
        submissions = []
        keys_to_flush = [k for k in self.queues.keys() if k[0] == provider]
        for key in keys_to_flush:
            submission = await self._flush_queue(key)
            if submission:
                submissions.append(submission)
        return submissions

    async def flush_all(self):
        """Flush all queues."""
        for key in list(self.queues.keys()):
            await self._flush_queue(key)

    def get_queue_size(self, provider: Optional[LLMProvider] = None) -> int:
        """Get queue size for provider or total."""
        if provider:
            return sum(len(q) for k, q in self.queues.items() if k[0] == provider)
        return sum(len(q) for q in self.queues.values())

    def _get_oldest_request_age(self, key: tuple) -> Optional[float]:
        """Get age in seconds of oldest request in a queue."""
        queue = self.queues[key]

        if not queue:
            return None

        oldest = min(queue, key=lambda r: r.created_at)
        return (datetime.utcnow() - oldest.created_at).total_seconds()

    def get_oldest_request_age(self, provider: LLMProvider) -> Optional[float]:
        """Get age in seconds of oldest request across all queues for provider."""
        ages = []
        for key in self.queues.keys():
            if key[0] == provider:
                age = self._get_oldest_request_age(key)
                if age is not None:
                    ages.append(age)
        return max(ages) if ages else None

    async def _auto_flush_loop(self):
        """Background task to auto-flush queues on timeout."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                for key in list(self.queues.keys()):
                    if self._should_flush_timeout(key):
                        provider, model = key
                        logger.info(
                            f"Auto-flushing {provider}:{model} queue (timeout)"
                        )
                        await self._flush_queue(key)

            except Exception as e:
                logger.error(f"Auto-flush loop error: {e}", exc_info=True)

    def _should_batch(self, request: LLMRequest) -> bool:
        """Check if request should be batched."""
        # High priority never batches
        if request.priority == RequestPriority.HIGH:
            return False

        # Explicit strategy override
        if request.strategy.value == "direct":
            return False

        if request.strategy.value == "batch":
            return True

        # Auto strategy: batch normal and low priority
        return request.priority in [RequestPriority.NORMAL, RequestPriority.LOW]

    def _should_flush(self, key: tuple) -> bool:
        """Check if queue should be flushed immediately."""
        queue_size = len(self.queues[key])
        return queue_size >= self.batch_threshold

    def _should_flush_timeout(self, key: tuple) -> bool:
        """Check if queue should be flushed due to timeout or idle time."""
        if not self.queues[key]:
            return False

        queue_size = len(self.queues[key])
        provider, model = key

        # Check if oldest request exceeded max wait time
        age = self._get_oldest_request_age(key)
        if age is not None and age >= self.max_wait_seconds:
            logger.info(
                f"Flushing {provider}:{model} queue: max wait time exceeded "
                f"(age: {age:.0f}s, threshold: {self.max_wait_seconds}s, queue size: {queue_size})"
            )
            return True

        # Check if queue has been idle (no new requests) for too long
        if key in self.last_request_time:
            idle_seconds = (datetime.utcnow() - self.last_request_time[key]).total_seconds()
            if idle_seconds >= self.max_idle_seconds:
                logger.info(
                    f"Flushing {provider}:{model} queue: idle for {idle_seconds:.0f}s "
                    f"(threshold: {self.max_idle_seconds}s, queue size: {queue_size})"
                )
                return True

        return False

    def get_stats(self) -> Dict:
        """Get queue statistics."""
        return {
            "total_queued": self.get_queue_size(),
            "queues": {
                f"{provider.value}:{model}": len(queue)
                for (provider, model), queue in self.queues.items()
            },
            "oldest_ages": {
                f"{provider.value}:{model}": self._get_oldest_request_age((provider, model))
                for (provider, model) in self.queues.keys()
            },
        }
