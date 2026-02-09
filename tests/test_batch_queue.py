"""Tests for BatchQueue: enqueue, flush logic, auto-flush triggers."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from aeonisk_llm_proxy.batch_queue import BatchQueue
from aeonisk_llm_proxy.models import (
    LLMRequest,
    LLMProvider,
    RequestPriority,
    RoutingStrategy,
    BatchSubmission,
)


def make_request(
    request_id: str = "req-1",
    provider: LLMProvider = LLMProvider.OPENAI,
    model: str = "gpt-4o-mini",
    priority: RequestPriority = RequestPriority.NORMAL,
    strategy: RoutingStrategy = RoutingStrategy.AUTO,
    created_at: datetime = None,
) -> LLMRequest:
    """Helper to create a request."""
    kwargs = dict(
        request_id=request_id,
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "test"}],
        priority=priority,
        strategy=strategy,
    )
    if created_at:
        kwargs["created_at"] = created_at
    return LLMRequest(**kwargs)


class TestShouldBatch:
    """Test _should_batch decision logic."""

    def test_high_priority_never_batches(self):
        queue = BatchQueue()
        req = make_request(priority=RequestPriority.HIGH)
        assert queue._should_batch(req) is False

    def test_normal_priority_batches(self):
        queue = BatchQueue()
        req = make_request(priority=RequestPriority.NORMAL)
        assert queue._should_batch(req) is True

    def test_low_priority_batches(self):
        queue = BatchQueue()
        req = make_request(priority=RequestPriority.LOW)
        assert queue._should_batch(req) is True

    def test_explicit_direct_strategy_skips_batch(self):
        queue = BatchQueue()
        req = make_request(strategy=RoutingStrategy.DIRECT)
        assert queue._should_batch(req) is False

    def test_explicit_batch_strategy_forces_batch(self):
        queue = BatchQueue()
        req = make_request(strategy=RoutingStrategy.BATCH)
        assert queue._should_batch(req) is True


class TestEnqueue:
    """Test enqueue behavior."""

    @pytest.mark.asyncio
    async def test_enqueue_normal_request(self):
        queue = BatchQueue()
        req = make_request()
        result = await queue.enqueue(req)
        assert result is True
        assert queue.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_enqueue_high_priority_rejected(self):
        queue = BatchQueue()
        req = make_request(priority=RequestPriority.HIGH)
        result = await queue.enqueue(req)
        assert result is False
        assert queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_enqueue_multiple_requests(self):
        queue = BatchQueue()
        for i in range(5):
            await queue.enqueue(make_request(request_id=f"req-{i}"))
        assert queue.get_queue_size() == 5

    @pytest.mark.asyncio
    async def test_enqueue_different_models_separate_queues(self):
        queue = BatchQueue()
        await queue.enqueue(make_request(request_id="req-1", model="gpt-4o-mini"))
        await queue.enqueue(make_request(request_id="req-2", model="gpt-4o"))

        assert queue.get_queue_size() == 2
        assert queue.get_queue_size(LLMProvider.OPENAI) == 2
        # Two separate queue keys
        assert len(queue.queues) == 2

    @pytest.mark.asyncio
    async def test_enqueue_tracks_pending(self):
        queue = BatchQueue()
        req = make_request(request_id="req-tracked")
        await queue.enqueue(req)
        assert "req-tracked" in queue.pending


class TestDequeue:
    """Test dequeue (cancellation) behavior."""

    @pytest.mark.asyncio
    async def test_dequeue_existing_request(self):
        queue = BatchQueue()
        req = make_request(request_id="req-cancel")
        await queue.enqueue(req)

        removed = await queue.dequeue("req-cancel")
        assert removed is not None
        assert removed.request_id == "req-cancel"
        assert queue.get_queue_size() == 0
        assert "req-cancel" not in queue.pending

    @pytest.mark.asyncio
    async def test_dequeue_nonexistent_returns_none(self):
        queue = BatchQueue()
        removed = await queue.dequeue("nonexistent")
        assert removed is None


class TestFlush:
    """Test flush logic."""

    @pytest.mark.asyncio
    async def test_threshold_flush(self):
        """Queue should flush when threshold is reached."""
        callback = AsyncMock()
        queue = BatchQueue(batch_threshold=3, flush_callback=callback)

        # Add 3 requests to trigger flush
        for i in range(3):
            await queue.enqueue(make_request(request_id=f"req-{i}"))

        # Give the created task a chance to run
        await asyncio.sleep(0.1)

        callback.assert_called_once()
        args = callback.call_args
        submission = args[0][0]
        requests = args[0][1]
        assert isinstance(submission, BatchSubmission)
        assert len(requests) == 3

    @pytest.mark.asyncio
    async def test_flush_all(self):
        """flush_all should flush all queues."""
        callback = AsyncMock()
        queue = BatchQueue(flush_callback=callback)

        await queue.enqueue(make_request(request_id="req-1", model="gpt-4o-mini"))
        await queue.enqueue(make_request(request_id="req-2", model="gpt-4o"))

        await queue.flush_all()

        assert callback.call_count == 2
        assert queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_flush_provider(self):
        """flush_provider should only flush queues for that provider."""
        callback = AsyncMock()
        queue = BatchQueue(flush_callback=callback)

        await queue.enqueue(make_request(
            request_id="req-oai",
            provider=LLMProvider.OPENAI,
        ))
        await queue.enqueue(make_request(
            request_id="req-ant",
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929",
        ))

        submissions = await queue.flush_provider(LLMProvider.OPENAI)

        assert len(submissions) == 1
        assert submissions[0].provider == LLMProvider.OPENAI
        # Anthropic request still in queue
        assert queue.get_queue_size(LLMProvider.ANTHROPIC) == 1

    @pytest.mark.asyncio
    async def test_flush_empty_queue_returns_none(self):
        queue = BatchQueue()
        result = await queue._flush_queue((LLMProvider.OPENAI, "gpt-4o-mini"))
        assert result is None


class TestShouldFlushTimeout:
    """Test timeout-based flush triggers."""

    def test_max_wait_exceeded(self):
        queue = BatchQueue(max_wait_seconds=10)
        key = (LLMProvider.OPENAI, "gpt-4o-mini")

        # Add an old request
        old_req = make_request(
            created_at=datetime.utcnow() - timedelta(seconds=15),
        )
        queue.queues[key].append(old_req)

        assert queue._should_flush_timeout(key) is True

    def test_max_wait_not_exceeded(self):
        queue = BatchQueue(max_wait_seconds=300)
        key = (LLMProvider.OPENAI, "gpt-4o-mini")

        req = make_request(created_at=datetime.utcnow())
        queue.queues[key].append(req)

        assert queue._should_flush_timeout(key) is False

    def test_idle_timeout(self):
        queue = BatchQueue(max_idle_seconds=5)
        key = (LLMProvider.OPENAI, "gpt-4o-mini")

        req = make_request(created_at=datetime.utcnow())
        queue.queues[key].append(req)
        # Simulate old last request time
        queue.last_request_time[key] = datetime.utcnow() - timedelta(seconds=10)

        assert queue._should_flush_timeout(key) is True

    def test_empty_queue_no_flush(self):
        queue = BatchQueue()
        key = (LLMProvider.OPENAI, "gpt-4o-mini")
        assert queue._should_flush_timeout(key) is False


class TestQueueStats:
    """Test queue statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        queue = BatchQueue()
        await queue.enqueue(make_request(request_id="req-1"))
        await queue.enqueue(make_request(request_id="req-2"))

        stats = queue.get_stats()
        assert stats["total_queued"] == 2
        assert "openai:gpt-4o-mini" in stats["queues"]
        assert stats["queues"]["openai:gpt-4o-mini"] == 2

    @pytest.mark.asyncio
    async def test_get_queue_size_by_provider(self):
        queue = BatchQueue()
        await queue.enqueue(make_request(request_id="req-1", provider=LLMProvider.OPENAI))
        await queue.enqueue(make_request(
            request_id="req-2",
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929",
        ))

        assert queue.get_queue_size(LLMProvider.OPENAI) == 1
        assert queue.get_queue_size(LLMProvider.ANTHROPIC) == 1
        assert queue.get_queue_size() == 2


class TestStartStop:
    """Test queue lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        queue = BatchQueue()
        await queue.start()
        assert queue.is_running is True
        assert queue.auto_flush_task is not None

        await queue.stop()
        assert queue.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        queue = BatchQueue()
        await queue.start()
        await queue.start()  # Should not create second task
        assert queue.is_running is True
        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self):
        callback = AsyncMock()
        queue = BatchQueue(flush_callback=callback)
        await queue.start()

        await queue.enqueue(make_request(request_id="req-1"))
        await queue.stop()

        # Should have flushed remaining request
        callback.assert_called_once()
