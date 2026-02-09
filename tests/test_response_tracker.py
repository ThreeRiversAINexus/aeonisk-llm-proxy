"""Tests for ResponseTracker: request/response correlation, async wait, cleanup."""

import asyncio
import pytest
from datetime import datetime, timedelta

from aeonisk_llm_proxy.response_tracker import ResponseTracker
from aeonisk_llm_proxy.models import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    RequestStatus,
)


def make_request(request_id: str = "req-1") -> LLMRequest:
    return LLMRequest(
        request_id=request_id,
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
    )


def make_response(
    request_id: str = "req-1",
    status: RequestStatus = RequestStatus.COMPLETED,
    content: str = "Hello!",
) -> LLMResponse:
    return LLMResponse(
        request_id=request_id,
        status=status,
        content=content,
        completed_at=datetime.utcnow(),
    )


class TestRegisterRequest:
    """Test request registration."""

    @pytest.mark.asyncio
    async def test_register_creates_event(self):
        tracker = ResponseTracker()
        req = make_request("req-1")
        await tracker.register_request(req)

        assert "req-1" in tracker.pending
        assert "req-1" in tracker.events
        assert not tracker.events["req-1"].is_set()

    @pytest.mark.asyncio
    async def test_register_multiple(self):
        tracker = ResponseTracker()
        for i in range(3):
            await tracker.register_request(make_request(f"req-{i}"))

        assert tracker.get_pending_count() == 3


class TestSetResponse:
    """Test setting responses."""

    @pytest.mark.asyncio
    async def test_set_response_signals_event(self):
        tracker = ResponseTracker()
        req = make_request("req-1")
        await tracker.register_request(req)

        resp = make_response("req-1")
        await tracker.set_response(resp)

        assert tracker.events["req-1"].is_set()
        assert "req-1" in tracker.responses
        assert "req-1" not in tracker.pending

    @pytest.mark.asyncio
    async def test_set_error(self):
        tracker = ResponseTracker()
        req = make_request("req-err")
        await tracker.register_request(req)

        await tracker.set_error("req-err", "Something broke")

        resp = tracker.get_response("req-err")
        assert resp is not None
        assert resp.status == RequestStatus.FAILED
        assert resp.error == "Something broke"


class TestWaitForResponse:
    """Test async wait mechanism."""

    @pytest.mark.asyncio
    async def test_wait_returns_when_response_set(self):
        tracker = ResponseTracker()
        req = make_request("req-1")
        await tracker.register_request(req)

        # Set response in background
        async def set_later():
            await asyncio.sleep(0.05)
            await tracker.set_response(make_response("req-1", content="Done!"))

        asyncio.create_task(set_later())

        resp = await tracker.wait_for_response("req-1", timeout=2.0)
        assert resp is not None
        assert resp.content == "Done!"

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        tracker = ResponseTracker()
        req = make_request("req-timeout")
        await tracker.register_request(req)

        with pytest.raises(asyncio.TimeoutError):
            await tracker.wait_for_response("req-timeout", timeout=0.05)

    @pytest.mark.asyncio
    async def test_wait_nonexistent_returns_none(self):
        tracker = ResponseTracker()
        result = await tracker.wait_for_response("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_no_timeout_blocks_until_set(self):
        tracker = ResponseTracker()
        req = make_request("req-block")
        await tracker.register_request(req)

        async def set_later():
            await asyncio.sleep(0.05)
            await tracker.set_response(make_response("req-block"))

        asyncio.create_task(set_later())

        # Should block but complete when response arrives
        resp = await asyncio.wait_for(
            tracker.wait_for_response("req-block"),
            timeout=2.0,
        )
        assert resp is not None


class TestSetBatchResponses:
    """Test batch response setting."""

    @pytest.mark.asyncio
    async def test_set_batch_responses(self):
        tracker = ResponseTracker()

        # Register multiple requests
        for i in range(3):
            await tracker.register_request(make_request(f"req-{i}"))

        # Set batch responses
        batch_data = {
            "req-0": {"content": "Response 0", "provider": LLMProvider.OPENAI, "model": "gpt-4o-mini"},
            "req-1": {"content": "Response 1", "provider": LLMProvider.OPENAI, "model": "gpt-4o-mini"},
            "req-2": {"content": "Response 2", "provider": LLMProvider.OPENAI, "model": "gpt-4o-mini"},
        }

        await tracker.set_batch_responses("batch-1", batch_data)

        for i in range(3):
            resp = tracker.get_response(f"req-{i}")
            assert resp is not None
            assert resp.status == RequestStatus.COMPLETED
            assert resp.content == f"Response {i}"
            assert resp.batch_id == "batch-1"
            assert resp.routed_via == "batch"

    @pytest.mark.asyncio
    async def test_batch_responses_signal_events(self):
        tracker = ResponseTracker()

        for i in range(2):
            await tracker.register_request(make_request(f"req-{i}"))

        await tracker.set_batch_responses("batch-1", {
            "req-0": {"content": "A"},
            "req-1": {"content": "B"},
        })

        # All events should be set
        assert tracker.events["req-0"].is_set()
        assert tracker.events["req-1"].is_set()


class TestGetResponse:
    """Test synchronous response retrieval."""

    @pytest.mark.asyncio
    async def test_get_response_exists(self):
        tracker = ResponseTracker()
        req = make_request("req-1")
        await tracker.register_request(req)
        await tracker.set_response(make_response("req-1"))

        resp = tracker.get_response("req-1")
        assert resp is not None
        assert resp.request_id == "req-1"

    def test_get_response_not_found(self):
        tracker = ResponseTracker()
        resp = tracker.get_response("nonexistent")
        assert resp is None


class TestCleanup:
    """Test old response cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_responses(self):
        tracker = ResponseTracker()

        # Register and complete a request
        req = make_request("req-old")
        await tracker.register_request(req)

        old_resp = LLMResponse(
            request_id="req-old",
            status=RequestStatus.COMPLETED,
            content="Old response",
            completed_at=datetime.utcnow() - timedelta(seconds=7200),
        )
        await tracker.set_response(old_resp)

        # Should have response before cleanup
        assert tracker.get_response("req-old") is not None

        await tracker.cleanup_old_responses(max_age_seconds=3600)

        # Old response should be cleaned up
        assert tracker.get_response("req-old") is None
        assert "req-old" not in tracker.events

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_responses(self):
        tracker = ResponseTracker()

        req = make_request("req-recent")
        await tracker.register_request(req)
        await tracker.set_response(make_response("req-recent"))

        await tracker.cleanup_old_responses(max_age_seconds=3600)

        # Recent response should still be there
        assert tracker.get_response("req-recent") is not None


class TestStats:
    """Test tracker statistics."""

    @pytest.mark.asyncio
    async def test_stats(self):
        tracker = ResponseTracker()

        # Register 2, complete 1
        await tracker.register_request(make_request("req-1"))
        await tracker.register_request(make_request("req-2"))
        await tracker.set_response(make_response("req-1"))

        stats = tracker.get_stats()
        assert stats["pending_requests"] == 1
        assert stats["completed_responses"] == 1
        assert stats["waiting_callers"] == 2  # Events for both still exist
