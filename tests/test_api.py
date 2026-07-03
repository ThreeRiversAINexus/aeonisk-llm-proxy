"""Tests for the FastAPI surface."""

import pytest

from aeonisk_llm_proxy.api import flush_queue
from aeonisk_llm_proxy.models import BatchSubmission, LLMProvider


class _FakeQueue:
    def __init__(self, submissions):
        self._submissions = submissions

    async def flush_provider(self, provider):
        assert provider == LLMProvider.GEMINI
        return self._submissions


class _FakeProxy:
    def __init__(self, submissions):
        self.queue = _FakeQueue(submissions)


@pytest.mark.asyncio
async def test_flush_queue_returns_batch_ids(monkeypatch):
    submissions = [
        BatchSubmission(batch_id="batch-a", provider=LLMProvider.GEMINI),
        BatchSubmission(batch_id="batch-b", provider=LLMProvider.GEMINI),
    ]
    monkeypatch.setattr("aeonisk_llm_proxy.api.proxy", _FakeProxy(submissions))

    response = await flush_queue("gemini")

    assert response == {
        "message": "Flushed queue for gemini",
        "batch_ids": ["batch-a", "batch-b"],
        "count": 2,
    }
