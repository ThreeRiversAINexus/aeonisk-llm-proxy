"""Tests for proxy server batch result processing."""

import json

import pytest

from aeonisk_llm_proxy.models import BatchSubmission, LLMProvider, LLMRequest
from aeonisk_llm_proxy.proxy_server import LLMProxyServer


@pytest.mark.asyncio
async def test_process_batch_results_accepts_normalized_gemini_output(tmp_path):
    proxy = LLMProxyServer()

    output_file = tmp_path / "gemini_batch_results.jsonl"
    output_file.write_text(
        json.dumps(
            {
                "custom_id": "request-1",
                "content": "hello from batch",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
                "provider": "gemini",
                "gemini_response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "hello from batch"}]
                            }
                        }
                    ]
                },
            }
        )
        + "\n"
    )

    submission = BatchSubmission(
        batch_id="batch-1",
        provider=LLMProvider.GEMINI,
        output_file_path=str(output_file),
    )
    requests = [
        LLMRequest(
            request_id="request-1",
            provider=LLMProvider.GEMINI,
            model="gemini-3.1-flash-lite",
            messages=[{"role": "user", "content": "Say hello"}],
        )
    ]

    captured = {}

    async def fake_set_batch_responses(batch_id, responses, routed_via="batch"):
        captured["batch_id"] = batch_id
        captured["responses"] = responses
        captured["routed_via"] = routed_via

    proxy.response_tracker.set_batch_responses = fake_set_batch_responses

    await proxy._process_batch_results("batch-1", submission, requests)

    assert captured["batch_id"] == "batch-1"
    assert captured["routed_via"] == "batch"
    assert captured["responses"]["request-1"]["content"] == "hello from batch"
    assert captured["responses"]["request-1"]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
