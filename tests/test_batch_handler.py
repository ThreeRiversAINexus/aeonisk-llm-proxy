"""Tests for provider-specific batch request and result normalization."""

import json

import pytest

from aeonisk_llm_proxy.batch_handler import BatchAPIHandler
from aeonisk_llm_proxy.models import BatchSubmission, LLMProvider, LLMRequest


def test_build_gemini_request_maps_messages_and_generation_config():
    request = LLMRequest(
        provider=LLMProvider.GEMINI,
        model="gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": "Follow the game rules."},
            {"role": "user", "content": "Declare an action."},
            {"role": "assistant", "content": "I take cover."},
        ],
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
    )

    payload = BatchAPIHandler._build_gemini_request(request)

    assert payload["systemInstruction"]["parts"] == [
        {"text": "Follow the game rules."}
    ]
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Declare an action."}]},
        {"role": "model", "parts": [{"text": "I take cover."}]},
    ]
    assert payload["generationConfig"] == {
        "temperature": 0.7,
        "maxOutputTokens": 512,
        "topP": 0.9,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("result_container", ["response", "dest"])
async def test_write_gemini_results_uses_metadata_and_order_fallback(
    tmp_path, monkeypatch, result_container
):
    state_file = tmp_path / "state.json"
    handler = BatchAPIHandler(state_file=str(state_file))
    submission = BatchSubmission(
        batch_id="batch-1",
        provider=LLMProvider.GEMINI,
        request_ids=["request-1", "request-2"],
        total_requests=2,
    )
    handler.active_batches[submission.batch_id] = submission

    monkeypatch.setattr(
        "aeonisk_llm_proxy.batch_handler.Path",
        lambda path: tmp_path / path.rsplit("/", 1)[-1],
    )

    await handler._write_gemini_results(
        submission.batch_id,
        {
            result_container: {
                "inlinedResponses": [
                    {
                        "metadata": {"key": "request-1"},
                        "response": {"candidates": []},
                    },
                    {
                        "error": {"message": "blocked"},
                    },
                ],
            },
        },
    )

    with open(submission.output_file_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert lines[0]["custom_id"] == "request-1"
    assert lines[1] == {
        "custom_id": "request-2",
        "error": {"message": "blocked"},
    }
    assert submission.status == "completed"


@pytest.mark.asyncio
async def test_poll_gemini_batch_accepts_batch_state_succeeded(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    handler = BatchAPIHandler(state_file=str(state_file))
    submission = BatchSubmission(
        batch_id="batch-2",
        provider=LLMProvider.GEMINI,
        provider_batch_id="provider-batch",
        request_ids=["request-1"],
        total_requests=1,
        status="submitted",
    )
    handler.active_batches[submission.batch_id] = submission

    async def fake_sleep(_seconds):
        return None

    class FakeResponse:
        status = 200

        async def json(self):
            return {
                "state": "BATCH_STATE_SUCCEEDED",
                "response": {
                    "inlinedResponses": [
                        {
                            "metadata": {"key": "request-1"},
                            "response": {"candidates": []},
                        }
                    ]
                },
            }

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def post(self, *args, **kwargs):
            raise AssertionError("poll test should not submit")

        def get(self, *args, **kwargs):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("aeonisk_llm_proxy.batch_handler.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("aeonisk_llm_proxy.batch_handler.aiohttp.ClientSession", FakeSession)

    await handler._poll_gemini_batch(submission.batch_id, submission.provider_batch_id)

    assert submission.status == "completed"


@pytest.mark.asyncio
async def test_write_gemini_results_parses_string_entries(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    handler = BatchAPIHandler(state_file=str(state_file))
    submission = BatchSubmission(
        batch_id="batch-3",
        provider=LLMProvider.GEMINI,
        request_ids=["request-1"],
        total_requests=1,
    )
    handler.active_batches[submission.batch_id] = submission

    monkeypatch.setattr(
        "aeonisk_llm_proxy.batch_handler.Path",
        lambda path: tmp_path / path.rsplit("/", 1)[-1],
    )

    await handler._write_gemini_results(
        submission.batch_id,
        {
            "response": {
                "inlinedResponses": [
                    "{\"metadata\":{\"key\":\"request-1\"},\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}}"
                ]
            }
        },
    )

    with open(submission.output_file_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert lines[0]["custom_id"] == "request-1"
    assert lines[0]["gemini_response"]["candidates"][0]["content"]["parts"][0]["text"] == "ok"
