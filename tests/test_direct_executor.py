"""Tests for DirectExecutor: direct API execution of LLM requests."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import ClientSession

from aeonisk_llm_proxy.direct_executor import DirectExecutor
from aeonisk_llm_proxy.models import LLMRequest, LLMProvider


def make_anthropic_request(**kwargs):
    """Create a sample Anthropic request with system message."""
    defaults = {
        "request_id": "test-req-anthropic",
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-sonnet-4-5-20250929",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
    }
    defaults.update(kwargs)
    return LLMRequest(**defaults)


def make_mock_session(captured_payload: dict, response_data: dict):
    """Create a mock aiohttp session that captures the JSON payload sent to post()."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    def capture_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return mock_response

    mock_session.post = MagicMock(side_effect=capture_post)
    return mock_session


class TestAnthropicSystemMessageExtraction:
    """Test that Anthropic direct execution extracts system messages from the messages array."""

    @pytest.mark.asyncio
    async def test_system_message_extracted_to_top_level(self):
        """System message should be extracted from messages and passed as top-level 'system' param."""
        executor = DirectExecutor()
        request = make_anthropic_request()

        captured_payload = {}
        mock_session = make_mock_session(captured_payload, {
            "content": [{"text": "Hi there!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor._execute_anthropic(request)

        # System should be a top-level param
        assert "system" in captured_payload, "System message not extracted to top-level param"
        assert captured_payload["system"] == "You are a helpful assistant."

        # Messages should NOT contain system role
        for msg in captured_payload["messages"]:
            assert msg["role"] != "system", f"System message still in messages array: {msg}"

        # Should only have the user message
        assert len(captured_payload["messages"]) == 1
        assert captured_payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_message_omits_param(self):
        """When no system message exists, 'system' param should not be in payload."""
        executor = DirectExecutor()
        request = make_anthropic_request(
            messages=[{"role": "user", "content": "Hello"}],
        )

        captured_payload = {}
        mock_session = make_mock_session(captured_payload, {
            "content": [{"text": "Hi!"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        })

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor._execute_anthropic(request)

        assert "system" not in captured_payload, "System param present when no system message sent"

    @pytest.mark.asyncio
    async def test_multiple_messages_preserves_order(self):
        """Non-system messages should preserve their original order."""
        executor = DirectExecutor()
        request = make_anthropic_request(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        )

        captured_payload = {}
        mock_session = make_mock_session(captured_payload, {
            "content": [{"text": "Second answer"}],
            "usage": {"input_tokens": 20, "output_tokens": 5},
        })

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor._execute_anthropic(request)

        assert captured_payload["system"] == "Be brief."
        assert len(captured_payload["messages"]) == 3
        assert captured_payload["messages"][0] == {"role": "user", "content": "First question"}
        assert captured_payload["messages"][1] == {"role": "assistant", "content": "First answer"}
        assert captured_payload["messages"][2] == {"role": "user", "content": "Second question"}

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_passed(self):
        """Temperature and max_tokens should still be passed correctly with system extraction."""
        executor = DirectExecutor()
        request = make_anthropic_request(
            temperature=0.7,
            max_tokens=2048,
        )

        captured_payload = {}
        mock_session = make_mock_session(captured_payload, {
            "content": [{"text": "Response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await executor._execute_anthropic(request)

        assert captured_payload["temperature"] == 0.7
        assert captured_payload["max_tokens"] == 2048
        # System extraction should still work alongside other params
        assert "system" in captured_payload
        assert captured_payload["system"] == "You are a helpful assistant."
