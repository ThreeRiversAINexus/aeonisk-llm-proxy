"""
Direct executor for sending LLM requests immediately to provider APIs.
"""

import os
import asyncio
import aiohttp
import logging
from typing import Dict, Any

from .models import LLMRequest, LLMProvider

logger = logging.getLogger(__name__)


class DirectExecutor:
    """Executes LLM requests directly via provider APIs."""

    def __init__(self):
        """Initialize direct executor."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    async def execute(self, request: LLMRequest) -> Dict[str, Any]:
        """
        Execute request directly.

        Args:
            request: LLM request

        Returns:
            Response data dict
        """
        if request.provider == LLMProvider.OPENAI:
            return await self._execute_openai(request)
        elif request.provider == LLMProvider.ANTHROPIC:
            return await self._execute_anthropic(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")

    async def _execute_openai(self, request: LLMRequest) -> Dict[str, Any]:
        """Execute OpenAI request."""
        payload = {
            "model": request.model,
            "messages": request.messages,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            # Newer OpenAI models (o1, gpt-5, etc.) require max_completion_tokens
            # instead of max_tokens
            model_lower = request.model.lower()
            if model_lower.startswith(("o1", "gpt-5", "o3")):
                payload["max_completion_tokens"] = request.max_tokens
            else:
                payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"OpenAI API error: {error}")

                data = await resp.json()

                return {
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage"),
                }

    async def _execute_anthropic(self, request: LLMRequest) -> Dict[str, Any]:
        """Execute Anthropic request."""
        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens or 4096,
            "messages": request.messages,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Anthropic API error: {error}")

                data = await resp.json()

                return {
                    "content": data["content"][0]["text"],
                    "usage": data.get("usage"),
                }
