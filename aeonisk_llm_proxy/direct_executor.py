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

# OpenAI-compatible provider configurations: (base_url, env_var_name)
OPENAI_COMPATIBLE_PROVIDERS = {
    LLMProvider.OPENAI: ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    LLMProvider.GROK: ("https://api.x.ai/v1", "XAI_API_KEY"),
    LLMProvider.GEMINI: ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    LLMProvider.DEEPINFRA: ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}


class DirectExecutor:
    """Executes LLM requests directly via provider APIs."""

    def __init__(self):
        """Initialize direct executor."""
        self.api_keys: Dict[str, str] = {}
        for provider, (_, env_var) in OPENAI_COMPATIBLE_PROVIDERS.items():
            self.api_keys[env_var] = os.getenv(env_var, "")
        self.api_keys["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

    def get_api_key(self, provider: LLMProvider) -> str:
        """Get API key for a provider."""
        if provider == LLMProvider.ANTHROPIC:
            return self.api_keys["ANTHROPIC_API_KEY"]
        _, env_var = OPENAI_COMPATIBLE_PROVIDERS[provider]
        return self.api_keys[env_var]

    async def execute(self, request: LLMRequest) -> Dict[str, Any]:
        """
        Execute request directly.

        Args:
            request: LLM request

        Returns:
            Response data dict
        """
        if request.provider == LLMProvider.ANTHROPIC:
            return await self._execute_anthropic(request)
        elif request.provider in OPENAI_COMPATIBLE_PROVIDERS:
            return await self._execute_openai_compatible(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")

    async def _execute_openai_compatible(self, request: LLMRequest) -> Dict[str, Any]:
        """Execute request via an OpenAI-compatible API."""
        base_url, env_var = OPENAI_COMPATIBLE_PROVIDERS[request.provider]
        api_key = self.api_keys[env_var]

        payload = {
            "model": request.model,
            "messages": request.messages,
        }

        if request.temperature is not None:
            # Some OpenAI models (o1, gpt-5, o3) don't support temperature
            if request.provider == LLMProvider.OPENAI:
                model_lower = request.model.lower()
                if not model_lower.startswith(("o1", "gpt-5", "o3")):
                    payload["temperature"] = request.temperature
            else:
                payload["temperature"] = request.temperature

        if request.max_tokens is not None:
            # OpenAI newer models require max_completion_tokens
            if request.provider == LLMProvider.OPENAI:
                model_lower = request.model.lower()
                if model_lower.startswith(("o1", "gpt-5", "o3")):
                    payload["max_completion_tokens"] = request.max_tokens
                else:
                    payload["max_tokens"] = request.max_tokens
            else:
                payload["max_tokens"] = request.max_tokens

        if request.top_p is not None:
            payload["top_p"] = request.top_p

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"{request.provider.value} API error: {error}")

                data = await resp.json()

                return {
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage"),
                }

    async def _execute_anthropic(self, request: LLMRequest) -> Dict[str, Any]:
        """Execute Anthropic request."""
        api_key = self.api_keys["ANTHROPIC_API_KEY"]

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
                    "x-api-key": api_key,
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
