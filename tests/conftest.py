"""Shared fixtures for LLM proxy tests."""

import pytest
from aeonisk_llm_proxy.models import (
    LLMRequest,
    LLMProvider,
    LLMResponse,
    RequestPriority,
    RequestStatus,
    RoutingStrategy,
    ProxyConfig,
    RoutingRule,
    BatchSubmission,
)


@pytest.fixture
def openai_request():
    """Create a sample OpenAI request."""
    return LLMRequest(
        request_id="test-req-1",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )


@pytest.fixture
def anthropic_request():
    """Create a sample Anthropic request."""
    return LLMRequest(
        request_id="test-req-2",
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello"}],
    )


@pytest.fixture
def high_priority_request():
    """Create a high priority request."""
    return LLMRequest(
        request_id="test-req-high",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Urgent"}],
        priority=RequestPriority.HIGH,
    )


@pytest.fixture
def low_priority_request():
    """Create a low priority request."""
    return LLMRequest(
        request_id="test-req-low",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "No rush"}],
        priority=RequestPriority.LOW,
    )


@pytest.fixture
def default_config():
    """Create default proxy config."""
    return ProxyConfig()


@pytest.fixture
def small_batch_config():
    """Create config with small batch threshold for testing."""
    return ProxyConfig(
        batch_threshold=3,
        max_wait_seconds=10,
        max_idle_seconds=30,
    )
