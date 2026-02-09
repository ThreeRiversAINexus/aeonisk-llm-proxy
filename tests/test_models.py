"""Tests for Pydantic models, enums, and configuration."""

import os
import pytest
from datetime import datetime

from aeonisk_llm_proxy.models import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    RequestPriority,
    RequestStatus,
    RoutingStrategy,
    BatchSubmission,
    RoutingRule,
    ProxyConfig,
    ProxyStats,
    HealthCheck,
    BATCH_PROVIDERS,
)


class TestEnums:
    """Test enum values and string representation."""

    def test_routing_strategy_values(self):
        assert RoutingStrategy.DIRECT == "direct"
        assert RoutingStrategy.BATCH == "batch"
        assert RoutingStrategy.AUTO == "auto"

    def test_request_priority_values(self):
        assert RequestPriority.LOW == "low"
        assert RequestPriority.NORMAL == "normal"
        assert RequestPriority.HIGH == "high"

    def test_request_status_values(self):
        assert RequestStatus.QUEUED == "queued"
        assert RequestStatus.BATCHED == "batched"
        assert RequestStatus.PROCESSING == "processing"
        assert RequestStatus.COMPLETED == "completed"
        assert RequestStatus.FAILED == "failed"

    def test_llm_provider_values(self):
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.GROK == "grok"
        assert LLMProvider.GEMINI == "gemini"
        assert LLMProvider.DEEPINFRA == "deepinfra"

    def test_enum_from_string(self):
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("grok") == LLMProvider.GROK
        assert LLMProvider("gemini") == LLMProvider.GEMINI
        assert LLMProvider("deepinfra") == LLMProvider.DEEPINFRA
        assert RequestPriority("high") == RequestPriority.HIGH
        assert RoutingStrategy("batch") == RoutingStrategy.BATCH


class TestBatchProviders:
    """Test BATCH_PROVIDERS constant."""

    def test_only_openai_and_anthropic_support_batch(self):
        assert LLMProvider.OPENAI in BATCH_PROVIDERS
        assert LLMProvider.ANTHROPIC in BATCH_PROVIDERS
        assert LLMProvider.GROK not in BATCH_PROVIDERS
        assert LLMProvider.GEMINI not in BATCH_PROVIDERS
        assert LLMProvider.DEEPINFRA not in BATCH_PROVIDERS


class TestNewProviderRequests:
    """Test creating requests for new providers."""

    def test_grok_request(self):
        req = LLMRequest(
            provider=LLMProvider.GROK,
            model="grok-3",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.provider == LLMProvider.GROK
        assert req.model == "grok-3"

    def test_gemini_request(self):
        req = LLMRequest(
            provider=LLMProvider.GEMINI,
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.provider == LLMProvider.GEMINI

    def test_deepinfra_request(self):
        req = LLMRequest(
            provider=LLMProvider.DEEPINFRA,
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.provider == LLMProvider.DEEPINFRA


class TestLLMRequest:
    """Test LLMRequest model creation and defaults."""

    def test_minimal_creation(self):
        req = LLMRequest(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.provider == LLMProvider.OPENAI
        assert req.model == "gpt-4o-mini"
        assert len(req.messages) == 1
        assert req.request_id  # Auto-generated UUID

    def test_defaults(self):
        req = LLMRequest(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.priority == RequestPriority.NORMAL
        assert req.strategy == RoutingStrategy.AUTO
        assert req.temperature is None
        assert req.max_tokens is None
        assert req.top_p is None
        assert req.caller_id is None
        assert req.tags == {}
        assert isinstance(req.created_at, datetime)

    def test_explicit_request_id(self):
        req = LLMRequest(
            request_id="my-custom-id",
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.request_id == "my-custom-id"

    def test_with_optional_params(self):
        req = LLMRequest(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            priority=RequestPriority.HIGH,
            strategy=RoutingStrategy.DIRECT,
            caller_id="test-caller",
            tags={"project": "test"},
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 1000
        assert req.top_p == 0.9
        assert req.priority == RequestPriority.HIGH
        assert req.strategy == RoutingStrategy.DIRECT
        assert req.caller_id == "test-caller"
        assert req.tags == {"project": "test"}

    def test_unique_request_ids(self):
        req1 = LLMRequest(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        req2 = LLMRequest(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req1.request_id != req2.request_id


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_completed_response(self):
        resp = LLMResponse(
            request_id="test-1",
            status=RequestStatus.COMPLETED,
            content="Hello there!",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            routed_via="direct",
        )
        assert resp.status == RequestStatus.COMPLETED
        assert resp.content == "Hello there!"
        assert resp.error is None

    def test_failed_response(self):
        resp = LLMResponse(
            request_id="test-2",
            status=RequestStatus.FAILED,
            error="API key invalid",
        )
        assert resp.status == RequestStatus.FAILED
        assert resp.content is None
        assert resp.error == "API key invalid"

    def test_batch_response(self):
        resp = LLMResponse(
            request_id="test-3",
            status=RequestStatus.COMPLETED,
            content="Batch result",
            routed_via="batch",
            batch_id="batch-123",
        )
        assert resp.routed_via == "batch"
        assert resp.batch_id == "batch-123"


class TestBatchSubmission:
    """Test BatchSubmission model."""

    def test_creation_with_defaults(self):
        sub = BatchSubmission(
            provider=LLMProvider.OPENAI,
            request_ids=["req-1", "req-2"],
            total_requests=2,
        )
        assert sub.batch_id  # Auto-generated
        assert sub.provider == LLMProvider.OPENAI
        assert sub.total_requests == 2
        assert sub.status == "pending"
        assert sub.provider_batch_id is None
        assert isinstance(sub.created_at, datetime)

    def test_unique_batch_ids(self):
        sub1 = BatchSubmission(provider=LLMProvider.OPENAI)
        sub2 = BatchSubmission(provider=LLMProvider.OPENAI)
        assert sub1.batch_id != sub2.batch_id


class TestRoutingRule:
    """Test RoutingRule model."""

    def test_simple_rule(self):
        rule = RoutingRule(
            name="always-batch-anthropic",
            provider=LLMProvider.ANTHROPIC,
            route_to=RoutingStrategy.BATCH,
        )
        assert rule.name == "always-batch-anthropic"
        assert rule.route_to == RoutingStrategy.BATCH
        assert rule.rule_priority == 0

    def test_complex_rule(self):
        rule = RoutingRule(
            name="fast-gpt4",
            provider=LLMProvider.OPENAI,
            model_pattern="gpt-4.*",
            priority=RequestPriority.HIGH,
            route_to=RoutingStrategy.DIRECT,
            rule_priority=10,
        )
        assert rule.model_pattern == "gpt-4.*"
        assert rule.rule_priority == 10


class TestProxyConfig:
    """Test ProxyConfig model and from_env."""

    def test_defaults(self):
        config = ProxyConfig()
        assert config.batch_threshold == 100
        assert config.max_wait_seconds == 300
        assert config.max_idle_seconds == 3600
        assert config.poll_interval_seconds == 60
        assert config.high_priority_always_direct is True
        assert config.low_priority_always_batch is True
        assert config.prefer_batch_api is True
        assert config.batch_api_min_requests == 10
        assert config.routing_rules == []

    def test_from_env_defaults(self, monkeypatch):
        # Clear any existing env vars
        monkeypatch.delenv('BATCH_THRESHOLD', raising=False)
        monkeypatch.delenv('BATCH_TIMEOUT', raising=False)
        monkeypatch.delenv('BATCH_MAX_IDLE', raising=False)
        monkeypatch.delenv('BATCH_POLL_INTERVAL', raising=False)

        config = ProxyConfig.from_env()
        assert config.batch_threshold == 100
        assert config.max_wait_seconds == 300
        assert config.max_idle_seconds == 3600
        assert config.poll_interval_seconds == 60

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv('BATCH_THRESHOLD', '50')
        monkeypatch.setenv('BATCH_TIMEOUT', '120')
        monkeypatch.setenv('BATCH_MAX_IDLE', '1800')
        monkeypatch.setenv('BATCH_POLL_INTERVAL', '30')

        config = ProxyConfig.from_env()
        assert config.batch_threshold == 50
        assert config.max_wait_seconds == 120
        assert config.max_idle_seconds == 1800
        assert config.poll_interval_seconds == 30

    def test_with_routing_rules(self):
        config = ProxyConfig(
            routing_rules=[
                RoutingRule(
                    name="test-rule",
                    provider=LLMProvider.OPENAI,
                    route_to=RoutingStrategy.BATCH,
                    rule_priority=5,
                ),
            ]
        )
        assert len(config.routing_rules) == 1
        assert config.routing_rules[0].name == "test-rule"


class TestProxyStats:
    """Test ProxyStats model."""

    def test_defaults(self):
        stats = ProxyStats()
        assert stats.total_requests == 0
        assert stats.completed_requests == 0
        assert stats.routed_direct == 0
        assert stats.routed_batch == 0
        assert stats.estimated_savings_usd == 0.0


class TestHealthCheck:
    """Test HealthCheck model."""

    def test_healthy(self):
        health = HealthCheck(
            status="healthy",
            uptime_seconds=3600.0,
            queue_size=5,
            active_batches=2,
            providers_healthy={"openai": True, "anthropic": True},
        )
        assert health.status == "healthy"
        assert health.recent_errors == []
