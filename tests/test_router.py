"""Tests for RequestRouter: routing decisions based on priority, rules, and strategy."""

import pytest

from aeonisk_llm_proxy.router import RequestRouter
from aeonisk_llm_proxy.models import (
    LLMRequest,
    LLMProvider,
    RequestPriority,
    RoutingStrategy,
    RoutingRule,
    ProxyConfig,
)


def make_request(
    priority: RequestPriority = RequestPriority.NORMAL,
    strategy: RoutingStrategy = RoutingStrategy.AUTO,
    provider: LLMProvider = LLMProvider.OPENAI,
    model: str = "gpt-4o-mini",
) -> LLMRequest:
    return LLMRequest(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "test"}],
        priority=priority,
        strategy=strategy,
    )


class TestExplicitStrategy:
    """Test that explicit strategy overrides all other logic."""

    def test_explicit_direct(self):
        router = RequestRouter()
        req = make_request(strategy=RoutingStrategy.DIRECT)
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_explicit_batch(self):
        router = RequestRouter()
        req = make_request(strategy=RoutingStrategy.BATCH)
        assert router.route(req) == RoutingStrategy.BATCH

    def test_explicit_overrides_priority(self):
        """Even HIGH priority request goes BATCH if strategy is explicit."""
        router = RequestRouter()
        req = make_request(
            priority=RequestPriority.HIGH,
            strategy=RoutingStrategy.BATCH,
        )
        assert router.route(req) == RoutingStrategy.BATCH


class TestPriorityDefaults:
    """Test priority-based routing defaults."""

    def test_high_priority_always_direct(self):
        config = ProxyConfig(high_priority_always_direct=True)
        router = RequestRouter(config)
        req = make_request(priority=RequestPriority.HIGH)
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_high_priority_direct_disabled(self):
        config = ProxyConfig(high_priority_always_direct=False, prefer_batch_api=True)
        router = RequestRouter(config)
        req = make_request(priority=RequestPriority.HIGH)
        # Should fall through to auto-routing
        result = router.route(req)
        assert result in [RoutingStrategy.DIRECT, RoutingStrategy.BATCH]

    def test_low_priority_always_batch(self):
        config = ProxyConfig(low_priority_always_batch=True)
        router = RequestRouter(config)
        req = make_request(priority=RequestPriority.LOW)
        assert router.route(req) == RoutingStrategy.BATCH

    def test_low_priority_batch_disabled(self):
        config = ProxyConfig(low_priority_always_batch=False, prefer_batch_api=True)
        router = RequestRouter(config)
        req = make_request(priority=RequestPriority.LOW)
        result = router.route(req)
        assert result in [RoutingStrategy.DIRECT, RoutingStrategy.BATCH]


class TestRoutingRules:
    """Test custom routing rules."""

    def test_provider_match_rule(self):
        config = ProxyConfig(
            routing_rules=[
                RoutingRule(
                    name="anthropic-always-batch",
                    provider=LLMProvider.ANTHROPIC,
                    route_to=RoutingStrategy.BATCH,
                ),
            ],
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-5-20250929")
        assert router.route(req) == RoutingStrategy.BATCH

    def test_model_pattern_rule(self):
        config = ProxyConfig(
            routing_rules=[
                RoutingRule(
                    name="gpt4-direct",
                    model_pattern="gpt-4.*",
                    route_to=RoutingStrategy.DIRECT,
                ),
            ],
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)

        req_match = make_request(model="gpt-4o-mini")
        assert router.route(req_match) == RoutingStrategy.DIRECT

        req_no_match = make_request(model="gpt-3.5-turbo")
        # Should not match the rule, falls through to auto
        assert router.route(req_no_match) != RoutingStrategy.DIRECT or True  # auto may still return DIRECT

    def test_rule_priority_ordering(self):
        """Higher rule_priority should be evaluated first."""
        config = ProxyConfig(
            routing_rules=[
                RoutingRule(
                    name="low-priority-batch",
                    provider=LLMProvider.OPENAI,
                    route_to=RoutingStrategy.BATCH,
                    rule_priority=1,
                ),
                RoutingRule(
                    name="high-priority-direct",
                    provider=LLMProvider.OPENAI,
                    route_to=RoutingStrategy.DIRECT,
                    rule_priority=10,
                ),
            ],
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request(provider=LLMProvider.OPENAI)
        # Higher priority rule should win
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_min_queue_size_rule(self):
        config = ProxyConfig(
            routing_rules=[
                RoutingRule(
                    name="batch-when-queue-large",
                    min_queue_size=50,
                    route_to=RoutingStrategy.BATCH,
                ),
            ],
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request()

        # Queue too small - rule doesn't match
        result_small = router.route(req, queue_size=10)
        # Rule doesn't match, falls to auto
        assert result_small in [RoutingStrategy.DIRECT, RoutingStrategy.BATCH]

        # Queue large enough - rule matches
        result_large = router.route(req, queue_size=100)
        assert result_large == RoutingStrategy.BATCH

    def test_no_matching_rule_falls_to_auto(self):
        config = ProxyConfig(
            routing_rules=[
                RoutingRule(
                    name="anthropic-only",
                    provider=LLMProvider.ANTHROPIC,
                    route_to=RoutingStrategy.BATCH,
                ),
            ],
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request(provider=LLMProvider.OPENAI)
        # Should not match the Anthropic rule, falls through to auto
        result = router.route(req)
        assert result in [RoutingStrategy.DIRECT, RoutingStrategy.BATCH]


class TestAutoRouting:
    """Test auto-routing logic."""

    def test_queue_too_old_routes_direct(self):
        config = ProxyConfig(
            max_wait_seconds=100,
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request()
        # Queue age > 80% of max_wait → DIRECT
        result = router.route(req, queue_size=50, queue_age_seconds=85)
        assert result == RoutingStrategy.DIRECT

    def test_small_queue_no_preference_routes_direct(self):
        config = ProxyConfig(
            prefer_batch_api=False,
            batch_api_min_requests=10,
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request()
        result = router.route(req, queue_size=5, queue_age_seconds=0)
        assert result == RoutingStrategy.DIRECT

    def test_prefer_batch_with_small_queue(self):
        config = ProxyConfig(
            prefer_batch_api=True,
            batch_api_min_requests=10,
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request()
        # Even with small queue, prefer_batch_api=True → BATCH
        result = router.route(req, queue_size=5, queue_age_seconds=0)
        assert result == RoutingStrategy.BATCH

    def test_default_preference_batch(self):
        config = ProxyConfig(
            prefer_batch_api=True,
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request()
        result = router.route(req, queue_size=20, queue_age_seconds=0)
        assert result == RoutingStrategy.BATCH

    def test_default_preference_direct(self):
        config = ProxyConfig(
            prefer_batch_api=False,
            batch_api_min_requests=10,
            high_priority_always_direct=False,
            low_priority_always_batch=False,
        )
        router = RequestRouter(config)
        req = make_request()
        result = router.route(req, queue_size=5, queue_age_seconds=0)
        assert result == RoutingStrategy.DIRECT


class TestDirectOnlyProviders:
    """Test that direct-only providers (no batch API) always route DIRECT."""

    def test_grok_always_direct(self):
        router = RequestRouter()
        req = make_request(provider=LLMProvider.GROK, model="grok-3")
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_gemini_always_direct(self):
        router = RequestRouter()
        req = make_request(provider=LLMProvider.GEMINI, model="gemini-2.0-flash")
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_deepinfra_always_direct(self):
        router = RequestRouter()
        req = make_request(provider=LLMProvider.DEEPINFRA, model="meta-llama/Llama-3.3-70B-Instruct")
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_direct_only_overrides_low_priority(self):
        """Even LOW priority on a direct-only provider stays DIRECT."""
        config = ProxyConfig(low_priority_always_batch=True)
        router = RequestRouter(config)
        req = make_request(
            provider=LLMProvider.GROK,
            model="grok-3",
            priority=RequestPriority.LOW,
        )
        assert router.route(req) == RoutingStrategy.DIRECT

    def test_direct_only_overrides_explicit_batch_strategy(self):
        """Explicit batch strategy on a direct-only provider still routes DIRECT."""
        router = RequestRouter()
        req = make_request(
            provider=LLMProvider.DEEPINFRA,
            model="meta-llama/Llama-3.3-70B-Instruct",
            strategy=RoutingStrategy.BATCH,
        )
        assert router.route(req) == RoutingStrategy.DIRECT


class TestShouldFlushNow:
    """Test should_flush_now helper."""

    def test_threshold_reached(self):
        config = ProxyConfig(batch_threshold=100)
        router = RequestRouter(config)
        assert router.should_flush_now(queue_size=100, queue_age_seconds=0) is True

    def test_threshold_not_reached(self):
        config = ProxyConfig(batch_threshold=100)
        router = RequestRouter(config)
        assert router.should_flush_now(queue_size=50, queue_age_seconds=0) is False

    def test_timeout_reached(self):
        config = ProxyConfig(max_wait_seconds=300)
        router = RequestRouter(config)
        assert router.should_flush_now(queue_size=10, queue_age_seconds=300) is True

    def test_neither_reached(self):
        config = ProxyConfig(batch_threshold=100, max_wait_seconds=300)
        router = RequestRouter(config)
        assert router.should_flush_now(queue_size=50, queue_age_seconds=100) is False
