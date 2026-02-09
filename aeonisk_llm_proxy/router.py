"""
Smart router for LLM requests.

Decides whether to send requests directly or batch them based on
priority, cost optimization, and configurable rules.
"""

import re
import logging
from typing import List, Optional

from .models import (
    LLMRequest,
    RoutingStrategy,
    RequestPriority,
    RoutingRule,
    ProxyConfig,
)

logger = logging.getLogger(__name__)


class RequestRouter:
    """Routes LLM requests to direct or batch processing."""

    def __init__(self, config: Optional[ProxyConfig] = None):
        """
        Initialize router.

        Args:
            config: Proxy configuration
        """
        self.config = config or ProxyConfig()

    def route(
        self,
        request: LLMRequest,
        queue_size: int = 0,
        queue_age_seconds: float = 0,
    ) -> RoutingStrategy:
        """
        Determine routing strategy for a request.

        Args:
            request: LLM request to route
            queue_size: Current queue size for this provider
            queue_age_seconds: Age of oldest request in queue

        Returns:
            Routing strategy (DIRECT or BATCH)
        """
        # 1. Check explicit strategy override
        if request.strategy != RoutingStrategy.AUTO:
            logger.debug(
                f"Request {request.request_id}: explicit strategy "
                f"{request.strategy}"
            )
            return request.strategy

        # 2. Check priority-based defaults
        if self.config.high_priority_always_direct:
            if request.priority == RequestPriority.HIGH:
                logger.debug(
                    f"Request {request.request_id}: HIGH priority \u2192 DIRECT"
                )
                return RoutingStrategy.DIRECT

        if self.config.low_priority_always_batch:
            if request.priority == RequestPriority.LOW:
                logger.debug(
                    f"Request {request.request_id}: LOW priority \u2192 BATCH"
                )
                return RoutingStrategy.BATCH

        # 3. Evaluate custom routing rules (sorted by priority)
        for rule in sorted(
            self.config.routing_rules,
            key=lambda r: r.rule_priority,
            reverse=True,
        ):
            if self._matches_rule(request, rule, queue_size, queue_age_seconds):
                logger.debug(
                    f"Request {request.request_id}: matched rule '{rule.name}' "
                    f"\u2192 {rule.route_to}"
                )
                return rule.route_to

        # 4. Default auto-routing logic
        return self._auto_route(request, queue_size, queue_age_seconds)

    def _matches_rule(
        self,
        request: LLMRequest,
        rule: RoutingRule,
        queue_size: int,
        queue_age_seconds: float,
    ) -> bool:
        """Check if request matches a routing rule."""
        # Provider match
        if rule.provider and request.provider != rule.provider:
            return False

        # Model pattern match
        if rule.model_pattern:
            if not re.match(rule.model_pattern, request.model):
                return False

        # Priority match
        if rule.priority and request.priority != rule.priority:
            return False

        # Queue size condition
        if rule.min_queue_size and queue_size < rule.min_queue_size:
            return False

        # Queue age condition
        if rule.max_wait_seconds and queue_age_seconds > rule.max_wait_seconds:
            return False

        return True

    def _auto_route(
        self,
        request: LLMRequest,
        queue_size: int,
        queue_age_seconds: float,
    ) -> RoutingStrategy:
        """
        Auto-routing logic when no rules match.

        Default behavior:
        - Prefer batch API if config says so
        - But flush immediately if queue is old
        - Use direct if queue is too small for efficient batching
        """
        # If queue is old, send direct to avoid further delays
        if queue_age_seconds > self.config.max_wait_seconds * 0.8:
            logger.debug(
                f"Request {request.request_id}: queue too old "
                f"({queue_age_seconds:.1f}s) \u2192 DIRECT"
            )
            return RoutingStrategy.DIRECT

        # If queue is too small for efficient batching, send direct
        if queue_size < self.config.batch_api_min_requests:
            if not self.config.prefer_batch_api:
                logger.debug(
                    f"Request {request.request_id}: queue too small "
                    f"({queue_size}) and batch not preferred \u2192 DIRECT"
                )
                return RoutingStrategy.DIRECT

        # Default based on preference
        strategy = (
            RoutingStrategy.BATCH
            if self.config.prefer_batch_api
            else RoutingStrategy.DIRECT
        )

        logger.debug(
            f"Request {request.request_id}: default routing \u2192 {strategy}"
        )
        return strategy

    def should_flush_now(
        self,
        queue_size: int,
        queue_age_seconds: float,
    ) -> bool:
        """
        Determine if queue should be flushed immediately.

        Args:
            queue_size: Current queue size
            queue_age_seconds: Age of oldest request

        Returns:
            True if should flush now
        """
        # Threshold reached
        if queue_size >= self.config.batch_threshold:
            return True

        # Timeout reached
        if queue_age_seconds >= self.config.max_wait_seconds:
            return True

        return False
