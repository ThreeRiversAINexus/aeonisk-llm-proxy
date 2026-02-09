"""
Data models for LLM batching proxy.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class RoutingStrategy(str, Enum):
    """Routing strategy for LLM requests."""
    DIRECT = "direct"  # Send immediately to API
    BATCH = "batch"    # Queue for batch API
    AUTO = "auto"      # Auto-decide based on rules


class RequestPriority(str, Enum):
    """Request priority levels."""
    LOW = "low"        # Can wait hours, use batch
    NORMAL = "normal"  # Default, auto-route
    HIGH = "high"      # Send immediately, no batching


class RequestStatus(str, Enum):
    """Status of a proxied request."""
    QUEUED = "queued"          # Waiting in queue
    BATCHED = "batched"        # Submitted to batch API
    PROCESSING = "processing"  # Being processed
    COMPLETED = "completed"    # Response ready
    FAILED = "failed"          # Error occurred


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Request Models

class LLMRequest(BaseModel):
    """A proxied LLM request."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: LLMProvider
    model: str
    messages: List[Dict[str, str]]

    # Optional parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Routing hints
    priority: RequestPriority = RequestPriority.NORMAL
    strategy: RoutingStrategy = RoutingStrategy.AUTO

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    caller_id: Optional[str] = None  # For tracking who made the request
    tags: Dict[str, str] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A proxied LLM response."""
    request_id: str
    status: RequestStatus

    # Response data (if completed)
    content: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # Can contain nested dicts (prompt_tokens_details, etc.)

    # Metadata
    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None

    # Error info (if failed)
    error: Optional[str] = None

    # Routing info
    routed_via: Optional[str] = None  # "direct", "batch", "cached"
    batch_id: Optional[str] = None    # If routed via batch API


# Batch Models

class BatchSubmission(BaseModel):
    """A batch submission to provider batch API."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: LLMProvider
    provider_batch_id: Optional[str] = None  # Provider's batch ID

    # Requests in this batch
    request_ids: List[str] = Field(default_factory=list)
    total_requests: int = 0

    # Status
    status: str = "pending"  # pending, submitted, processing, completed, failed

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    input_file_path: Optional[str] = None
    output_file_path: Optional[str] = None
    completed_requests: int = 0
    failed_requests: int = 0


# Routing Configuration

class RoutingRule(BaseModel):
    """Rules for routing requests."""
    name: str

    # Conditions (all must match)
    provider: Optional[LLMProvider] = None
    model_pattern: Optional[str] = None  # Regex pattern
    priority: Optional[RequestPriority] = None
    min_queue_size: Optional[int] = None
    max_wait_seconds: Optional[int] = None

    # Action
    route_to: RoutingStrategy

    # Priority (higher = evaluated first)
    rule_priority: int = 0


class ProxyConfig(BaseModel):
    """Configuration for LLM proxy."""

    # Queue settings
    batch_threshold: int = 100  # Flush queue when this many requests
    max_wait_seconds: int = 300  # Flush queue after this many seconds
    max_idle_seconds: int = 3600  # Flush queue if no new requests for this long (for overnight batches)
    poll_interval_seconds: int = 60  # Poll batch API every N seconds

    # Routing defaults
    default_strategy: RoutingStrategy = RoutingStrategy.AUTO
    high_priority_always_direct: bool = True
    low_priority_always_batch: bool = True

    # Cost optimization
    prefer_batch_api: bool = True  # Prefer batch when possible
    batch_api_min_requests: int = 10  # Min requests to use batch API

    # Routing rules
    routing_rules: List[RoutingRule] = Field(default_factory=list)

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """Load configuration from environment variables."""
        import os
        return cls(
            batch_threshold=int(os.getenv('BATCH_THRESHOLD', '100')),
            max_wait_seconds=int(os.getenv('BATCH_TIMEOUT', '300')),
            max_idle_seconds=int(os.getenv('BATCH_MAX_IDLE', '3600')),
            poll_interval_seconds=int(os.getenv('BATCH_POLL_INTERVAL', '60')),
        )


# Statistics

class ProxyStats(BaseModel):
    """Proxy statistics."""
    total_requests: int = 0
    queued_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0

    # Routing breakdown
    routed_direct: int = 0
    routed_batch: int = 0

    # Batches
    active_batches: int = 0
    completed_batches: int = 0

    # Performance
    avg_response_time_seconds: float = 0.0
    total_cost_usd: float = 0.0
    estimated_savings_usd: float = 0.0


class HealthCheck(BaseModel):
    """Proxy health check."""
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    queue_size: int
    active_batches: int

    # Provider connectivity
    providers_healthy: Dict[str, bool]

    # Recent errors
    recent_errors: List[str] = Field(default_factory=list)
