"""
LLM Batching Proxy

A smart proxy server that batches LLM requests on-the-fly, routing them to
either direct APIs (fast) or Batch APIs (50% cheaper) based on configurable rules.
"""

from .proxy_server import LLMProxyServer
from .batch_queue import BatchQueue
from .router import RequestRouter

__all__ = ["LLMProxyServer", "BatchQueue", "RequestRouter"]
