# aeonisk-llm-proxy

A smart HTTP proxy that sits between your application and LLM providers. It exposes an OpenAI-compatible endpoint, automatically routing requests to either real-time APIs or provider Batch APIs (50% cheaper) based on priority, strategy hints, and queue state.

Your application sends a single HTTP POST. The proxy decides the cheapest path, manages batching logistics, polls for completion, and returns the response -- all behind one uniform interface.

## Table of Contents

- [Supported Providers](#supported-providers)
- [Quick Start](#quick-start)
- [Integration Guide](#integration-guide)
  - [Endpoint Reference](#endpoint-reference)
  - [Request Schema](#request-schema)
  - [Response Schema](#response-schema)
  - [Routing Control](#routing-control)
  - [Client Examples](#client-examples)
- [Management API](#management-api)
- [Configuration](#configuration)
- [How Batching Works](#how-batching-works)
- [Architecture](#architecture)

## Supported Providers

| Provider | Direct API | Batch API | Env Variable | Example Models |
|----------|-----------|-----------|--------------|----------------|
| OpenAI | yes | yes | `OPENAI_API_KEY` | gpt-4o, gpt-5, o3 |
| Anthropic | yes | yes | `ANTHROPIC_API_KEY` | claude-sonnet-4-5, claude-opus-4-6 |
| Grok (xAI) | yes | -- | `XAI_API_KEY` | grok-3 |
| Gemini | yes | -- | `GEMINI_API_KEY` | gemini-2.5-pro |
| DeepInfra | yes | -- | `DEEPINFRA_API_KEY` | meta-llama/Llama-3.3-70B-Instruct-Turbo |

Providers without a Batch API are always routed directly.

## Quick Start

```bash
# Install
git clone <repo-url>
cd aeonisk-llm-proxy
pip install -e .

# Configure (at least one provider key required)
cp .env.example .env
# Edit .env with your API keys

# Run
aeonisk-llm-proxy start
```

The proxy listens on `http://0.0.0.0:8000` by default. Interactive API docs (Swagger UI) are available at `/docs`.

## Integration Guide

### Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Submit an LLM request (OpenAI-compatible) |
| `POST` | `/submit` | Alias for the above |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Proxy-wide statistics |
| `GET` | `/queue/stats` | Per-queue statistics |
| `GET` | `/batches/stats` | Batch handler statistics |
| `GET` | `/batches` | List active batches |
| `GET` | `/batches/{batch_id}` | Get status of a specific batch |
| `DELETE` | `/batches/{batch_id}` | Delete a batch |
| `POST` | `/batches/purge` | Purge old/completed batches |
| `POST` | `/queue/flush/{provider}` | Manually flush a provider's queue |
| `GET` | `/requests/{request_id}` | Get status/response for a request |

### Request Schema

```
POST /v1/chat/completions
Content-Type: application/json
```

```jsonc
{
  // Required
  "provider": "openai",                // openai | anthropic | grok | gemini | deepinfra
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],

  // Optional model parameters
  "temperature": 0.7,                  // default: provider default
  "max_tokens": 1024,                  // default: provider default (Anthropic defaults to 4096)
  "top_p": null,                       // default: provider default

  // Routing hints
  "priority": "normal",                // low | normal | high
  "strategy": "auto",                  // auto | direct | batch

  // Metadata (optional)
  "request_id": "your-correlation-id", // auto-generated UUID if omitted
  "caller_id": "my-service",           // free-form identifier for your service
  "tags": {"project": "summarizer"}    // arbitrary key-value metadata
}
```

### Response Schema

```jsonc
{
  "request_id": "uuid",
  "status": "completed",               // completed | failed
  "content": "Hello! How can I help?", // null on failure
  "usage": {                           // token usage (shape varies by provider)
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  },
  "provider": "openai",
  "model": "gpt-4o",
  "completed_at": "2025-01-15T12:00:00Z",
  "routed_via": "direct",              // "direct" or "batch"
  "batch_id": null,                    // set when routed_via == "batch"
  "error": null                        // error message on failure
}
```

### Routing Control

Every request accepts two routing hints that together determine the execution path.

**`strategy`** -- explicit routing override:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Proxy decides based on priority, queue state, and config |
| `direct` | Always send immediately via real-time API |
| `batch` | Always queue for Batch API (OpenAI & Anthropic only) |

**`priority`** -- influences the `auto` strategy:

| Value | Behavior |
|-------|----------|
| `high` | Always routed direct, never batched |
| `normal` (default) | Auto-routed; will batch when the proxy prefers it |
| `low` | Always batched when the provider supports it |

Providers without a Batch API (Grok, Gemini, DeepInfra) are always sent direct regardless of these hints.

### Client Examples

All examples use the same endpoint (`POST /v1/chat/completions`). The only things that change between providers are the `provider` and `model` fields.

---

#### Per-Provider Examples (curl)

**OpenAI** -- supports both direct and batch:

```bash
# Direct (immediate response)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "strategy": "direct"
  }'

# Batch (50% cheaper, queued until flush)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize this long article..."}],
    "strategy": "batch"
  }'

# Auto (proxy decides based on queue state and priority)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "priority": "low"
  }'
```

**Anthropic** -- supports both direct and batch:

```bash
# Direct
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing."}
    ],
    "strategy": "direct"
  }'

# Batch (50% cheaper)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "system", "content": "Summarize concisely."},
      {"role": "user", "content": "...long document..."}
    ],
    "priority": "low"
  }'
```

> **Note:** The proxy automatically extracts `system` messages from the `messages` array and passes them as Anthropic's top-level `system` parameter. You don't need to handle this yourself.

**Grok (xAI)** -- direct only (no batch API):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "grok",
    "model": "grok-3",
    "messages": [{"role": "user", "content": "What is the meaning of life?"}]
  }'
```

**Gemini (Google)** -- direct only:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Write a haiku about code."}]
  }'
```

**DeepInfra** -- direct only (any model hosted on DeepInfra):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "deepinfra",
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "messages": [{"role": "user", "content": "Explain REST APIs."}]
  }'
```

> For Grok, Gemini, and DeepInfra, `strategy` and `priority` hints are accepted but ignored -- these providers always route direct since they have no Batch API.

---

#### Direct vs Batch Mode

The key difference is **latency vs cost**:

| Mode | Latency | Cost | When to use |
|------|---------|------|-------------|
| **Direct** | Seconds | Full price | Interactive / real-time use cases |
| **Batch** | Minutes to hours | ~50% cheaper | Offline processing, bulk workloads |

You control this per-request:

```bash
# Force direct -- response in seconds, full price
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai", "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Quick question"}],
    "strategy": "direct"
  }'

# Force batch -- response when batch completes, 50% cheaper
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai", "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Analyze this dataset..."}],
    "strategy": "batch"
  }'

# Auto -- let the proxy decide (default)
# HIGH priority → always direct
# NORMAL priority → proxy decides based on queue state
# LOW priority → always batch (when provider supports it)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic", "model": "claude-sonnet-4-5-20250929",
    "messages": [{"role": "user", "content": "Process this..."}],
    "priority": "high"
  }'
```

The response tells you which path was taken:

```jsonc
{
  "routed_via": "direct",  // or "batch"
  "batch_id": null          // set to a UUID when routed_via == "batch"
}
```

---

#### Python (requests)

```python
import requests

PROXY = "http://localhost:8000/v1/chat/completions"

# Direct request to any provider -- same interface
for provider, model in [
    ("openai", "gpt-4o"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("grok", "grok-3"),
    ("gemini", "gemini-2.5-pro"),
    ("deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
]:
    resp = requests.post(PROXY, json={
        "provider": provider,
        "model": model,
        "messages": [{"role": "user", "content": "Hello from Python!"}],
        "strategy": "direct",
    })
    data = resp.json()
    print(f"[{provider}] {data['content'][:80]}...")
    print(f"  routed_via={data['routed_via']}")
```

```python
# Batch request -- blocks until batch completes (can be minutes)
resp = requests.post(PROXY, json={
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize this document..."}],
    "priority": "low",       # batched for 50% savings
}, timeout=3600)             # batch may take a while

data = resp.json()
print(data["content"])
print(f"Routed via: {data['routed_via']}")  # "batch"
print(f"Batch ID:   {data['batch_id']}")
```

#### Python (aiohttp, async)

```python
import aiohttp, asyncio

PROXY = "http://localhost:8000/v1/chat/completions"

async def query(prompt, provider="openai", model="gpt-4o", **kwargs):
    async with aiohttp.ClientSession() as session:
        async with session.post(PROXY, json={
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }) as resp:
            return await resp.json()

async def main():
    # Fan out to multiple providers concurrently
    results = await asyncio.gather(
        query("Hello!", provider="openai", model="gpt-4o"),
        query("Hello!", provider="anthropic", model="claude-sonnet-4-5-20250929"),
        query("Hello!", provider="grok", model="grok-3"),
    )
    for r in results:
        print(f"[{r['provider']}] {r['content'][:60]}...")

asyncio.run(main())
```

#### TypeScript / JavaScript (fetch)

```typescript
const PROXY = "http://localhost:8000/v1/chat/completions";

async function query(
  provider: string,
  model: string,
  prompt: string,
  options: Record<string, unknown> = {}
) {
  const response = await fetch(PROXY, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider,
      model,
      messages: [{ role: "user", content: prompt }],
      ...options,
    }),
  });
  return response.json();
}

// Direct
const fast = await query("openai", "gpt-4o", "Hello", { strategy: "direct" });
console.log(fast.content, `(${fast.routed_via})`);

// Batch (cheaper, slower)
const cheap = await query("openai", "gpt-4o", "Summarize...", { strategy: "batch" });
console.log(cheap.content, `(${cheap.routed_via}, batch_id=${cheap.batch_id})`);
```

---

#### Bulk Processing (many requests, batch mode)

For offline workloads like processing a corpus, use `priority: "low"` to batch everything at 50% cost. Submit requests concurrently -- each POST blocks until its result is ready.

```python
import requests
from concurrent.futures import ThreadPoolExecutor

PROXY = "http://localhost:8000/v1/chat/completions"
documents = ["doc1 text...", "doc2 text...", "doc3 text..."]

def process_doc(doc):
    resp = requests.post(PROXY, json={
        "provider": "openai",
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": f"Summarize: {doc}"}],
        "priority": "low",   # will be batched
        "caller_id": "corpus-processor",
        "tags": {"job": "summarization"},
    }, timeout=7200)  # batches can take hours
    return resp.json()

# Submit all in parallel -- proxy batches them together
with ThreadPoolExecutor(max_workers=50) as pool:
    results = list(pool.map(process_doc, documents))

for r in results:
    print(f"[{r['request_id']}] {r['routed_via']}: {r['content'][:80]}...")
```

You can also mix providers in the same bulk job. The proxy batches per `(provider, model)` automatically:

```python
jobs = [
    ("openai", "gpt-4o", "Summarize this..."),
    ("anthropic", "claude-sonnet-4-5-20250929", "Translate this..."),
    ("openai", "gpt-4o", "Classify this..."),
    ("anthropic", "claude-sonnet-4-5-20250929", "Extract entities..."),
]

def run_job(args):
    provider, model, prompt = args
    return requests.post(PROXY, json={
        "provider": provider,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "priority": "low",
    }, timeout=7200).json()

with ThreadPoolExecutor(max_workers=50) as pool:
    results = list(pool.map(run_job, jobs))
```

---

#### Checking Request Status

If you want to check on a request after submission (e.g., from a different process):

```bash
# Check if a request has completed
curl -s http://localhost:8000/requests/<request_id>

# Response when still pending:
# { "request_id": "...", "status": "pending", "message": "Request is still processing" }

# Response when completed:
# { "request_id": "...", "status": "completed", "response": { "content": "...", ... } }
```

> **Important:** The `POST /v1/chat/completions` endpoint blocks until the response is ready (whether direct or batch). For batch-routed requests this can take minutes to hours. Set your HTTP client timeout accordingly, or use the `timeout` query parameter: `POST /v1/chat/completions?timeout=300` (returns 504 if the batch hasn't completed in 300 seconds).

## Management API

### Health & Stats

```bash
# CLI
aeonisk-llm-proxy status

# HTTP
curl http://localhost:8000/health
curl http://localhost:8000/stats
curl http://localhost:8000/queue/stats
curl http://localhost:8000/batches/stats
```

### Health Response

```jsonc
{
  "status": "healthy",             // healthy | stopped
  "uptime_seconds": 3600.0,
  "queue_size": 12,
  "active_batches": 2,
  "providers_healthy": {           // true if API key is configured
    "openai": true,
    "anthropic": true,
    "grok": false,
    "gemini": false,
    "deepinfra": false
  }
}
```

### Flushing Queues

Force-flush all queued requests for a provider to the Batch API immediately:

```bash
curl -X POST http://localhost:8000/queue/flush/openai
curl -X POST http://localhost:8000/queue/flush/anthropic
```

### Purging Batches

```bash
# Via CLI (proxy must be running)
aeonisk-llm-proxy purge --older-than 3600
aeonisk-llm-proxy purge --status failed
aeonisk-llm-proxy purge --batch-id <id>

# Directly from state file (proxy can be stopped)
aeonisk-llm-proxy purge --older-than 3600 --local

# Dry run
aeonisk-llm-proxy purge --older-than 3600 --dry-run

# Via HTTP
curl -X POST "http://localhost:8000/batches/purge?older_than_seconds=3600"
curl -X POST "http://localhost:8000/batches/purge?status=failed,expired"
curl -X DELETE "http://localhost:8000/batches/<batch_id>"
```

## Configuration

### Environment Variables

Set these in a `.env` file or your environment. See `.env.example` for a template.

#### API Keys

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI | At least one key |
| `ANTHROPIC_API_KEY` | Anthropic | required |
| `XAI_API_KEY` | Grok (xAI) | |
| `GEMINI_API_KEY` | Google Gemini | |
| `DEEPINFRA_API_KEY` | DeepInfra | |

#### Batch Queue Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_THRESHOLD` | `100` | Flush queue when this many requests accumulate |
| `BATCH_TIMEOUT` | `300` | Max seconds before queue is flushed (even below threshold) |
| `BATCH_MAX_IDLE` | `3600` | Flush queue after no new requests for this long |
| `BATCH_POLL_INTERVAL` | `60` | How often (seconds) to poll the provider Batch API for results |

### CLI Flags

CLI flags override environment variables:

```bash
aeonisk-llm-proxy start \
  --host 0.0.0.0 \
  --port 8080 \
  --batch-size 50 \
  --flush-interval 60 \
  --idle-timeout 1800 \
  --poll-interval 30 \
  --log-level debug \
  --reload            # dev mode with auto-reload
```

## How Batching Works

1. Client sends a request to the proxy via `POST /v1/chat/completions`.
2. The **Router** evaluates `strategy`, `priority`, provider capabilities, queue state, and routing rules to decide: **direct** (send now) or **batch** (queue it).
3. **Direct** requests are forwarded immediately to the provider API and the response is returned to the caller.
4. **Batch** requests are placed in the **BatchQueue**, keyed by `(provider, model)`.
5. The queue flushes when any trigger fires:
   - Request count reaches `BATCH_THRESHOLD`
   - Oldest request exceeds `BATCH_TIMEOUT` seconds
   - No new requests arrive for `BATCH_MAX_IDLE` seconds
   - Manual flush via `POST /queue/flush/{provider}`
6. Flushed requests are submitted to the provider's Batch API (OpenAI or Anthropic).
7. The **BatchAPIHandler** polls the provider for completion, downloads JSONL results, and the **ResponseTracker** resolves each waiting caller's response.
8. Batch state is persisted to `/tmp/llm_proxy_batches.json` for crash recovery.

Batch API requests are typically **50% cheaper** than real-time API calls. The trade-off is latency -- batch results can take minutes to hours.

## Architecture

See the [plantuml/](plantuml/) directory for detailed diagrams:

- **[component.puml](plantuml/component.puml)** -- High-level component architecture
- **[sequence_direct.puml](plantuml/sequence_direct.puml)** -- Direct request flow
- **[sequence_batch.puml](plantuml/sequence_batch.puml)** -- Batch request flow
- **[classes.puml](plantuml/classes.puml)** -- Data models and class relationships
- **[state_batch.puml](plantuml/state_batch.puml)** -- Batch lifecycle state machine

### Component Overview

```
Client
  |
  v
FastAPI (api.py)
  |
  v
LLMProxyServer (proxy_server.py)
  |
  +---> RequestRouter (router.py)        -- decides direct vs batch
  +---> DirectExecutor (direct_executor.py) -- real-time provider API calls
  +---> BatchQueue (batch_queue.py)       -- per-(provider, model) queuing
  +---> BatchAPIHandler (batch_handler.py) -- submits/polls provider Batch APIs
  +---> ResponseTracker (response_tracker.py) -- correlates responses to callers
```
