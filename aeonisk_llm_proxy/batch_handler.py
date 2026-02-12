"""
Batch API handler for submitting and polling batch requests.

Integrates with OpenAI and Anthropic Batch APIs.
"""

import os
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    LLMRequest,
    LLMProvider,
    BatchSubmission,
)

logger = logging.getLogger(__name__)


class BatchAPIHandler:
    """Handles batch API submissions and polling."""

    def __init__(self, poll_interval_seconds: int = 60, state_file: str = "/tmp/llm_proxy_batches.json"):
        """
        Initialize batch API handler.

        Args:
            poll_interval_seconds: Interval for polling batch status
            state_file: Path to persist batch state for recovery
        """
        self.poll_interval = poll_interval_seconds
        self.state_file = Path(state_file)

        # API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        # Active batches being polled
        self.active_batches: Dict[str, BatchSubmission] = {}

        # Polling tasks
        self.poll_tasks: Dict[str, asyncio.Task] = {}

        # Load persisted state
        self._load_state()

    async def submit_batch(
        self,
        submission: BatchSubmission,
        requests: List[LLMRequest],
    ) -> str:
        """
        Submit batch to provider API.

        Args:
            submission: Batch submission metadata
            requests: List of requests in batch

        Returns:
            Provider batch ID
        """
        if submission.provider == LLMProvider.OPENAI:
            return await self._submit_openai_batch(submission, requests)
        elif submission.provider == LLMProvider.ANTHROPIC:
            return await self._submit_anthropic_batch(submission, requests)
        else:
            raise ValueError(f"Unsupported provider: {submission.provider}")

    async def _submit_openai_batch(
        self,
        submission: BatchSubmission,
        requests: List[LLMRequest],
    ) -> str:
        """Submit batch to OpenAI Batch API."""
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # 1. Create JSONL file
        batch_file = Path(f"/tmp/openai_batch_{submission.batch_id}.jsonl")

        with open(batch_file, "w") as f:
            for req in requests:
                batch_req = {
                    "custom_id": req.request_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": req.model,
                        "messages": req.messages,
                    },
                }

                # Add optional parameters
                # GPT-5 models require max_completion_tokens instead of max_tokens
                # and only support temperature=1 (default)
                is_gpt5 = req.model.startswith('gpt-5')

                if req.temperature is not None and not is_gpt5:
                    batch_req["body"]["temperature"] = req.temperature
                if req.max_tokens is not None:
                    if is_gpt5:
                        batch_req["body"]["max_completion_tokens"] = req.max_tokens
                    else:
                        batch_req["body"]["max_tokens"] = req.max_tokens
                if req.top_p is not None:
                    batch_req["body"]["top_p"] = req.top_p

                f.write(json.dumps(batch_req) + "\n")

        submission.input_file_path = str(batch_file)

        # 2. Upload file with retry logic
        max_retries = 5
        retry_delay = 60  # Start with 1 minute for batch API operations
        file_id = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    with open(batch_file, "rb") as f:
                        data = aiohttp.FormData()
                        data.add_field(
                            "file",
                            f,
                            filename=batch_file.name,
                            content_type="application/x-ndjson",
                        )
                        data.add_field("purpose", "batch")

                        async with session.post(
                            "https://api.openai.com/v1/files",
                            headers={"Authorization": f"Bearer {self.openai_key}"},
                            data=data,
                        ) as resp:
                            if resp.status == 200:
                                file_data = await resp.json()
                                file_id = file_data["id"]
                                break  # Success
                            else:
                                error = await resp.text()
                                if attempt < max_retries - 1:
                                    logger.warning(
                                        f"File upload failed (attempt {attempt + 1}/{max_retries}): {error}. "
                                        f"Retrying in {retry_delay}s..."
                                    )
                                    await asyncio.sleep(retry_delay)
                                    retry_delay = min(600, retry_delay * 2)  # Cap at 10 minutes
                                else:
                                    raise Exception(f"File upload failed after {max_retries} attempts: {error}")

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Network error during file upload (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(600, retry_delay * 2)
                else:
                    raise Exception(f"File upload failed after {max_retries} attempts: {e}")

        if not file_id:
            raise Exception("File upload failed: no file_id returned")

        # 3. Create batch with retry logic
        retry_delay = 60  # Reset for batch creation
        provider_batch_id = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/batches",
                        headers={
                            "Authorization": f"Bearer {self.openai_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "input_file_id": file_id,
                            "endpoint": "/v1/chat/completions",
                            "completion_window": "24h",
                            "metadata": {"batch_id": submission.batch_id},
                        },
                    ) as resp:
                        if resp.status in [200, 201]:
                            batch_data = await resp.json()
                            provider_batch_id = batch_data["id"]
                            break  # Success
                        else:
                            error = await resp.text()
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Batch creation failed (attempt {attempt + 1}/{max_retries}): {error}. "
                                    f"Retrying in {retry_delay}s..."
                                )
                                await asyncio.sleep(retry_delay)
                                retry_delay = min(600, retry_delay * 2)
                            else:
                                raise Exception(f"Batch creation failed after {max_retries} attempts: {error}")

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Network error during batch creation (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(600, retry_delay * 2)
                else:
                    raise Exception(f"Batch creation failed after {max_retries} attempts: {e}")

        if not provider_batch_id:
            raise Exception("Batch creation failed: no provider_batch_id returned")

        logger.info(
            f"Submitted OpenAI batch {submission.batch_id} "
            f"(provider ID: {provider_batch_id}, {len(requests)} requests)"
        )

        # Start polling
        submission.provider_batch_id = provider_batch_id
        submission.status = "submitted"
        submission.submitted_at = datetime.utcnow()
        self.active_batches[submission.batch_id] = submission
        self._save_state()  # Persist state

        task = asyncio.create_task(
            self._poll_openai_batch(submission.batch_id, provider_batch_id)
        )
        self.poll_tasks[submission.batch_id] = task

        return provider_batch_id

    async def _submit_anthropic_batch(
        self,
        submission: BatchSubmission,
        requests: List[LLMRequest],
    ) -> str:
        """Submit batch to Anthropic Message Batches API."""
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        # Format requests for Anthropic
        batch_requests = []
        for req in requests:
            # Anthropic requires system message as top-level param, not in messages array
            system_content = None
            non_system_messages = []
            for msg in req.messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                else:
                    non_system_messages.append(msg)

            batch_req = {
                "custom_id": req.request_id,
                "params": {
                    "model": req.model,
                    "max_tokens": req.max_tokens or 4096,
                    "messages": non_system_messages,
                },
            }

            if system_content:
                batch_req["params"]["system"] = system_content

            if req.temperature is not None:
                batch_req["params"]["temperature"] = req.temperature
            if req.top_p is not None:
                batch_req["params"]["top_p"] = req.top_p

            batch_requests.append(batch_req)

        # Submit batch with retry logic
        max_retries = 5
        retry_delay = 60  # Start with 1 minute for batch API operations
        provider_batch_id = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.anthropic.com/v1/messages/batches",
                        headers={
                            "x-api-key": self.anthropic_key,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                        },
                        json={"requests": batch_requests},
                    ) as resp:
                        if resp.status in [200, 201]:
                            batch_data = await resp.json()
                            provider_batch_id = batch_data["id"]
                            break  # Success
                        else:
                            error = await resp.text()
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Anthropic batch creation failed (attempt {attempt + 1}/{max_retries}): {error}. "
                                    f"Retrying in {retry_delay}s..."
                                )
                                await asyncio.sleep(retry_delay)
                                retry_delay = min(600, retry_delay * 2)  # Cap at 10 minutes
                            else:
                                raise Exception(f"Batch creation failed after {max_retries} attempts: {error}")

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Network error during Anthropic batch creation (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(600, retry_delay * 2)
                else:
                    raise Exception(f"Batch creation failed after {max_retries} attempts: {e}")

        if not provider_batch_id:
            raise Exception("Anthropic batch creation failed: no provider_batch_id returned")

        logger.info(
            f"Submitted Anthropic batch {submission.batch_id} "
            f"(provider ID: {provider_batch_id}, {len(requests)} requests)"
        )

        # Start polling
        submission.provider_batch_id = provider_batch_id
        submission.status = "submitted"
        submission.submitted_at = datetime.utcnow()
        self.active_batches[submission.batch_id] = submission
        self._save_state()  # Persist state

        task = asyncio.create_task(
            self._poll_anthropic_batch(submission.batch_id, provider_batch_id)
        )
        self.poll_tasks[submission.batch_id] = task

        return provider_batch_id

    async def _poll_openai_batch(self, batch_id: str, provider_batch_id: str):
        """Poll OpenAI batch until completion."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                await asyncio.sleep(self.poll_interval)

                # Get batch status with retry logic
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.openai.com/v1/batches/{provider_batch_id}",
                        headers={"Authorization": f"Bearer {self.openai_key}"},
                    ) as resp:
                        if resp.status != 200:
                            logger.error(f"Failed to get batch status: {await resp.text()}")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                logger.error(f"Batch {batch_id}: too many consecutive errors, giving up")
                                break
                            continue

                        batch_data = await resp.json()
                        status = batch_data["status"]
                        consecutive_errors = 0  # Reset on success

                logger.debug(f"OpenAI batch {batch_id} status: {status}")

                # Update submission
                submission = self.active_batches[batch_id]
                submission.status = status
                self._save_state()  # Persist status change

                # Check if complete
                if status in ["completed", "failed", "expired", "cancelled"]:
                    if status == "completed":
                        await self._download_openai_results(batch_id, batch_data)
                    else:
                        logger.warning(f"Batch {batch_id} finished with status: {status}")

                    # Don't remove from active_batches yet - let proxy_server handle cleanup
                    # after it processes the results
                    break

            except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError) as e:
                # Network/DNS errors - retry with exponential backoff
                consecutive_errors += 1
                backoff = min(300, self.poll_interval * (2 ** (consecutive_errors - 1)))
                logger.warning(
                    f"Network error polling batch {batch_id} (attempt {consecutive_errors}/{max_consecutive_errors}): {e}. "
                    f"Retrying in {backoff:.0f}s..."
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Batch {batch_id}: too many network errors, giving up")
                    break
                await asyncio.sleep(backoff)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error polling batch {batch_id} (attempt {consecutive_errors}/{max_consecutive_errors}): {e}", exc_info=True)
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Batch {batch_id}: too many errors, giving up")
                    break
                # Exponential backoff for other errors too
                backoff = min(300, self.poll_interval * (2 ** (consecutive_errors - 1)))
                await asyncio.sleep(backoff)

    async def _poll_anthropic_batch(self, batch_id: str, provider_batch_id: str):
        """Poll Anthropic batch until completion."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                await asyncio.sleep(self.poll_interval)

                # Get batch status with retry logic
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.anthropic.com/v1/messages/batches/{provider_batch_id}",
                        headers={
                            "x-api-key": self.anthropic_key,
                            "anthropic-version": "2023-06-01",
                        },
                    ) as resp:
                        if resp.status != 200:
                            logger.error(f"Failed to get batch status: {await resp.text()}")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                logger.error(f"Batch {batch_id}: too many consecutive errors, giving up")
                                break
                            continue

                        batch_data = await resp.json()
                        status = batch_data["processing_status"]
                        consecutive_errors = 0  # Reset on success

                logger.debug(f"Anthropic batch {batch_id} status: {status}")

                # Update submission
                submission = self.active_batches[batch_id]
                submission.status = status
                self._save_state()  # Persist status change

                # Check if complete
                if status == "ended":
                    await self._download_anthropic_results(batch_id, provider_batch_id)

                    # Don't remove from active_batches yet - let proxy_server handle cleanup
                    # after it processes the results
                    break

            except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError) as e:
                # Network/DNS errors - retry with exponential backoff
                consecutive_errors += 1
                backoff = min(300, self.poll_interval * (2 ** (consecutive_errors - 1)))
                logger.warning(
                    f"Network error polling batch {batch_id} (attempt {consecutive_errors}/{max_consecutive_errors}): {e}. "
                    f"Retrying in {backoff:.0f}s..."
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Batch {batch_id}: too many network errors, giving up")
                    break
                await asyncio.sleep(backoff)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error polling batch {batch_id} (attempt {consecutive_errors}/{max_consecutive_errors}): {e}", exc_info=True)
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Batch {batch_id}: too many errors, giving up")
                    break
                # Exponential backoff for other errors too
                backoff = min(300, self.poll_interval * (2 ** (consecutive_errors - 1)))
                await asyncio.sleep(backoff)

    async def _download_openai_results(self, batch_id: str, batch_data: Dict):
        """Download and process OpenAI batch results."""
        output_file_id = batch_data.get("output_file_id")

        if not output_file_id:
            logger.error(f"No output file for batch {batch_id}")
            return

        output_file = Path(f"/tmp/openai_batch_{batch_id}_results.jsonl")

        # Download results with timeout and chunked reading
        # Large batch results can be several MB - stream to file to avoid truncation
        timeout = aiohttp.ClientTimeout(total=300, sock_read=60)  # 5 min total, 60s per read
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"https://api.openai.com/v1/files/{output_file_id}/content",
                    headers={"Authorization": f"Bearer {self.openai_key}"},
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to download results: {await resp.text()}")
                        return

                    # Stream to file in chunks to handle large responses
                    bytes_written = 0
                    with open(output_file, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                            bytes_written += len(chunk)

                    logger.info(f"Downloaded {bytes_written} bytes for batch {batch_id}")

            # Validate JSONL completeness - check last line is valid JSON
            truncated = False
            with open(output_file, 'r') as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line
                if last_line:
                    try:
                        import json
                        json.loads(last_line)
                    except json.JSONDecodeError as e:
                        logger.error(f"Batch {batch_id}: Last line is truncated/invalid JSON: {last_line[:100]}...")
                        truncated = True

            if truncated:
                logger.warning(f"Batch {batch_id} results may be incomplete - proceeding with partial data")

        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading results for batch {batch_id}")
            return
        except Exception as e:
            logger.error(f"Error downloading results for batch {batch_id}: {e}", exc_info=True)
            if output_file.exists():
                output_file.unlink()
            return

        submission = self.active_batches.get(batch_id)
        if submission:
            submission.output_file_path = str(output_file)
            submission.completed_at = datetime.utcnow()
            self._save_state()  # Persist output_file_path immediately

        logger.info(f"Downloaded results for batch {batch_id} to {output_file}")

    async def _download_anthropic_results(self, batch_id: str, provider_batch_id: str):
        """Download and process Anthropic batch results."""
        output_file = Path(f"/tmp/anthropic_batch_{batch_id}_results.jsonl")

        # Download results with timeout and chunked reading
        timeout = aiohttp.ClientTimeout(total=300, sock_read=60)  # 5 min total, 60s per read
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"https://api.anthropic.com/v1/messages/batches/{provider_batch_id}/results",
                    headers={
                        "x-api-key": self.anthropic_key,
                        "anthropic-version": "2023-06-01",
                    },
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to download results: {await resp.text()}")
                        return

                    # Stream to file in chunks to handle large responses
                    bytes_written = 0
                    with open(output_file, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                            bytes_written += len(chunk)

                    logger.info(f"Downloaded {bytes_written} bytes for batch {batch_id}")

            # Validate JSONL completeness - check last line is valid JSON
            truncated = False
            with open(output_file, 'r') as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line
                if last_line:
                    try:
                        import json
                        json.loads(last_line)
                    except json.JSONDecodeError as e:
                        logger.error(f"Batch {batch_id}: Last line is truncated/invalid JSON: {last_line[:100]}...")
                        truncated = True

            if truncated:
                logger.warning(f"Batch {batch_id} results may be incomplete - proceeding with partial data")

        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading results for batch {batch_id}")
            return
        except Exception as e:
            logger.error(f"Error downloading results for batch {batch_id}: {e}", exc_info=True)
            if output_file.exists():
                output_file.unlink()
            return

        submission = self.active_batches.get(batch_id)
        if submission:
            submission.output_file_path = str(output_file)
            submission.completed_at = datetime.utcnow()
            self._save_state()  # Persist output_file_path immediately

        logger.info(f"Downloaded results for batch {batch_id} to {output_file}")

    async def start(self):
        """Start batch handler and resume polling for in-progress batches."""
        # Clean up orphaned completed batches (completed but never downloaded/processed)
        orphaned_batches = []
        for batch_id, submission in list(self.active_batches.items()):
            if submission.status in ["completed", "failed", "expired", "cancelled", "ended"]:
                if not submission.output_file_path:
                    # Batch completed but results were never downloaded - orphaned
                    orphaned_batches.append(batch_id)
                    logger.warning(
                        f"Removing orphaned batch {batch_id} (status: {submission.status}, "
                        f"no output file, created: {submission.created_at})"
                    )

        for batch_id in orphaned_batches:
            self.active_batches.pop(batch_id, None)

        if orphaned_batches:
            self._save_state()
            logger.info(f"Cleaned up {len(orphaned_batches)} orphaned batches")

        # Resume polling for batches that were in progress
        for batch_id, submission in list(self.active_batches.items()):
            if submission.status not in ["completed", "failed", "expired", "cancelled", "ended"]:
                if submission.provider_batch_id:
                    logger.info(f"Resuming polling for batch {batch_id} (provider: {submission.provider.value}, status: {submission.status})")
                    if submission.provider == LLMProvider.OPENAI:
                        task = asyncio.create_task(
                            self._poll_openai_batch(batch_id, submission.provider_batch_id)
                        )
                        self.poll_tasks[batch_id] = task
                    elif submission.provider == LLMProvider.ANTHROPIC:
                        task = asyncio.create_task(
                            self._poll_anthropic_batch(batch_id, submission.provider_batch_id)
                        )
                        self.poll_tasks[batch_id] = task

    async def stop(self):
        """Stop all polling tasks and save state."""
        logger.info(f"Stopping batch handler, saving state for {len(self.active_batches)} batches")
        self._save_state()

        for task in self.poll_tasks.values():
            task.cancel()

        await asyncio.gather(*self.poll_tasks.values(), return_exceptions=True)
        self.poll_tasks.clear()

    def get_stats(self) -> Dict:
        """Get batch handler statistics."""
        return {
            "active_batches": len(self.active_batches),
            "batches": {
                batch_id: {
                    "provider": sub.provider.value,
                    "status": sub.status,
                    "total_requests": sub.total_requests,
                    "age_seconds": (
                        (datetime.utcnow() - sub.created_at).total_seconds()
                        if sub.created_at
                        else 0
                    ),
                }
                for batch_id, sub in self.active_batches.items()
            },
        }

    def _load_state(self):
        """Load persisted batch state from file."""
        if not self.state_file.exists():
            logger.info("No previous batch state found")
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            # Restore active batches
            for batch_id, batch_dict in data.get("active_batches", {}).items():
                submission = BatchSubmission(**batch_dict)
                self.active_batches[batch_id] = submission

            logger.info(f"Restored {len(self.active_batches)} active batches from state file")

            # Resume polling for in-progress batches
            for batch_id, submission in self.active_batches.items():
                if submission.status not in ["completed", "failed", "expired", "cancelled", "ended"]:
                    if submission.provider_batch_id:
                        logger.info(f"Resuming polling for batch {batch_id} (status: {submission.status})")
                        # Polling will be started by the start() method

        except Exception as e:
            logger.error(f"Failed to load batch state: {e}", exc_info=True)

    def _save_state(self):
        """Persist batch state to file."""
        try:
            data = {
                "active_batches": {
                    batch_id: submission.dict()
                    for batch_id, submission in self.active_batches.items()
                },
                "saved_at": datetime.utcnow().isoformat(),
            }

            # Write atomically
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            temp_file.replace(self.state_file)
            logger.debug(f"Saved state for {len(self.active_batches)} batches")

        except Exception as e:
            logger.error(f"Failed to save batch state: {e}", exc_info=True)

    def purge_batches(
        self,
        batch_ids: Optional[List[str]] = None,
        older_than_seconds: Optional[int] = None,
        status_filter: Optional[List[str]] = None,
        delete_files: bool = True,
    ) -> Dict[str, Any]:
        """
        Purge batches from state.

        Args:
            batch_ids: Specific batch IDs to purge. If None, uses filters.
            older_than_seconds: Purge batches older than this (based on created_at).
            status_filter: Only purge batches with these statuses.
                          Default for age-based: ["completed", "failed", "expired", "cancelled", "ended"]
            delete_files: Also delete associated temp files (input/output JSONL).

        Returns:
            Dict with purged count, batch IDs, and any errors.
        """
        purged = []
        errors = []
        files_deleted = []

        now = datetime.utcnow()

        # Default status filter for age-based purge (don't purge in-progress batches by age)
        if older_than_seconds is not None and status_filter is None:
            status_filter = ["completed", "failed", "expired", "cancelled", "ended"]

        for batch_id, submission in list(self.active_batches.items()):
            should_purge = False

            # Check if specific batch ID requested
            if batch_ids is not None:
                should_purge = batch_id in batch_ids
            else:
                # Age filter
                if older_than_seconds is not None and submission.created_at:
                    age = (now - submission.created_at).total_seconds()
                    if age > older_than_seconds:
                        should_purge = True

                # Status filter
                if should_purge and status_filter:
                    should_purge = submission.status in status_filter

            if should_purge:
                # Cancel polling task if running
                if batch_id in self.poll_tasks:
                    self.poll_tasks[batch_id].cancel()
                    self.poll_tasks.pop(batch_id, None)

                # Delete associated files
                if delete_files:
                    for file_path in [submission.input_file_path, submission.output_file_path]:
                        if file_path:
                            try:
                                path = Path(file_path)
                                if path.exists():
                                    path.unlink()
                                    files_deleted.append(file_path)
                            except Exception as e:
                                errors.append(f"Failed to delete {file_path}: {e}")

                # Remove from active batches
                self.active_batches.pop(batch_id, None)
                purged.append(batch_id)
                logger.info(f"Purged batch {batch_id} (status: {submission.status}, created: {submission.created_at})")

        if purged:
            self._save_state()

        return {
            "purged_count": len(purged),
            "purged_batch_ids": purged,
            "files_deleted": files_deleted,
            "errors": errors,
        }
