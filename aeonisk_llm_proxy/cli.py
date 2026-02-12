#!/usr/bin/env python3
"""
CLI for LLM Batching Proxy.

Usage:
    python -m aeonisk_llm_proxy start
    python -m aeonisk_llm_proxy purge --older-than 3600
"""

import click
import json
import logging
from datetime import datetime
from pathlib import Path


@click.group()
def cli():
    """LLM Batching Proxy - Cost optimization through request batching"""
    pass


@cli.command('start')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--batch-size', type=int, help='Flush queue when it reaches this many requests [env: BATCH_THRESHOLD, default: 100]')
@click.option('--flush-interval', type=int, help='Max seconds to wait before flushing the queue [env: BATCH_TIMEOUT, default: 300]')
@click.option('--idle-timeout', type=int, help='Flush queue after this many idle seconds [env: BATCH_MAX_IDLE, default: 3600]')
@click.option('--poll-interval', type=int, help='Poll provider batch API every N seconds [env: BATCH_POLL_INTERVAL, default: 60]')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--log-level', default='info', type=click.Choice(['debug', 'info', 'warning', 'error']))
def start(host, port, batch_size, flush_interval, idle_timeout, poll_interval, reload, log_level):
    """Start the LLM batching proxy server.

    Queue tuning: requests are collected per (provider, model) and flushed to
    the provider batch API when any of these triggers fire:

    \b
      --batch-size      queue length trigger  (default 100)
      --flush-interval  max age trigger in s  (default 300)
      --idle-timeout    no-new-requests flush  (default 3600)

    CLI flags override env vars, which override the built-in defaults.

    Examples:

    \b
      # Start with defaults
      aeonisk-llm-proxy start
      # Flush every 30s or every 20 requests
      aeonisk-llm-proxy start --flush-interval 30 --batch-size 20
      # Custom port, debug logging
      aeonisk-llm-proxy start --port 8080 --log-level debug
    """
    import os
    import uvicorn

    # CLI flags override env vars
    if batch_size is not None:
        os.environ['BATCH_THRESHOLD'] = str(batch_size)
    if flush_interval is not None:
        os.environ['BATCH_TIMEOUT'] = str(flush_interval)
    if idle_timeout is not None:
        os.environ['BATCH_MAX_IDLE'] = str(idle_timeout)
    if poll_interval is not None:
        os.environ['BATCH_POLL_INTERVAL'] = str(poll_interval)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    click.echo(f"Starting LLM proxy server on {host}:{port}...")
    click.echo("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "aeonisk_llm_proxy.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


@cli.command('purge')
@click.option('--older-than', type=int, help='Purge batches older than N seconds')
@click.option('--status', help='Only purge batches with this status (comma-separated: completed,failed,expired)')
@click.option('--batch-id', multiple=True, help='Specific batch ID(s) to purge')
@click.option('--keep-files', is_flag=True, help='Keep temp files (input/output JSONL)')
@click.option('--proxy', default='http://localhost:8000', help='LLM proxy server URL')
@click.option('--local', is_flag=True, help='Purge directly from state file (no running proxy required)')
@click.option('--dry-run', is_flag=True, help='Show what would be purged without actually purging')
def purge(older_than, status, batch_id, keep_files, proxy, local, dry_run):
    """Purge stale batches from the proxy state.

    Examples:

        # Purge completed batches older than 1 hour (via API)
        python -m aeonisk_llm_proxy purge --older-than 3600

        # Purge specific batch IDs
        python -m aeonisk_llm_proxy purge --batch-id abc123 --batch-id def456

        # Purge all failed batches
        python -m aeonisk_llm_proxy purge --status failed

        # Purge directly from state file (proxy doesn't need to be running)
        python -m aeonisk_llm_proxy purge --older-than 3600 --local

        # Dry run to see what would be purged
        python -m aeonisk_llm_proxy purge --older-than 3600 --dry-run
    """
    import requests

    if local:
        _purge_local(older_than, status, batch_id, keep_files, dry_run)
    else:
        _purge_via_api(older_than, status, batch_id, keep_files, proxy, dry_run)


def _purge_local(older_than, status, batch_id, keep_files, dry_run):
    """Purge batches directly from state file (no proxy needed)."""
    state_file = Path('/tmp/llm_proxy_batches.json')

    if not state_file.exists():
        click.echo('No batch state file found at /tmp/llm_proxy_batches.json')
        return

    with open(state_file) as f:
        data = json.load(f)

    batches = data.get('active_batches', {})
    if not batches:
        click.echo('No batches in state file.')
        return

    # Parse status filter
    status_filter = None
    if status:
        status_filter = [s.strip() for s in status.split(',')]
    elif older_than and not batch_id:
        # Default to only completed statuses for age-based purge
        status_filter = ['completed', 'failed', 'expired', 'cancelled', 'ended']

    now = datetime.utcnow()
    to_purge = []

    for bid, batch in batches.items():
        should_purge = False

        if batch_id:
            should_purge = bid in batch_id
        else:
            if older_than:
                created = batch.get('created_at')
                if created:
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace('Z', '+00:00').replace('+00:00', ''))
                    age = (now - created).total_seconds()
                    if age > older_than:
                        should_purge = True

            if should_purge and status_filter:
                should_purge = batch.get('status') in status_filter

        if should_purge:
            to_purge.append((bid, batch))

    if not to_purge:
        click.echo('No batches match the purge criteria.')
        return

    click.echo(f'Found {len(to_purge)} batch(es) to purge:')
    for bid, batch in to_purge:
        age = 'unknown'
        created = batch.get('created_at')
        if created:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace('Z', '+00:00').replace('+00:00', ''))
            age = f'{(now - created).total_seconds():.0f}s'
        click.echo(f'  - {bid} (status: {batch.get("status")}, age: {age})')

    if dry_run:
        click.echo('\n[DRY RUN] No changes made.')
        return

    # Actually purge
    files_deleted = []
    for bid, batch in to_purge:
        if not keep_files:
            for key in ['input_file_path', 'output_file_path']:
                fpath = batch.get(key)
                if fpath:
                    p = Path(fpath)
                    if p.exists():
                        p.unlink()
                        files_deleted.append(fpath)
        del batches[bid]

    # Save updated state
    data['saved_at'] = datetime.utcnow().isoformat()
    with open(state_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    click.echo(f'\nPurged {len(to_purge)} batch(es).')
    if files_deleted:
        click.echo(f'Deleted {len(files_deleted)} temp file(s).')


def _purge_via_api(older_than, status, batch_id, keep_files, proxy, dry_run):
    """Purge batches via API endpoint."""
    import requests

    params = {'delete_files': not keep_files}
    if older_than:
        params['older_than_seconds'] = older_than
    if status:
        params['status'] = status

    if batch_id:
        # Purge specific batch IDs one by one
        for bid in batch_id:
            if dry_run:
                click.echo(f'[DRY RUN] Would delete batch: {bid}')
                continue
            try:
                resp = requests.delete(f'{proxy}/batches/{bid}', params={'delete_files': not keep_files})
                if resp.status_code == 200:
                    click.echo(f'Purged batch {bid}')
                elif resp.status_code == 404:
                    click.echo(f'Batch {bid} not found')
                else:
                    click.echo(f'Error purging {bid}: {resp.text}')
            except requests.exceptions.ConnectionError:
                click.echo(f'Error: Cannot connect to proxy at {proxy}. Use --local to purge directly from state file.')
                return
    else:
        if dry_run:
            click.echo(f'[DRY RUN] Would call POST /batches/purge with params: {params}')
            return
        try:
            resp = requests.post(f'{proxy}/batches/purge', params=params)
            if resp.status_code == 200:
                result = resp.json()
                click.echo(f'Purged {result.get("purged_count", 0)} batch(es)')
                if result.get('purged_batch_ids'):
                    for bid in result['purged_batch_ids']:
                        click.echo(f'  - {bid}')
                if result.get('files_deleted'):
                    click.echo(f'Deleted {len(result["files_deleted"])} temp file(s)')
                if result.get('errors'):
                    click.echo('Errors:')
                    for err in result['errors']:
                        click.echo(f'  - {err}')
            else:
                click.echo(f'Error: {resp.text}')
        except requests.exceptions.ConnectionError:
            click.echo(f'Error: Cannot connect to proxy at {proxy}. Use --local to purge directly from state file.')


@cli.command('status')
@click.option('--proxy', default='http://localhost:8000', help='LLM proxy server URL')
def status(proxy):
    """Show proxy status and batch statistics.

    Examples:

        python -m aeonisk_llm_proxy status
    """
    import requests

    try:
        # Get health
        health_resp = requests.get(f'{proxy}/health')
        if health_resp.status_code != 200:
            click.echo(f'Proxy unhealthy: {health_resp.text}')
            return

        health = health_resp.json()
        click.echo(f'Status: {health.get("status")}')
        click.echo(f'Uptime: {health.get("uptime_seconds", 0):.0f}s')
        click.echo(f'Queue size: {health.get("queue_size", 0)}')
        click.echo(f'Active batches: {health.get("active_batches", 0)}')

        providers = health.get('providers_healthy', {})
        click.echo(f'Providers:')
        for provider, healthy in providers.items():
            status_icon = '\u2713' if healthy else '\u2717'
            click.echo(f'  {status_icon} {provider}')

        # Get batch stats
        batch_resp = requests.get(f'{proxy}/batches/stats')
        if batch_resp.status_code == 200:
            stats = batch_resp.json()
            if stats.get('batches'):
                click.echo(f'\nBatches:')
                for batch_id, batch in stats['batches'].items():
                    age = batch.get('age_seconds', 0)
                    click.echo(f'  - {batch_id}: {batch.get("status")} ({age:.0f}s old)')

    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Cannot connect to proxy at {proxy}')
        click.echo('Is the proxy server running?')


if __name__ == '__main__':
    cli()
