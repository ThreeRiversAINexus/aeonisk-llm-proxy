#!/usr/bin/env python3
"""
Batch Orchestrator for Concurrent Narrative Generation

Runs N narrative generation tasks concurrently, with all LLM calls
routed through the batching proxy for cost optimization.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.narrative_generator import NarrativeGenerator
from src.utils.ai_client import UnifiedAIClient


# Output mode enum
OUTPUT_MODES = ['narrative', 'simplified', 'radio-play', 'catalog', 'all']


async def generate_single_narrative(
    session_path: Path,
    use_proxy: bool = True,
    proxy_mode: str = 'auto',
    output_dir: Path = Path("./output"),
    output_mode: str = 'narrative',
) -> dict:
    """
    Generate narrative for a single session.

    Args:
        session_path: Path to session JSONL file
        use_proxy: Use LLM batching proxy
        proxy_mode: Routing strategy ('batch', 'direct', 'auto')
        output_dir: Base output directory
        output_mode: Output mode ('narrative', 'simplified', 'radio-play', 'all')

    Returns:
        Result dict with session_path, success, and optional error
    """
    session_name = session_path.stem
    start_time = time.time()

    try:
        # Parse session to get metadata
        from src.utils.jsonl_parser import SessionParser
        parser = SessionParser(session_path)
        timeline = parser.get_timeline()

        total_rounds = len(timeline['rounds'])
        print(f"[{session_name}] Starting narrative generation ({total_rounds} rounds)...", flush=True)

        # Set proxy environment variables
        if use_proxy:
            os.environ['USE_LLM_PROXY'] = 'true'
            os.environ['LLM_PROXY_MODE'] = proxy_mode
            print(f"[{session_name}]   → Proxy mode: {proxy_mode.upper()}", flush=True)

        # Create generator with synthesis style (only round syntheses)
        print(f"[{session_name}]   → Creating narrative generator (synthesis style)", flush=True)
        generator = NarrativeGenerator(
            style='synthesis',
            use_ai_enhancement=True
        )

        # Generate script
        print(f"[{session_name}]   → Generating script...", flush=True)
        script = generator.generate_script(session_path)

        # Save script
        script_path = output_dir / session_name / "script.json"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)

        segments = len(script.get('segments', []))
        env_prompts = len(script.get('environment_prompts', []))
        print(f"[{session_name}]   → Script: {segments} segments, {env_prompts} environment prompts", flush=True)

        # Extract narrative text from script segments
        print(f"[{session_name}]   → Extracting narrative prose...", flush=True)
        narrative_parts = []

        # Opening scene (if exists)
        if 'opening' in script and script['opening'].get('text'):
            narrative_parts.append(script['opening']['text'])

        # Round narratives
        for segment in script.get('segments', []):
            if segment.get('text'):
                narrative_parts.append(segment['text'])

        narrative = "\n\n".join(narrative_parts)

        # Save narrative
        narrative_path = output_dir / session_name / "narrative.md"
        with open(narrative_path, 'w') as f:
            f.write(narrative)

        output_files = {
            "script_path": str(script_path),
            "narrative_path": str(narrative_path),
        }

        # Generate simplified narrative if requested
        if output_mode in ['simplified', 'all']:
            print(f"[{session_name}]   → Generating simplified narrative...", flush=True)
            from src.narration.simplified_editor import SimplifiedEditor

            editor = SimplifiedEditor()
            simplified_text, stats = editor.edit_narrative(narrative, session_path, use_ai=True)

            simplified_path = output_dir / session_name / "narrative_simplified.md"
            with open(simplified_path, 'w') as f:
                f.write(simplified_text)

            # Save stats
            stats_path = output_dir / session_name / "simplified_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)

            output_files["simplified_path"] = str(simplified_path)
            output_files["simplified_stats_path"] = str(stats_path)
            print(f"[{session_name}]   → Simplified: {stats.reduction_percentage:.1f}% reduction", flush=True)

        # Generate radio play if requested
        if output_mode in ['radio-play', 'all']:
            print(f"[{session_name}]   → Generating radio play...", flush=True)
            from src.narration.radio_play_generator import RadioPlayGenerator
            from src.narration.radio_play_formatter import format_radio_play_script

            rp_generator = RadioPlayGenerator(narrator_usage='minimal')
            radio_play_json = rp_generator.generate_radio_play(session_path)

            # Save JSON
            rp_json_path = output_dir / session_name / "radio_play.json"
            with open(rp_json_path, 'w') as f:
                json.dump(radio_play_json, f, indent=2)

            # Format script
            script_text, guide_text = format_radio_play_script(radio_play_json)

            rp_script_path = output_dir / session_name / "radio_play.txt"
            with open(rp_script_path, 'w') as f:
                f.write(script_text)

            rp_guide_path = output_dir / session_name / "radio_play_guide.md"
            with open(rp_guide_path, 'w') as f:
                f.write(guide_text)

            output_files["radio_play_json_path"] = str(rp_json_path)
            output_files["radio_play_script_path"] = str(rp_script_path)
            output_files["radio_play_guide_path"] = str(rp_guide_path)

            scenes_count = len(radio_play_json.get('scenes', []))
            print(f"[{session_name}]   → Radio play: {scenes_count} scenes", flush=True)

        # Unset proxy flags
        if use_proxy:
            os.environ.pop('USE_LLM_PROXY', None)
            os.environ.pop('LLM_PROXY_MODE', None)

        elapsed = time.time() - start_time
        print(f"[{session_name}] ✓ Complete ({elapsed:.1f}s)", flush=True)
        for key, path in output_files.items():
            print(f"[{session_name}]   → {key}: {path}", flush=True)

        return {
            "session_path": str(session_path),
            "session_name": session_name,
            "success": True,
            "elapsed_seconds": elapsed,
            "segments": segments,
            "output_mode": output_mode,
            **output_files,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{session_name}] ✗ Failed ({elapsed:.1f}s): {e}", flush=True)

        return {
            "session_path": str(session_path),
            "session_name": session_name,
            "success": False,
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def generate_single_catalog(
    session_path: Path,
    batch_id: str,
    use_proxy: bool = True,
    proxy_mode: str = 'auto',
    output_dir: Path = Path("./output"),
    force: bool = False,
) -> dict:
    """
    Generate catalog entry for a single session within a batch.

    Uses the batch catalog system: each session gets a synopsis in
    output/by_project/<batch_id>/<session_id>/synopsis.md

    Args:
        session_path: Path to session JSONL file
        batch_id: Batch identifier (timestamp) for grouping sessions
        use_proxy: Use LLM batching proxy
        proxy_mode: Routing strategy ('batch', 'direct', 'auto')
        output_dir: Base output directory
        force: Force regeneration even if cache is valid

    Returns:
        Result dict with session_path, success, and optional error
    """
    from src.narration.catalog_generator import CatalogGenerator, BatchCatalogManager
    from src.utils.jsonl_parser import SessionParser
    from src.utils.project_paths import BatchCatalogPaths

    session_name = session_path.stem
    session_id = session_name.replace('session_', '')
    start_time = time.time()

    try:
        # Get batch paths
        batch_paths = BatchCatalogPaths(output_dir, batch_id)
        catalog_manager = BatchCatalogManager(batch_paths.batch_root, batch_id)

        # Check compatibility first
        parser = SessionParser(session_path)
        compatible, reason = CatalogGenerator.is_catalog_compatible(parser)
        if not compatible:
            print(f"[{session_name}] \u2298 Skipped ({reason})", flush=True)
            return {
                "session_path": str(session_path),
                "session_name": session_name,
                "session_id": session_id,
                "batch_id": batch_id,
                "success": True,
                "skipped": True,
                "reason": reason,
                "elapsed_seconds": time.time() - start_time,
            }

        # Compute checksum for cache check
        checksum = CatalogGenerator.compute_checksum(session_path)

        # Check cache validity (skip if unchanged and not forced)
        if not force and catalog_manager.is_session_cached(session_id, checksum):
            print(f"[{session_name}] \u2298 Skipped (unchanged)", flush=True)
            return {
                "session_path": str(session_path),
                "session_name": session_name,
                "session_id": session_id,
                "batch_id": batch_id,
                "success": True,
                "skipped": True,
                "reason": "cache_valid",
                "elapsed_seconds": time.time() - start_time,
            }

        print(f"[{session_name}] Starting synopsis generation...", flush=True)
        print(f"[{session_name}]   \u2192 Batch: {batch_id}", flush=True)

        # Set proxy environment variables
        if use_proxy:
            os.environ['USE_LLM_PROXY'] = 'true'
            os.environ['LLM_PROXY_MODE'] = proxy_mode

        # Generate catalog
        generator = CatalogGenerator()
        catalog = generator.generate_catalog(session_path)

        # Add to batch catalog (saves synopsis.json/md + updates catalog.json/md)
        catalog_manager.add_session(catalog)

        # Unset proxy flags
        if use_proxy:
            os.environ.pop('USE_LLM_PROXY', None)
            os.environ.pop('LLM_PROXY_MODE', None)

        elapsed = time.time() - start_time
        print(f"[{session_name}] \u2713 Complete ({elapsed:.1f}s)", flush=True)
        print(f"[{session_name}]   \u2192 Title: {catalog.title}", flush=True)
        print(f"[{session_name}]   \u2192 Characters: {len(catalog.characters)}", flush=True)
        print(f"[{session_name}]   \u2192 Synopsis: {batch_paths.get_synopsis_md(session_id)}", flush=True)

        return {
            "session_path": str(session_path),
            "session_name": session_name,
            "session_id": session_id,
            "batch_id": batch_id,
            "success": True,
            "skipped": False,
            "elapsed_seconds": elapsed,
            "title": catalog.title,
            "characters": len(catalog.characters),
            "synopsis_path": str(batch_paths.get_synopsis_md(session_id)),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{session_name}] \u2717 Failed ({elapsed:.1f}s): {e}", flush=True)
        import traceback
        traceback.print_exc()

        return {
            "session_path": str(session_path),
            "session_name": session_name,
            "session_id": session_id,
            "batch_id": batch_id,
            "success": False,
            "skipped": False,
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


async def batch_generate(
    session_paths: List[Path],
    max_workers: int = 10,
    use_proxy: bool = True,
    proxy_mode: str = 'auto',
    output_dir: Path = Path("./output"),
    output_mode: str = 'narrative',
    force: bool = False,
) -> List[dict]:
    """
    Generate narratives for N sessions concurrently.

    Args:
        session_paths: List of session JSONL paths
        max_workers: Maximum concurrent tasks
        use_proxy: Use LLM batching proxy
        proxy_mode: Routing strategy ('batch', 'direct', 'auto')
        output_dir: Base output directory
        output_mode: Output mode ('narrative', 'simplified', 'radio-play', 'catalog', 'all')
        force: Force regeneration even if cache is valid (catalog mode)

    Returns:
        List of result dicts
    """
    from datetime import datetime

    # Determine job type based on mode
    job_type = "CATALOG GENERATION" if output_mode == 'catalog' else "NARRATIVE GENERATION"

    # Generate batch_id for catalog mode (shared across all sessions)
    batch_id = None
    if output_mode == 'catalog':
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Pre-filter sessions for catalog mode to maximize worker efficiency
    skipped_results = []
    if output_mode == 'catalog':
        from src.narration.catalog_generator import CatalogGenerator
        from src.utils.jsonl_parser import SessionParser

        compatible_paths = []
        print(f"Pre-filtering {len(session_paths)} sessions for compatibility...", flush=True)

        for session_path in session_paths:
            session_name = session_path.stem
            session_id = session_name.replace('session_', '')

            # Check compatibility (no cache check - new batch each time)
            parser = SessionParser(session_path)
            compatible, reason = CatalogGenerator.is_catalog_compatible(parser)
            if not compatible:
                skipped_results.append({
                    "session_path": str(session_path),
                    "session_name": session_name,
                    "session_id": session_id,
                    "batch_id": batch_id,
                    "success": True,
                    "skipped": True,
                    "reason": reason,
                    "elapsed_seconds": 0,
                })
                continue

            compatible_paths.append(session_path)

        print(f"  \u2192 {len(compatible_paths)} compatible, {len(skipped_results)} skipped", flush=True)
        session_paths = compatible_paths

    print(f"\n{'='*60}", flush=True)
    print(f"BATCH {job_type}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Sessions to process: {len(session_paths)}", flush=True)
    if skipped_results:
        print(f"Sessions pre-skipped: {len(skipped_results)}", flush=True)
    print(f"Max workers: {max_workers}", flush=True)
    print(f"Output mode: {output_mode.upper()}", flush=True)
    print(f"Proxy mode: {'ENABLED' if use_proxy else 'DISABLED'}", flush=True)
    if use_proxy:
        print(f"Routing strategy: {proxy_mode.upper()}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"\nSessions to process (newest to oldest):", flush=True)
    for i, path in enumerate(session_paths[:10], 1):
        print(f"  {i}. {path.stem}", flush=True)
    if len(session_paths) > 10:
        print(f"  ... and {len(session_paths) - 10} more", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Print force flag and batch info for catalog mode
    if output_mode == 'catalog':
        print(f"Batch ID: {batch_id}", flush=True)
        print(f"Output: {output_dir}/by_project/{batch_id}/", flush=True)

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)

    async def bounded_generate(path):
        async with semaphore:
            # Run in thread pool since LLM calls are sync
            if output_mode == 'catalog':
                return await asyncio.to_thread(
                    generate_single_catalog, path, batch_id, use_proxy, proxy_mode, output_dir, force
                )
            else:
                return await asyncio.to_thread(
                    lambda: asyncio.run(
                        generate_single_narrative(path, use_proxy, proxy_mode, output_dir, output_mode)
                    )
                )

    # Create tasks
    tasks = [bounded_generate(path) for path in session_paths]

    # Wait for all to complete
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "session_path": str(session_paths[i]),
                "session_name": session_paths[i].stem,
                "success": False,
                "error": str(result),
            })
        else:
            final_results.append(result)

    # Add pre-skipped results (from catalog mode pre-filtering)
    final_results.extend(skipped_results)

    # Print summary
    total_time = time.time() - start_time
    successful = sum(1 for r in final_results if r.get("success"))
    failed = len(final_results) - successful

    # Calculate statistics based on mode
    avg_time = sum(r.get("elapsed_seconds", 0) for r in final_results if r.get("success")) / max(successful, 1)

    print(f"\n{'='*60}", flush=True)
    print(f"BATCH COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total time: {total_time:.1f}s", flush=True)
    print(f"Sessions: {successful}/{len(final_results)} successful", flush=True)

    if failed > 0:
        print(f"Failed: {failed}", flush=True)
        print(f"\nFailed sessions:", flush=True)
        for r in final_results:
            if not r.get("success"):
                print(f"  - {r['session_name']}: {r.get('error', 'Unknown error')}", flush=True)

    print(f"\nGeneration stats:", flush=True)

    if output_mode == 'catalog':
        # Catalog-specific stats
        skipped = sum(1 for r in final_results if r.get("skipped", False))
        generated = sum(1 for r in final_results if r.get("success") and not r.get("skipped", False))
        total_characters = sum(r.get("characters", 0) for r in final_results if r.get("success") and not r.get("skipped"))
        print(f"  Catalogs generated: {generated}", flush=True)
        print(f"  Catalogs skipped (cached): {skipped}", flush=True)
        print(f"  Total characters cataloged: {total_characters}", flush=True)
    else:
        # Narrative-specific stats
        total_segments = sum(r.get("segments", 0) for r in final_results if r.get("success"))
        print(f"  Total segments: {total_segments}", flush=True)
        print(f"  Scripts generated: {successful}", flush=True)
        print(f"  Narratives generated: {successful}", flush=True)

    print(f"  Avg time per session: {avg_time:.1f}s", flush=True)
    print(f"  Throughput: {len(final_results) / (total_time / 60):.1f} sessions/min", flush=True)
    print(f"{'='*60}\n", flush=True)

    return final_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch narrative generation orchestrator")
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default="archive/batches",
        help="Directory containing session JSONL files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern for session files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default="http://localhost:8000",
        help="LLM proxy URL (or 'direct' to skip proxy)"
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Bypass LLM proxy entirely and call APIs directly"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["batch", "direct", "auto"],
        help="Routing mode: 'batch' (always batch), 'direct' (real-time), 'auto' (smart routing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Base output directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of sessions to process"
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="narrative",
        choices=OUTPUT_MODES,
        help="Output mode: 'narrative' (default), 'simplified', 'radio-play', 'catalog', or 'all'"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache is valid (catalog mode)"
    )

    args = parser.parse_args()

    # Find sessions
    sessions_dir = Path(args.sessions_dir)
    session_paths = list(sessions_dir.glob(args.pattern))

    if not session_paths:
        print(f"No sessions found matching {args.pattern} in {sessions_dir}")
        return 1

    # Sort by modification time (newest to oldest)
    session_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Limit if requested
    if args.limit:
        session_paths = session_paths[:args.limit]

    # Check proxy mode
    use_proxy = not args.no_proxy and args.proxy != "direct"
    if use_proxy:
        os.environ['LLM_PROXY_URL'] = args.proxy

    # Run batch generation
    results = asyncio.run(
        batch_generate(
            session_paths,
            max_workers=args.workers,
            use_proxy=use_proxy,
            proxy_mode=args.mode,
            output_dir=Path(args.output),
            output_mode=args.output_mode,
            force=args.force,
        )
    )

    # Exit with error if any failed
    failed = sum(1 for r in results if not r.get("success"))
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
