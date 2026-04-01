#!/usr/bin/env python3
"""
Benchmark runner using Claude Code (local `claude` CLI) instead of the API.

Usage:
    python3 scripts/run_benchmark_cc.py --run-id cc_test1 --max-requests 3
    python3 scripts/run_benchmark_cc.py --run-id cc_full --max-requests 50 --model opus
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tempfile
from pathlib import Path

PYTHON = str(Path(__file__).parent.parent.parent / "mvp" / ".venv" / "bin" / "python3")
BASE_DIR = Path(__file__).parent.parent


def build_mcp_config(npz_path):
    """Build MCP config JSON for the retrieval tools server."""
    return {
        "mcpServers": {
            "ads-retrieval": {
                "command": PYTHON,
                "args": [
                    str(BASE_DIR / "tools" / "mcp_server.py"),
                    "--request-npz", str(npz_path),
                ],
            }
        }
    }


def load_user_folder(requests_dir, request_id):
    """Load all MD files from the user's folder."""
    folder = Path(requests_dir) / str(request_id)
    parts = []
    for fname in ["profile.md", "engagement.md", "interest_clusters.md", "context.md"]:
        fpath = folder / fname
        if fpath.exists():
            parts.append(fpath.read_text())
    if not parts:
        # Fallback to old single-file format
        old = folder / "user_context.md"
        if old.exists():
            parts.append(old.read_text())
    return "\n\n---\n\n".join(parts) if parts else f"Request {request_id}"


def build_prompt(request_id, user_context):
    """Build the prompt for Claude Code."""
    return (
        f"You are an ads recommendation orchestrator for request {request_id}.\n\n"
        f"## User Information\n{user_context}\n\n"
        f"## Instructions\n"
        f"Produce a ranked list of ads using these tools. Be EFFICIENT — make at most 5-6 tool calls total.\n\n"
        f"Recommended tool call sequence:\n"
        f"1. engagement_pattern_analyzer — check similarity_gap to assess signal quality\n"
        f"2. forced_retrieval(top_k=150) — ALWAYS call, independent retrieval route\n"
        f"3. IF similarity_gap > 0.01: pselect_main_route(top_k=150)\n"
        f"   IF similarity_gap < 0.01: cluster_explorer(n_clusters=5) for engagement-rate-based ranking\n"
        f"4. prod_model_ranker(top_k=100) — production model signal\n"
        f"5. Merge all results, de-duplicate, output ranked list\n\n"
        f"Do NOT call ads_pool_stats, similar_ads_lookup, or feature_filter unless absolutely needed.\n"
        f"Do NOT call lookup_similar_requests unless you need historical context.\n\n"
        f"Output EXACTLY this JSON format at the end (nothing else after it):\n"
        f'```json\n{{"ranked_ads": [ad_id_1, ad_id_2, ...], "strategy": "brief description"}}\n```\n'
        f"Include at least 150 ad IDs in ranked_ads, ordered by predicted engagement likelihood."
    )


def parse_ranked_ads(text):
    """Extract ranked_ads from Claude's response."""
    # Try JSON code blocks first
    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "ranked_ads" in parsed:
                    return [int(x) for x in parsed["ranked_ads"]], parsed.get("strategy", "")
            except (json.JSONDecodeError, ValueError):
                continue

    # Fallback: find last occurrence of ranked_ads
    try:
        idx = text.rfind('"ranked_ads"')
        if idx >= 0:
            start = text.rfind('{', 0, idx)
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{': depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            parsed = json.loads(text[start:i+1])
                            if "ranked_ads" in parsed:
                                return [int(x) for x in parsed["ranked_ads"]], parsed.get("strategy", "")
                            break
    except (json.JSONDecodeError, ValueError):
        pass

    return [], ""


def run_single_request(npz_path, request_id, user_context, model="sonnet"):
    """Run Claude Code on a single request."""
    # Build MCP config
    mcp_config = build_mcp_config(npz_path)

    # Write MCP config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='mcp_') as f:
        json.dump(mcp_config, f)
        mcp_config_path = f.name

    prompt = build_prompt(request_id, user_context)

    # Build system prompt from CLAUDE.md + architecture.md + skill.md + learnings.md
    system_parts = []
    for fname in ["CLAUDE.md", "architecture.md", "skill.md", "learnings.md"]:
        fpath = BASE_DIR / fname
        if fpath.exists():
            system_parts.append(fpath.read_text())
    system_prompt = "\n\n---\n\n".join(system_parts)

    cmd = [
        "claude",
        "-p", prompt,
        "--model", model,
        "--output-format", "text",
        "--mcp-config", mcp_config_path,
        "--permission-mode", "bypassPermissions",
        "--append-system-prompt", system_prompt,
        "--no-session-persistence",
        "--max-budget-usd", "1.0",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(BASE_DIR),
        )
        response_text = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print(f"    claude exited with code {result.returncode}")
            if stderr:
                print(f"    stderr: {stderr[:500]}")

        ranked_ads, strategy = parse_ranked_ads(response_text)
        return {
            "request_id": request_id,
            "ranked_ads": ranked_ads,
            "strategy": strategy,
            "raw_response_length": len(response_text),
            "raw_response": response_text[:3000],
        }

    except subprocess.TimeoutExpired:
        print(f"    Timeout (300s)")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None
    finally:
        os.unlink(mcp_config_path)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark via Claude Code CLI")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--data-dir", default="data", help="Directory with npz files")
    parser.add_argument("--requests-dir", default="requests", help="User context directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--request-ids", nargs="+", type=int, help="Specific request IDs to run")
    parser.add_argument("--model", default="sonnet", help="Model: sonnet, opus, haiku")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    requests_dir = Path(args.requests_dir)
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.request_ids:
        npz_files = [data_dir / f"request_{rid}.npz" for rid in args.request_ids]
        npz_files = [f for f in npz_files if f.exists()]
    else:
        npz_files = sorted(data_dir.glob("request_*.npz"))[:args.max_requests]
    print(f"Running benchmark '{args.run_id}' on {len(npz_files)} requests with model {args.model}")

    results = []
    for i, npz_path in enumerate(npz_files):
        # Extract request_id from filename
        import numpy as np
        data = np.load(npz_path)
        request_id = int(data["request_id"])

        # Load user context (all MD files from user folder)
        user_context = load_user_folder(str(requests_dir), request_id)

        print(f"\n[{i+1}/{len(npz_files)}] Request {request_id}...")
        start = time.time()

        result = run_single_request(npz_path, request_id, user_context, args.model)
        elapsed = time.time() - start

        if result:
            result["elapsed_seconds"] = round(elapsed, 1)
            n_ranked = len(result.get("ranked_ads", []))
            print(f"  Done: {n_ranked} ranked ads, {elapsed:.1f}s")

            out_path = run_dir / f"{request_id}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            results.append(result)
        else:
            print(f"  FAILED")

    # Summary
    summary = {
        "run_id": args.run_id,
        "model": args.model,
        "n_requests": len(results),
        "avg_ranked_ads": sum(len(r.get("ranked_ads", [])) for r in results) / max(len(results), 1),
        "avg_elapsed": sum(r["elapsed_seconds"] for r in results) / max(len(results), 1),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Run '{args.run_id}' complete: {len(results)}/{len(npz_files)} requests")
    print(f"Avg ranked ads: {summary['avg_ranked_ads']:.0f}")
    print(f"Results saved to {run_dir}/")
    print(f"\nEvaluate with:")
    print(f"  python3 evaluation/evaluate.py --run-id {args.run_id} --baseline evaluation/results/baseline.json")


if __name__ == "__main__":
    main()
