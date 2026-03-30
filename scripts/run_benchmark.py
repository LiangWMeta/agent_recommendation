#!/usr/bin/env python3
"""
Benchmark runner: loads data, calls Claude API with tool_use, evaluates results.

Usage:
    python3 scripts/run_benchmark.py --run-id test1 --max-requests 5
    python3 scripts/run_benchmark.py --run-id full --max-requests 50
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent to path for tools import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip3 install anthropic")
    sys.exit(1)

from tools.tool_registry import TOOLS, execute_tool


def load_request(npz_path):
    """Load a request's data from npz file."""
    data = np.load(npz_path)
    return {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
        "labels": data["labels"],
    }


def load_context_files():
    """Load the orchestration context files."""
    base = Path(__file__).parent.parent
    context_parts = []

    for fname in ["architecture.md", "skill.md"]:
        fpath = base / fname
        if fpath.exists():
            context_parts.append(fpath.read_text())

    return "\n\n---\n\n".join(context_parts)


def load_user_context(request_id, requests_dir):
    """Load user context markdown for a specific request."""
    ctx_path = Path(requests_dir) / str(request_id) / "user_context.md"
    if ctx_path.exists():
        return ctx_path.read_text()
    return f"No user context available for request {request_id}."


def build_system_prompt(context, user_context):
    """Build the system prompt for Claude."""
    base = Path(__file__).parent.parent
    claude_md = base / "CLAUDE.md"
    system = claude_md.read_text() if claude_md.exists() else ""
    system += f"\n\n---\n\n{context}\n\n---\n\n{user_context}"
    return system


def run_single_request(client, request_data, context, requests_dir, model="claude-sonnet-4-20250514"):
    """Run Claude on a single request with tool_use."""
    request_id = request_data["request_id"]
    user_context = load_user_context(request_id, requests_dir)
    system_prompt = build_system_prompt(context, user_context)

    n_pos = int((request_data["labels"] == 1).sum())
    n_neg = int((request_data["labels"] == 0).sum())

    user_message = (
        f"Please recommend ads for this user (request {request_id}). "
        f"There are {n_pos + n_neg} candidate ads in the pool. "
        f"Use the available tools to analyze the pool, understand the user's preferences, "
        f"and produce a ranked list of at least 100 ad IDs.\n\n"
        f"Start by calling ads_pool_stats and engagement_pattern_analyzer to understand the request, "
        f"then use retrieval tools to build your candidate set."
    )

    messages = [{"role": "user", "content": user_message}]
    tool_calls_log = []
    total_input_tokens = 0
    total_output_tokens = 0

    # Agentic loop: keep calling Claude until it produces final output
    max_iterations = 15
    for iteration in range(max_iterations):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"    API error: {e}")
            return None

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Process response content
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check if we need to handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    tool_id = block.id

                    # Execute the tool
                    try:
                        result = execute_tool(tool_name, tool_args, request_data)
                        result_str = json.dumps(result, default=str)
                        # Truncate very long results
                        if len(result_str) > 10000:
                            result_str = result_str[:10000] + "... (truncated)"
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})

                    tool_calls_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result_length": len(result_str),
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Extract final text response
            text_content = ""
            for block in assistant_content:
                if hasattr(block, "text"):
                    text_content += block.text

            # Parse ranked_ads from JSON in response
            ranked_ads = parse_ranked_ads(text_content)

            return {
                "request_id": request_id,
                "ranked_ads": ranked_ads,
                "tool_calls": tool_calls_log,
                "n_tool_calls": len(tool_calls_log),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "iterations": iteration + 1,
                "raw_response": text_content[:2000],  # Truncate for storage
            }
        else:
            print(f"    Unexpected stop_reason: {response.stop_reason}")
            break

    print(f"    Max iterations reached for request {request_id}")
    return None


def parse_ranked_ads(text):
    """Extract ranked_ads list from Claude's response text."""
    import re

    # Try to find JSON block
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*"ranked_ads"[^{}]*\[.*?\][^{}]*\})',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "ranked_ads" in parsed:
                    return [int(x) for x in parsed["ranked_ads"]]
            except (json.JSONDecodeError, ValueError):
                continue

    # Fallback: try to find any JSON with ranked_ads
    try:
        # Find the last occurrence of ranked_ads
        idx = text.rfind('"ranked_ads"')
        if idx >= 0:
            # Find enclosing braces
            start = text.rfind('{', 0, idx)
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            parsed = json.loads(text[start:i+1])
                            if "ranked_ads" in parsed:
                                return [int(x) for x in parsed["ranked_ads"]]
                            break
    except (json.JSONDecodeError, ValueError):
        pass

    print("    WARNING: Could not parse ranked_ads from response")
    return []


def main():
    parser = argparse.ArgumentParser(description="Run agent recommendation benchmark")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--data-dir", default="data", help="Directory with npz files")
    parser.add_argument("--requests-dir", default="requests", help="Directory with user contexts")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--max-requests", type=int, default=5, help="Max requests to process")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    args = parser.parse_args()

    # Load context files
    context = load_context_files()

    # Find npz files
    data_dir = Path(args.data_dir)
    npz_files = sorted(data_dir.glob("request_*.npz"))
    if args.max_requests:
        npz_files = npz_files[:args.max_requests]

    print(f"Running benchmark '{args.run_id}' on {len(npz_files)} requests with model {args.model}")

    # Create output directory
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Claude client
    client = anthropic.Anthropic()

    results = []
    for i, npz_path in enumerate(npz_files):
        request_data = load_request(npz_path)
        request_id = request_data["request_id"]
        print(f"\n[{i+1}/{len(npz_files)}] Processing request {request_id}...")

        start_time = time.time()
        result = run_single_request(client, request_data, context, args.requests_dir, args.model)
        elapsed = time.time() - start_time

        if result:
            result["elapsed_seconds"] = round(elapsed, 1)
            n_ranked = len(result.get("ranked_ads", []))
            print(f"  Done: {result['n_tool_calls']} tool calls, {n_ranked} ranked ads, "
                  f"{result['input_tokens']}+{result['output_tokens']} tokens, {elapsed:.1f}s")

            # Save per-request output
            out_path = run_dir / f"{request_id}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)
        else:
            print(f"  FAILED")

    # Save run summary
    summary = {
        "run_id": args.run_id,
        "model": args.model,
        "n_requests": len(results),
        "avg_tool_calls": np.mean([r["n_tool_calls"] for r in results]) if results else 0,
        "avg_ranked_ads": np.mean([len(r.get("ranked_ads", [])) for r in results]) if results else 0,
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "avg_elapsed": np.mean([r["elapsed_seconds"] for r in results]) if results else 0,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Run '{args.run_id}' complete: {len(results)}/{len(npz_files)} requests")
    print(f"Avg tool calls: {summary['avg_tool_calls']:.1f}")
    print(f"Avg ranked ads: {summary['avg_ranked_ads']:.0f}")
    print(f"Total tokens: {summary['total_input_tokens']} in + {summary['total_output_tokens']} out")
    print(f"Results saved to {run_dir}/")


if __name__ == "__main__":
    main()
