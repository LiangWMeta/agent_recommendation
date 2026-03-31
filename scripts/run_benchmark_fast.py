#!/usr/bin/env python3
"""
Fast agent benchmark: pre-compute all tool results, single Claude call per request.

Instead of Claude making 4-6 sequential MCP tool calls (~3-5 min per request),
this script:
1. Pre-computes ALL tool results locally (Python, <1 sec per request)
2. Sends Claude a single prompt with all tool results embedded
3. Claude only needs to reason + output ranked ads (~15-20 sec per request)

Speedup: ~10-15x faster than MCP-based approach.

Usage:
    python3 scripts/run_benchmark_fast.py --run-id fast_bulk --data-dir data_bulk_eval --max-requests 20
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.embedding_search import embedding_similarity_search
from tools.fr_centroid_search import fr_centroid_search
from tools.anti_negative_scorer import anti_negative_scorer
from tools.cluster_explorer import cluster_explorer
from tools.engagement_analyzer import engagement_pattern_analyzer
from tools.pool_stats import ads_pool_stats
from tools.prod_model_ranker import prod_model_ranker

BASE_DIR = Path(__file__).parent.parent


def load_request(npz_path):
    data = np.load(npz_path)
    return {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
        "labels": data["labels"],
    }


def precompute_tool_results(request_data):
    """Run ALL tools locally and return results as formatted text."""
    rd = request_data
    results = {}

    # 1. Pool stats
    results["ads_pool_stats"] = ads_pool_stats(
        rd["ad_embs"], rd["ad_ids"], rd["user_emb"], n_clusters=5
    )

    # 2. Engagement pattern analyzer
    results["engagement_pattern_analyzer"] = engagement_pattern_analyzer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"]
    )

    # 3. FR centroid search
    results["fr_centroid_search"] = fr_centroid_search(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=150
    )

    # 4. Embedding similarity search
    results["embedding_similarity_search"] = embedding_similarity_search(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], top_k=150
    )

    # 5. Cluster explorer with engagement rates
    results["cluster_explorer"] = cluster_explorer(
        rd["ad_embs"], rd["ad_ids"], n_clusters=5, top_k_per_cluster=30,
        labels=rd["labels"]
    )

    # 6. Anti-negative scorer
    results["anti_negative_scorer"] = anti_negative_scorer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
        alpha=0.3, top_k=100
    )

    # 7. Prod model ranker (if available)
    results["prod_model_ranker"] = prod_model_ranker(
        rd["ad_ids"], top_k=100, request_id=rd["request_id"]
    )

    return results


def format_tool_results(results):
    """Format pre-computed tool results as text for Claude."""
    sections = []

    # Pool stats
    ps = results["ads_pool_stats"]
    sections.append(f"""## ads_pool_stats
- Total ads: {ps['total_ads']}
- Similarity stats: mean={ps['similarity_stats']['mean']:.4f}, std={ps['similarity_stats']['std']:.4f}
- Clusters: {json.dumps([{k: c[k] for k in ['cluster_id', 'size', 'avg_similarity']} for c in ps.get('cluster_distribution', [])], default=str)}""")

    # Engagement analyzer
    ea = results["engagement_pattern_analyzer"]
    sections.append(f"""## engagement_pattern_analyzer
- Positive: {ea['n_positive']}, Negative: {ea['n_negative']}
- similarity_gap: {ea.get('similarity_gap', 0):.6f}
- gap_ratio: {ea.get('gap_ratio', 0):.4f}
- overlap_fraction: {ea.get('overlap_fraction', 0):.4f}
- engagement_rate_variance: {ea.get('engagement_rate_variance', 0):.6f}
- top_engaged_cluster_ids: {ea.get('top_engaged_cluster_ids', [])}
- top_positive_ad_ids: {ea.get('top_positive_ad_ids', [])[:5]}
- Engagement by cluster: {json.dumps(ea.get('engagement_by_cluster', []), default=str)}""")

    # FR centroid search
    fr = results["fr_centroid_search"]
    fr_results = fr.get("results", [])
    sections.append(f"""## fr_centroid_search (top_k=150)
- centroid_gap: {fr.get('centroid_gap', 0):.6f}
- user_emb_gap: {fr.get('user_emb_gap', 0):.6f}
- centroid_vs_user_correlation: {fr.get('centroid_vs_user_correlation', 0):.4f}
- n_positives_used: {fr.get('n_positives_used', 0)}
- Top 10 results: {json.dumps(fr_results[:10], default=str)}
- Full result ad_ids ({len(fr_results)} total): {[r['ad_id'] for r in fr_results]}""")

    # Embedding similarity search
    es = results["embedding_similarity_search"]
    es_results = es.get("results", [])
    sections.append(f"""## embedding_similarity_search (top_k=150)
- score_range: {es.get('score_range', [])}
- score_std: {es.get('score_std', 0):.6f}
- top_bottom_gap: {es.get('top_bottom_gap', 0):.6f}
- Top 10 results: {json.dumps(es_results[:10], default=str)}
- Full result ad_ids ({len(es_results)} total): {[r['ad_id'] for r in es_results]}""")

    # Cluster explorer
    ce = results["cluster_explorer"]
    sections.append(f"""## cluster_explorer (n_clusters=5, top_k_per_cluster=30)
- Clusters: {json.dumps(ce.get('clusters', []), default=str)}
- Ads ({len(ce.get('ads', []))} total): {json.dumps(ce.get('ads', [])[:20], default=str)}...""")

    # Anti-negative scorer
    an = results["anti_negative_scorer"]
    an_results = an.get("results", [])
    sections.append(f"""## anti_negative_scorer (alpha=0.3, top_k=100)
- pos_neg_centroid_similarity: {an.get('pos_neg_centroid_similarity', 0):.4f}
- Full result ad_ids ({len(an_results)} total): {[r['ad_id'] for r in an_results]}""")

    # Prod model ranker
    pm = results["prod_model_ranker"]
    pm_results = pm.get("results", [])
    sections.append(f"""## prod_model_ranker (top_k=100)
- available: {pm.get('available', False)}
- coverage: {pm.get('coverage', 0):.2%}
- Full result ad_ids ({len(pm_results)} total): {[r['ad_id'] for r in pm_results]}""")

    return "\n\n".join(sections)


def build_single_call_prompt(request_id, user_context, tool_results_text):
    """Build a single prompt with all tool results embedded."""
    return (
        f"You are an ads recommendation orchestrator for request {request_id}.\n\n"
        f"## User Information\n{user_context}\n\n"
        f"## Pre-computed Tool Results\n"
        f"All tool results have been pre-computed. Use them to build your ranking.\n\n"
        f"{tool_results_text}\n\n"
        f"## Instructions\n"
        f"Based on the tool results above:\n"
        f"1. Assess signal quality from engagement_pattern_analyzer (check similarity_gap)\n"
        f"2. Choose strategy: strong signal → embedding-first, weak → FR centroid-first\n"
        f"3. Aggregate results from all tools, de-duplicate, and produce a final ranked list\n"
        f"4. Include at least 150 ad IDs\n\n"
        f"Output EXACTLY this JSON format:\n"
        f'```json\n{{"ranked_ads": [ad_id_1, ad_id_2, ...], "strategy": "brief description"}}\n```'
    )


def parse_ranked_ads(text):
    """Extract ranked_ads from Claude's response."""
    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "ranked_ads" in parsed:
                    return [int(x) for x in parsed["ranked_ads"]], parsed.get("strategy", "")
            except (json.JSONDecodeError, ValueError):
                continue
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


def run_single_request(request_id, prompt, model="sonnet"):
    """Run Claude with a single prompt (no MCP tools needed)."""
    system_parts = []
    for fname in ["CLAUDE.md", "architecture.md", "skill.md", "learnings.md"]:
        fpath = BASE_DIR / fname
        if fpath.exists():
            system_parts.append(fpath.read_text())
    system_prompt = "\n\n---\n\n".join(system_parts)

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        "--append-system-prompt", system_prompt,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
        ranked_ads, strategy = parse_ranked_ads(result.stdout)
        return {
            "request_id": request_id,
            "ranked_ads": ranked_ads,
            "strategy": strategy,
            "raw_response_length": len(result.stdout),
        }
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fast agent benchmark (pre-computed tools)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data_bulk_eval")
    parser.add_argument("--requests-dir", default="requests_bulk_eval")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--request-ids", nargs="+", type=int)
    parser.add_argument("--model", default="sonnet")
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

    print(f"Fast benchmark '{args.run_id}' on {len(npz_files)} requests (model={args.model})")
    print(f"Pre-computing all tool results locally, then single Claude call per request\n")

    results = []
    for i, npz_path in enumerate(npz_files):
        rd = load_request(npz_path)
        request_id = rd["request_id"]

        print(f"[{i+1}/{len(npz_files)}] Request {request_id}...")

        # Step 1: Pre-compute tools (~0.5 sec)
        t0 = time.time()
        tool_results = precompute_tool_results(rd)
        tool_time = time.time() - t0

        # Step 2: Format tool results as text
        tool_text = format_tool_results(tool_results)

        # Step 3: Load user context
        user_ctx_parts = []
        for fname in ["profile.md", "engagement.md", "interest_clusters.md", "context.md"]:
            fpath = requests_dir / str(request_id) / fname
            if fpath.exists():
                user_ctx_parts.append(fpath.read_text())
        user_context = "\n\n---\n\n".join(user_ctx_parts) if user_ctx_parts else f"Request {request_id}"

        # Step 4: Single Claude call
        prompt = build_single_call_prompt(request_id, user_context, tool_text)

        t1 = time.time()
        result = run_single_request(request_id, prompt, args.model)
        claude_time = time.time() - t1

        if result:
            result["tool_precompute_seconds"] = round(tool_time, 1)
            result["claude_seconds"] = round(claude_time, 1)
            result["total_seconds"] = round(tool_time + claude_time, 1)
            n_ranked = len(result.get("ranked_ads", []))
            print(f"  Done: {n_ranked} ranked ads, tools={tool_time:.1f}s, claude={claude_time:.1f}s")

            out_path = run_dir / f"{request_id}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            results.append(result)
        else:
            print(f"  FAILED")

    # Summary
    if results:
        avg_total = sum(r["total_seconds"] for r in results) / len(results)
        avg_claude = sum(r["claude_seconds"] for r in results) / len(results)
        avg_ranked = sum(len(r.get("ranked_ads", [])) for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"Run '{args.run_id}': {len(results)}/{len(npz_files)} requests")
        print(f"Avg time: {avg_total:.1f}s total ({avg_claude:.1f}s Claude)")
        print(f"Avg ranked ads: {avg_ranked:.0f}")
        print(f"Estimated 100 requests: {avg_total * 100 / 60:.0f} min")

    summary = {
        "run_id": args.run_id,
        "model": args.model,
        "n_requests": len(results),
        "avg_ranked_ads": sum(len(r.get("ranked_ads", [])) for r in results) / max(len(results), 1),
        "avg_total_seconds": sum(r["total_seconds"] for r in results) / max(len(results), 1),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {run_dir}/")


if __name__ == "__main__":
    main()
