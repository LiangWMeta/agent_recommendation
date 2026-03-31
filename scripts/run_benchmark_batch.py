#!/usr/bin/env python3
"""
Batch agent benchmark: pre-compute tool results to a folder, single claude -p call.

1. Pre-computes all tool results for all requests → writes to a temp folder
2. Points claude -p at the folder with --add-dir
3. Claude reads all files and outputs rankings for all requests in one call

Usage:
    python3 scripts/run_benchmark_batch.py --run-id batch_20 --data-dir data_bulk_eval --max-requests 20
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    rd = {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
    }
    # Use history_labels for tools (no leakage), fall back to labels
    if "history_labels" in data:
        rd["labels"] = data["history_labels"]  # tools see only history
        rd["test_labels"] = data["test_labels"]  # for evaluation
    else:
        rd["labels"] = data["labels"]
        rd["test_labels"] = data["labels"]
    return rd


def precompute_and_write(rd, output_dir):
    """Pre-compute all tools and write results as a markdown file."""
    request_id = rd["request_id"]

    ea = engagement_pattern_analyzer(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"])
    fr = fr_centroid_search(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=150)
    emb = embedding_similarity_search(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], top_k=150)
    cl = cluster_explorer(rd["ad_embs"], rd["ad_ids"], n_clusters=5, top_k_per_cluster=30, labels=rd["labels"])
    an = anti_negative_scorer(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], alpha=0.3, top_k=100)
    pm = prod_model_ranker(rd["ad_ids"], top_k=100, request_id=request_id)

    fr_ids = [r["ad_id"] for r in fr.get("results", [])]
    emb_ids = [r["ad_id"] for r in emb.get("results", [])]
    an_ids = [r["ad_id"] for r in an.get("results", [])]
    pm_ids = [r["ad_id"] for r in pm.get("results", [])]
    cl_ids = [r["ad_id"] for r in cl.get("ads", [])]

    # Cross-route overlap
    routes = {"FR": set(fr_ids), "Emb": set(emb_ids), "AntiNeg": set(an_ids), "Cluster": set(cl_ids), "Prod": set(pm_ids)}
    all_unique = set()
    for s in routes.values():
        all_unique |= s

    ad_route_count = {}
    for ad in all_unique:
        ad_route_count[ad] = sum(1 for s in routes.values() if ad in s)
    consensus_4plus = [ad for ad, c in ad_route_count.items() if c >= 4]
    consensus_3 = [ad for ad, c in ad_route_count.items() if c == 3]

    # Per-cluster analysis with route coverage
    from sklearn.cluster import KMeans as _KM
    _km = _KM(n_clusters=5, random_state=42, n_init=10)
    _cids = _km.fit_predict(rd["ad_embs"])
    _cosines = (rd["ad_embs"] / np.linalg.norm(rd["ad_embs"], axis=1, keepdims=True).clip(1e-8)) @ (rd["user_emb"] / np.linalg.norm(rd["user_emb"]).clip(1e-8))
    _norms = np.linalg.norm(rd["ad_embs"], axis=1)
    _pos = rd["labels"] == 1

    # User engagement profile
    pos_by_c = {}
    for i in range(len(rd["labels"])):
        if rd["labels"][i] == 1:
            c = _cids[i]
            pos_by_c[c] = pos_by_c.get(c, 0) + 1
    total_pos = sum(pos_by_c.values()) or 1

    user_profile = []
    for c in sorted(pos_by_c, key=pos_by_c.get, reverse=True):
        user_profile.append(f"  {pos_by_c[c]/total_pos*100:.0f}% in Cluster {c} ({pos_by_c[c]} ads)")

    # Cluster pool description
    cluster_desc = []
    patterns = []
    for c in range(5):
        cm = _cids == c
        c_total = int(cm.sum())
        c_pos = int((cm & _pos).sum())
        eng_rate = c_pos / c_total if c_total > 0 else 0
        c_ad_set = set(int(rd["ad_ids"][i]) for i in range(len(rd["ad_ids"])) if cm[i])
        fr_cov = len(c_ad_set & routes["FR"])
        emb_cov = len(c_ad_set & routes["Emb"])
        cold = int((_norms[cm] < 2.0).sum())

        cluster_desc.append(
            f"  Cluster {c}: {c_total} ads, {c_pos} engaged ({eng_rate:.0%} rate), "
            f"user_affinity={_cosines[cm].mean():.3f}, cold_start={cold}, "
            f"FR_finds={fr_cov}, Emb_finds={emb_cov}"
        )

        # Detect patterns worth reasoning about
        if eng_rate > 0.15 and emb_cov < 5:
            patterns.append(f"  ⚠ Cluster {c}: HIGH engagement ({eng_rate:.0%}) but Embedding blind ({emb_cov} ads) — use FR/Cluster route")
        if eng_rate < 0.05 and fr_cov > 15:
            patterns.append(f"  ⚠ Cluster {c}: LOW engagement ({eng_rate:.0%}) but FR finds {fr_cov} ads — possible noise, deprioritize")
        if fr_cov > 20 and emb_cov > 20 and eng_rate > 0.1:
            patterns.append(f"  ✓ Cluster {c}: Both routes agree, good engagement ({eng_rate:.0%}) — high confidence")

    content = f"""# Request {request_id}

## Ads Pool Profile
Total: {len(rd['ad_ids'])} candidates, {int(_pos.sum())} engaged ({int(_pos.sum())/len(rd['ad_ids'])*100:.0f}% rate)
{chr(10).join(cluster_desc)}

## User Engagement Profile
{chr(10).join(user_profile)}

## Route Diagnostics
Embedding discrimination: gap={ea.get('similarity_gap',0):.4f} (>0.05=strong, <0.01=unreliable)
FR centroid discrimination: gap={fr.get('centroid_gap',0):.4f}
FR vs Embedding correlation: {fr.get('centroid_vs_user_correlation',0):.3f} (low=independent signals)
FR ∩ Embedding overlap: {len(routes['FR'] & routes['Emb'])}/{min(len(fr_ids),len(emb_ids))} ({len(routes['FR'] & routes['Emb'])/max(min(len(fr_ids),len(emb_ids)),1)*100:.0f}%)
Consensus ads (4+ routes): {len(consensus_4plus)}
Total unique across all routes: {len(all_unique)}

## Key Patterns
{chr(10).join(patterns) if patterns else '  No major route conflicts detected'}

## Route Results (ranked by score, best first)
FR centroid ({len(fr_ids)}): {fr_ids}
Embedding ({len(emb_ids)}): {emb_ids}
Anti-negative ({len(an_ids)}): {an_ids}
Cluster ({len(cl_ids)}): {cl_ids}
Prod model ({len(pm_ids)}, avail={pm.get('available',False)}): {pm_ids}
"""

    fpath = Path(output_dir) / f"request_{request_id}.md"
    fpath.write_text(content)
    return request_id


def parse_batch_results(text):
    """Parse batch results — array of per-request outputs."""
    for pattern in [r'```json\s*(\[.*?\])\s*```', r'```\s*(\[.*?\])\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue
    # Try raw JSON
    try:
        idx = text.find('[')
        if idx >= 0:
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == '[': depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[idx:i+1])
    except json.JSONDecodeError:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Batch agent benchmark via folder")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--requests-dir", default="requests")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20, help="Requests per Claude call")
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("request_*.npz"))[:args.max_requests]
    batch_size = args.batch_size
    n_batches = (len(npz_files) + batch_size - 1) // batch_size
    print(f"Batch benchmark '{args.run_id}': {len(npz_files)} requests, {n_batches} batches of {batch_size}")

    t0 = time.time()
    all_results = []

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(npz_files))
        batch_files = npz_files[batch_start:batch_end]

        print(f"\n--- Batch {batch_idx+1}/{n_batches} ({len(batch_files)} requests) ---")

        # Pre-compute tools for this batch
        tool_dir = BASE_DIR / "tool_results_batch"
        if tool_dir.exists():
            shutil.rmtree(tool_dir)
        tool_dir.mkdir()

        request_ids = []
        for npz_path in batch_files:
            rd = load_request(npz_path)
            rid = precompute_and_write(rd, tool_dir)
            request_ids.append(rid)
        print(f"  Tools pre-computed for {len(request_ids)} requests")

        instruction = f"""You are analyzing ads retrieval results from multiple routes for {len(request_ids)} requests.

For EACH request file, you have:
- Signal diagnostics (how reliable each route is for this specific request)
- Cross-route analysis (which routes agree/disagree, consensus ads)
- Results from 5 retrieval routes, each finding different candidate ads

YOUR JOB: For each request, reason about the signals and produce a ranking. Think about:

1. ROUTE RELIABILITY: Which routes are trustworthy for this request?
   - similarity_gap tells you if embedding search is discriminative (high = reliable)
   - FR centroid_gap vs user_emb_gap shows which query vector works better
   - correlation tells you if FR and Embedding are finding the same or different ads

2. CONSENSUS: Ads appearing in 4+ routes are high-confidence. But don't ignore
   route-specific ads — they may capture signals others miss.

3. CLUSTER PATTERNS: High-engagement clusters deserve more candidates.
   If a cluster has 23% engagement rate but only 3 ads in embedding search,
   the embedding route is blind to that cluster — use FR or cluster explorer ads instead.

4. ROUTE CONFLICTS: When FR and Embedding disagree strongly (low overlap),
   understand WHY — they capture different aspects of user preference.
   Both perspectives may be valid.

5. PROD MODEL: When available, prod model scores reflect production ranking quality.
   But low coverage means many ads lack this signal.

Explain your reasoning briefly in the strategy field. Output 150+ ranked ad IDs per request.

Output format:
```json
[
  {{"request_id": <id>, "ranked_ads": [ad1, ad2, ...], "strategy": "reasoning about route selection and merging"}},
  ...
]
```

Process all {len(request_ids)} requests. Output ONLY the JSON array."""
        (tool_dir / "INSTRUCTIONS.md").write_text(instruction)

        print(f"  Calling Claude...")
        t1 = time.time()
        cmd = [
            "claude", "-p",
            f"Process all {len(request_ids)} request files in the tool_results_batch/ folder. Follow INSTRUCTIONS.md.",
            "--model", args.model,
            "--output-format", "text",
            "--permission-mode", "bypassPermissions",
            "--add-dir", str(tool_dir),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(BASE_DIR))
            response = result.stdout
        except subprocess.TimeoutExpired:
            print("  Timed out (1800s)")
            response = ""
        except Exception as e:
            print(f"  Error: {e}")
            response = ""

        claude_time = time.time() - t1
        print(f"  Claude responded in {claude_time:.1f}s ({len(response)} chars)")

        batch_results = parse_batch_results(response)
        print(f"  Parsed {len(batch_results)} results")

        for result_item in batch_results:
            rid = result_item.get("request_id")
            ranked = result_item.get("ranked_ads", [])
            if rid and ranked:
                out = {
                    "request_id": int(rid),
                    "ranked_ads": [int(a) for a in ranked],
                    "strategy": result_item.get("strategy", ""),
                }
                with open(run_dir / f"{rid}.json", "w") as f:
                    json.dump(out, f, indent=2)
                all_results.append(out)
                print(f"    Request {rid}: {len(ranked)} ranked ads")

        result_ids = {r.get("request_id") for r in batch_results}
        for rid in request_ids:
            if rid not in result_ids:
                print(f"    Request {rid}: MISSING")

        shutil.rmtree(tool_dir)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Run '{args.run_id}': {len(all_results)}/{len(npz_files)} requests in {total_time:.0f}s")
    if all_results:
        avg_ranked = sum(len(r.get("ranked_ads", [])) for r in all_results) / len(all_results)
        print(f"Avg ranked ads: {avg_ranked:.0f}")
        print(f"Time per request: {total_time/max(len(npz_files),1):.1f}s")
    print(f"  Estimated 100 requests: {total_time/max(len(npz_files),1)*100/60:.0f} min")

    summary = {"run_id": args.run_id, "n_results": len(batch_results), "elapsed": total_time}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
