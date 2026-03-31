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
    return {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
        "labels": data["labels"],
    }


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

    cluster_info = []
    for c in cl.get("clusters", []):
        cluster_info.append(f"C{c['cluster_id']}(size={c['size']},eng_rate={c.get('engagement_rate',0):.3f})")

    content = f"""# Request {request_id}

## Signal Assessment
- similarity_gap: {ea.get('similarity_gap',0):.6f}
- gap_ratio: {ea.get('gap_ratio',0):.4f}
- overlap_fraction: {ea.get('overlap_fraction',0):.4f}
- n_positive: {ea['n_positive']}, n_negative: {ea['n_negative']}
- FR centroid_gap: {fr.get('centroid_gap',0):.6f} (vs user_gap: {fr.get('user_emb_gap',0):.6f})

## Clusters
{', '.join(cluster_info)}
Top engaged: {ea.get('top_engaged_cluster_ids',[])}

## Tool Results (ad_id lists ranked by score)
FR centroid search ({len(fr_ids)}): {fr_ids}
Embedding similarity ({len(emb_ids)}): {emb_ids}
Anti-negative scorer ({len(an_ids)}): {an_ids}
Cluster explorer ({len(cl_ids)}): {cl_ids}
Prod model ranker ({len(pm_ids)}): {pm_ids}
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
    parser.add_argument("--data-dir", default="data_bulk_eval")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("request_*.npz"))[:args.max_requests]
    print(f"Batch benchmark '{args.run_id}': {len(npz_files)} requests, single Claude call")

    # Step 1: Pre-compute all tools → write to temp folder
    tool_dir = BASE_DIR / "tool_results_batch"
    if tool_dir.exists():
        shutil.rmtree(tool_dir)
    tool_dir.mkdir()

    print(f"\nStep 1: Pre-computing tools for {len(npz_files)} requests...")
    t0 = time.time()
    request_ids = []
    for i, npz_path in enumerate(npz_files):
        rd = load_request(npz_path)
        rid = precompute_and_write(rd, tool_dir)
        request_ids.append(rid)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(npz_files)}] done")
    tool_time = time.time() - t0
    print(f"  All tools pre-computed in {tool_time:.1f}s")

    # Write instruction file
    instruction = f"""Read ALL request files in this folder. For EACH request:
1. Check similarity_gap to assess signal quality
2. If gap > 0.05: prioritize embedding similarity results
3. If gap < 0.01: prioritize FR centroid + cluster engagement
4. Merge all tool result ad_id lists, de-duplicate
5. Output 150+ ranked ad IDs per request

Output a single JSON array with one object per request:
```json
[
  {{"request_id": <id>, "ranked_ads": [ad1, ad2, ...], "strategy": "brief"}},
  ...
]
```

Process all {len(request_ids)} requests. Output ONLY the JSON array."""

    (tool_dir / "INSTRUCTIONS.md").write_text(instruction)

    # Step 2: Single Claude call pointing to the folder
    print(f"\nStep 2: Calling Claude to process {len(request_ids)} requests...")
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(BASE_DIR))
        response = result.stdout
    except subprocess.TimeoutExpired:
        print("  Timed out (600s)")
        response = ""
    except Exception as e:
        print(f"  Error: {e}")
        response = ""

    claude_time = time.time() - t1
    print(f"  Claude responded in {claude_time:.1f}s ({len(response)} chars)")

    # Step 3: Parse and save results
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
            print(f"  Request {rid}: {len(ranked)} ranked ads")

    # Cleanup
    shutil.rmtree(tool_dir)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Run '{args.run_id}': {len(batch_results)}/{len(npz_files)} requests in {total_time:.0f}s")
    print(f"  Tool pre-computation: {tool_time:.0f}s")
    print(f"  Claude call: {claude_time:.0f}s")
    print(f"  Time per request: {total_time/max(len(npz_files),1):.1f}s")
    print(f"  Estimated 100 requests: {total_time/max(len(npz_files),1)*100/60:.0f} min")

    summary = {"run_id": args.run_id, "n_results": len(batch_results), "elapsed": total_time}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
