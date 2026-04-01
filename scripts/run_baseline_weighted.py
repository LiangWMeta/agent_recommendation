#!/usr/bin/env python3
"""
Fixed-weight baseline: combine all tool signals with predetermined weights.
No LLM reasoning — just weighted rank aggregation.

This is the "dumb" baseline to compare against the LLM agent.
If the LLM can't beat this, its reasoning isn't adding value.

Usage:
    python3 scripts/run_baseline_weighted.py --run-id baseline_weighted --data-dir data --max-requests 100
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pselect_main_route import pselect_main_route
from tools.forced_retrieval import forced_retrieval
from tools.anti_negative_scorer import anti_negative_scorer
from tools.cluster_explorer import cluster_explorer
from tools.engagement_analyzer import engagement_pattern_analyzer
from tools.prod_model_ranker import prod_model_ranker


def load_request(npz_path):
    data = np.load(npz_path)
    rd = {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
    }
    if "history_labels" in data:
        rd["labels"] = data["history_labels"]
        rd["test_labels"] = data["test_labels"]
    else:
        rd["labels"] = data["labels"]
        rd["test_labels"] = data["labels"]
    return rd


def weighted_rank_fusion(route_results, weights):
    """Reciprocal Rank Fusion with weights.

    For each ad, score = sum(weight_i / rank_i) across routes.
    """
    ad_scores = defaultdict(float)

    for route_name, (ad_ids_list, weight) in route_results.items():
        for rank, ad_id in enumerate(ad_ids_list):
            ad_scores[ad_id] += weight / (rank + 1)  # reciprocal rank

    # Sort by combined score descending
    ranked = sorted(ad_scores.items(), key=lambda x: x[1], reverse=True)
    return [int(ad_id) for ad_id, _ in ranked]


def process_request(rd):
    """Run all tools and combine with fixed weights."""
    # Run tools
    fr = forced_retrieval(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=150)
    emb = pselect_main_route(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], top_k=150)
    an = anti_negative_scorer(rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], alpha=0.3, top_k=100)
    cl = cluster_explorer(rd["ad_embs"], rd["ad_ids"], n_clusters=5, top_k_per_cluster=30, labels=rd["labels"])
    pm = prod_model_ranker(rd["ad_ids"], top_k=100, request_id=rd["request_id"])

    # Extract ranked ad_id lists
    fr_ids = [r["ad_id"] for r in fr.get("results", [])]
    emb_ids = [r["ad_id"] for r in emb.get("results", [])]
    an_ids = [r["ad_id"] for r in an.get("results", [])]
    cl_ids = [r["ad_id"] for r in cl.get("ads", [])]
    pm_ids = [r["ad_id"] for r in pm.get("results", [])]

    # Fixed weights — equal weighting across all routes
    route_results = {
        "fr_centroid": (fr_ids, 1.0),
        "embedding": (emb_ids, 1.0),
        "anti_negative": (an_ids, 1.0),
        "cluster": (cl_ids, 0.5),  # slightly lower — cluster is diversity-focused
        "prod_model": (pm_ids, 1.0),
    }

    ranked = weighted_rank_fusion(route_results, None)
    return ranked[:300]  # top 300


def main():
    parser = argparse.ArgumentParser(description="Fixed-weight baseline")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-requests", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("request_*.npz"))[:args.max_requests]
    print(f"Fixed-weight baseline '{args.run_id}': {len(npz_files)} requests")

    results = []
    for i, npz_path in enumerate(npz_files):
        rd = load_request(npz_path)
        ranked = process_request(rd)

        out = {
            "request_id": rd["request_id"],
            "ranked_ads": ranked,
            "strategy": "fixed_weight_reciprocal_rank_fusion",
        }
        with open(run_dir / f"{rd['request_id']}.json", "w") as f:
            json.dump(out, f)
        results.append(out)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(npz_files)}] done")

    avg_ranked = sum(len(r["ranked_ads"]) for r in results) / len(results) if results else 0
    print(f"\nDone: {len(results)} requests, avg {avg_ranked:.0f} ranked ads")

    summary = {"run_id": args.run_id, "n_results": len(results), "strategy": "fixed_weight_RRF"}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
