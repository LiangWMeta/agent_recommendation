#!/usr/bin/env python3
"""Accumulate benchmark results into evaluation/history.json.

Usage:
    python3 scripts/update_history.py --run-id cc_5req --data-dir data

Loads output files from outputs/<run_id>/, computes signal characteristics
from the corresponding npz data files, loads evaluation results, and appends
history entries to evaluation/history.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_signal_characteristics(npz_path: str) -> dict:
    """Compute signal characteristics from a request npz file."""
    data = np.load(npz_path)
    user_emb = data["user_emb"]
    ad_embs = data["ad_embs"]
    labels = data["labels"]

    n_candidates = len(ad_embs)

    # Positive rate
    n_positive = int((labels == 1).sum())
    positive_rate = n_positive / n_candidates if n_candidates > 0 else 0.0

    # Cosine similarities
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
    similarities = ad_norms @ user_norm

    pos_mask = labels == 1
    neg_mask = labels == 0

    avg_pos_sim = float(similarities[pos_mask].mean()) if n_positive > 0 else 0.0
    avg_neg_sim = float(similarities[neg_mask].mean()) if (neg_mask.sum()) > 0 else 0.0

    similarity_gap = avg_pos_sim - avg_neg_sim
    gap_ratio = similarity_gap / abs(avg_neg_sim) if avg_neg_sim != 0 else 0.0

    return {
        "similarity_gap": float(similarity_gap),
        "gap_ratio": float(gap_ratio),
        "positive_rate": float(positive_rate),
        "n_candidates": int(n_candidates),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Accumulate benchmark results into evaluation/history.json"
    )
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., cc_5req)")
    parser.add_argument("--data-dir", default="data", help="Directory containing npz files")
    parser.add_argument("--base-dir", default=".", help="Base directory for the project")
    args = parser.parse_args()

    base = Path(args.base_dir)
    outputs_dir = base / "outputs" / args.run_id
    data_dir = base / args.data_dir
    eval_results_path = base / "evaluation" / "results" / f"{args.run_id}.json"
    history_path = base / "evaluation" / "history.json"

    # Load evaluation results
    if not eval_results_path.exists():
        print(f"Error: Evaluation results not found at {eval_results_path}", file=sys.stderr)
        sys.exit(1)

    with open(eval_results_path) as f:
        eval_results = json.load(f)

    # Build per-request recall lookup
    recall_lookup = {}
    for entry in eval_results.get("per_request", []):
        rid = entry["request_id"]
        recall_lookup[rid] = {
            "recall_at_50": entry.get("recall@50"),
            "recall_at_100": entry.get("recall@100"),
        }

    # Load existing history
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    # Build set of existing (request_id, run_id) pairs to avoid duplicates
    existing_keys = {(e["request_id"], e["run_id"]) for e in history}

    # Process each output file
    output_files = sorted(outputs_dir.glob("*.json"))
    added = 0

    for output_file in output_files:
        if output_file.name == "summary.json":
            continue

        try:
            with open(output_file) as f:
                output_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read {output_file}", file=sys.stderr)
            continue

        request_id = output_data.get("request_id")
        if request_id is None:
            continue

        # Skip duplicates
        if (request_id, args.run_id) in existing_keys:
            continue

        # Find corresponding npz
        npz_path = data_dir / f"request_{request_id}.npz"
        if not npz_path.exists():
            print(f"Warning: npz not found for request {request_id}", file=sys.stderr)
            continue

        # Compute signal characteristics
        signals = compute_signal_characteristics(str(npz_path))

        # Get recall metrics
        recalls = recall_lookup.get(request_id, {})

        # Extract tool_calls from raw_response if present (parse tool use patterns)
        tool_calls = []
        raw = output_data.get("raw_response", "")
        # Simple heuristic: look for tool names in the response
        tool_names = [
            "embedding_similarity_search", "feature_filter", "cluster_explorer",
            "similar_ads_lookup", "engagement_pattern_analyzer", "ads_pool_stats",
            "lookup_similar_requests",
        ]
        for name in tool_names:
            if name in raw:
                tool_calls.append(name)

        history_entry = {
            "request_id": request_id,
            "run_id": args.run_id,
            "similarity_gap": signals["similarity_gap"],
            "gap_ratio": signals["gap_ratio"],
            "positive_rate": signals["positive_rate"],
            "n_candidates": signals["n_candidates"],
            "strategy": output_data.get("strategy", ""),
            "recall_at_50": recalls.get("recall_at_50"),
            "recall_at_100": recalls.get("recall_at_100"),
            "tool_calls": tool_calls,
        }

        history.append(history_entry)
        existing_keys.add((request_id, args.run_id))
        added += 1

    # Write updated history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Added {added} entries to {history_path} (total: {len(history)})")


if __name__ == "__main__":
    main()
