#!/usr/bin/env python3
"""Evaluate Claude agent outputs against RAA ground-truth labels.

Loads ranked ad lists from outputs/<run_id>/<request_id>.json and computes
Recall@K, Precision@K, and NDCG@K against ground truth from data/ npz files.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


K_VALUES = [10, 20, 50, 100]


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute DCG@K."""
    rel = relevances[:k]
    positions = np.arange(1, len(rel) + 1)
    return float(np.sum(rel / np.log2(positions + 1)))


def ndcg_at_k(ranked_relevances: np.ndarray, all_relevances: np.ndarray, k: int) -> float:
    """Compute NDCG@K."""
    dcg = dcg_at_k(ranked_relevances, k)
    # Ideal: sort all relevances descending
    ideal = np.sort(all_relevances)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_request(ranked_ad_ids: list, ad_ids: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate a single request given ranked output and ground truth."""
    # Build ad_id -> label lookup
    id_to_label = dict(zip(ad_ids.tolist(), labels.tolist()))
    n_positives = int(labels.sum())

    # Map ranked ads to relevance (1 if positive, 0 if negative or unknown)
    ranked_relevances = np.array([id_to_label.get(int(aid), 0) for aid in ranked_ad_ids])
    all_relevances = labels.astype(float)

    result = {"n_positives": n_positives, "n_ranked": len(ranked_ad_ids)}
    for k in K_VALUES:
        top_k_rel = ranked_relevances[:k]
        hits = top_k_rel.sum()

        result[f"recall@{k}"] = float(hits / n_positives) if n_positives > 0 else 0.0
        result[f"precision@{k}"] = float(hits / k)
        result[f"ndcg@{k}"] = ndcg_at_k(ranked_relevances, all_relevances, k)

    return result


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent run against ground truth")
    parser.add_argument("--run-id", required=True, help="Run ID (subfolder under outputs/)")
    parser.add_argument("--baseline", default="evaluation/results/baseline.json",
                        help="Path to baseline results JSON")
    parser.add_argument("--data-dir", default="data/", help="Directory with .npz files")
    args = parser.parse_args()

    outputs_dir = os.path.join("outputs", args.run_id)
    if not os.path.isdir(outputs_dir):
        print(f"ERROR: outputs directory not found: {outputs_dir}")
        sys.exit(1)

    # Load ground truth index: request_id -> npz path
    npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    gt_index = {}
    for path in npz_files:
        data = np.load(path)
        rid = int(data["request_id"])
        gt_index[rid] = path

    # Load output files
    output_files = sorted(glob.glob(os.path.join(outputs_dir, "*.json")))
    if not output_files:
        print(f"ERROR: No output JSON files in {outputs_dir}")
        sys.exit(1)

    print(f"Evaluating run '{args.run_id}' ({len(output_files)} outputs)...")
    t0 = time.time()

    per_request = []
    skipped = 0
    for path in output_files:
        with open(path) as f:
            output = json.load(f)

        ranked_ads = output.get("ranked_ads", [])
        # Extract request_id from filename or content
        request_id = output.get("request_id")
        if request_id is None:
            basename = os.path.basename(path).replace(".json", "")
            try:
                request_id = int(basename)
            except ValueError:
                try:
                    request_id = int(basename.replace("request_", ""))
                except ValueError:
                    continue  # Skip non-request files like summary.json

        if request_id not in gt_index:
            skipped += 1
            continue

        data = np.load(gt_index[request_id])
        ad_ids = data["ad_ids"]
        labels = data["labels"]

        result = evaluate_request(ranked_ads, ad_ids, labels)
        result["request_id"] = request_id
        per_request.append(result)

    elapsed = time.time() - t0
    if skipped > 0:
        print(f"  skipped {skipped} outputs (no matching ground truth)")
    print(f"  evaluated {len(per_request)} requests in {elapsed:.1f}s\n")

    if not per_request:
        print("ERROR: No requests evaluated.")
        sys.exit(1)

    # Aggregate
    metrics = ["recall", "precision", "ndcg"]
    aggregate = {}
    for metric in metrics:
        for k in K_VALUES:
            key = f"{metric}@{k}"
            values = np.array([r[key] for r in per_request])
            ci_lo, ci_hi = bootstrap_ci(values)
            aggregate[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "ci_95": [ci_lo, ci_hi],
            }

    output_data = {
        "run_id": args.run_id,
        "n_requests": len(per_request),
        "aggregate": aggregate,
        "per_request": per_request,
    }

    # Load baseline for comparison if available
    baseline = None
    if os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)

    # Save results
    result_path = os.path.join("evaluation", "results", f"{args.run_id}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {result_path}\n")

    # Print summary
    print(f"{'Metric':<16} {'Mean':>8} {'Std':>8} {'Median':>8}", end="")
    if baseline:
        print(f" {'Baseline':>10} {'Delta':>10}")
    else:
        print()
    print("-" * (56 + (22 if baseline else 0)))

    for metric in metrics:
        for k in K_VALUES:
            key = f"{metric}@{k}"
            a = aggregate[key]
            line = f"{key:<16} {a['mean']:>8.4f} {a['std']:>8.4f} {a['median']:>8.4f}"
            if baseline and key in baseline.get("aggregate", {}):
                bm = baseline["aggregate"][key]["mean"]
                delta = a["mean"] - bm
                line += f" {bm:>10.4f} {delta:>+10.4f}"
            elif baseline and f"recall@{k}" == key:
                # baseline only has recall
                bkey = f"recall@{k}"
                if bkey in baseline.get("aggregate", {}):
                    bm = baseline["aggregate"][bkey]["mean"]
                    delta = a["mean"] - bm
                    line += f" {bm:>10.4f} {delta:>+10.4f}"
            print(line)


if __name__ == "__main__":
    main()
