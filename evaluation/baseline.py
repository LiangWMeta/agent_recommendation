#!/usr/bin/env python3
"""Dot-product (cosine similarity) baseline for agent recommendation evaluation.

For each request, ranks ads by cosine(user_emb, ad_emb) and computes recall@K.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


K_VALUES = [10, 20, 50, 100]


def cosine_similarity(user_emb: np.ndarray, ad_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between user embedding and each ad embedding."""
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-12)
    ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-12)
    return ad_norms @ user_norm


def compute_recall_at_k(ranked_indices: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute recall@K: |top_K ∩ positives| / |positives|."""
    n_positives = labels.sum()
    if n_positives == 0:
        return 0.0
    top_k = ranked_indices[:k]
    hits = labels[top_k].sum()
    return float(hits / n_positives)


def evaluate_request(npz_path: str) -> dict:
    """Evaluate a single request and return per-request metrics."""
    data = np.load(npz_path)
    user_emb = data["user_emb"]
    ad_embs = data["ad_embs"]
    labels = data["labels"]
    ad_ids = data["ad_ids"]
    request_id = int(data["request_id"])

    scores = cosine_similarity(user_emb, ad_embs)
    ranked_indices = np.argsort(-scores)

    n_positives = int(labels.sum())
    n_total = len(labels)

    result = {
        "request_id": request_id,
        "n_candidates": n_total,
        "n_positives": n_positives,
    }
    for k in K_VALUES:
        result[f"recall@{k}"] = compute_recall_at_k(ranked_indices, labels, k)

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
    parser = argparse.ArgumentParser(description="Cosine similarity baseline evaluation")
    parser.add_argument("--data-dir", default="data/", help="Directory with .npz files")
    parser.add_argument("--output", default="evaluation/results/baseline.json",
                        help="Output JSON path")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Max requests to evaluate (for testing)")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"ERROR: No .npz files found in {args.data_dir}")
        sys.exit(1)

    if args.max_requests:
        npz_files = npz_files[:args.max_requests]

    print(f"Evaluating cosine baseline on {len(npz_files)} requests...")
    t0 = time.time()

    per_request = []
    for i, path in enumerate(npz_files):
        result = evaluate_request(path)
        per_request.append(result)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(npz_files)}")

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s\n")

    # Aggregate metrics
    aggregate = {}
    for k in K_VALUES:
        key = f"recall@{k}"
        values = np.array([r[key] for r in per_request])
        ci_lo, ci_hi = bootstrap_ci(values)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "ci_95": [ci_lo, ci_hi],
        }

    output = {
        "method": "cosine_baseline",
        "n_requests": len(per_request),
        "aggregate": aggregate,
        "per_request": per_request,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}\n")

    # Print summary table
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Median':>8} {'95% CI':>20}")
    print("-" * 60)
    for k in K_VALUES:
        key = f"recall@{k}"
        a = aggregate[key]
        ci_str = f"[{a['ci_95'][0]:.4f}, {a['ci_95'][1]:.4f}]"
        print(f"{key:<12} {a['mean']:>8.4f} {a['std']:>8.4f} {a['median']:>8.4f} {ci_str:>20}")


if __name__ == "__main__":
    main()
