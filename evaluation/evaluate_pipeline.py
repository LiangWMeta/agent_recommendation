#!/usr/bin/env python3
"""Pipeline-aware evaluation: per-stage recall and cross-stage consistency metrics."""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

# Import standard evaluation from evaluate.py
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_request, K_VALUES, bootstrap_ci


# Pipeline stage sizes
PM_SIZE = 500
AI_SIZE = 100
AF_SIZE = 20

# Truncation levels for robustness analysis
TRUNCATION_PCTS = [0, 5, 10, 15, 20, 30, 40, 50]


def _load_prod_predictions(prod_data_dir: str, request_id) -> dict:
    """Load prod predictions for a request. Returns {ad_id: prod_prediction} or None."""
    prod_path = os.path.join(prod_data_dir, f"{request_id}_prod.json")
    if not os.path.exists(prod_path):
        return None
    with open(prod_path) as f:
        prod_list = json.load(f)
    return {
        int(item["ad_id"]): float(item["prod_prediction"])
        for item in prod_list
        if item.get("prod_prediction") is not None
    }


def _cosine_scores(user_emb: np.ndarray, ad_embs: np.ndarray, ad_ids: np.ndarray) -> dict:
    """Compute cosine similarity between user embedding and each ad embedding.

    Returns {ad_id: cosine_similarity}.
    """
    user_norm = np.linalg.norm(user_emb)
    if user_norm == 0:
        return {int(aid): 0.0 for aid in ad_ids}
    ad_norms = np.linalg.norm(ad_embs, axis=1)
    # Avoid division by zero
    ad_norms = np.where(ad_norms == 0, 1e-8, ad_norms)
    cosines = ad_embs @ user_emb / (ad_norms * user_norm)
    return {int(aid): float(c) for aid, c in zip(ad_ids, cosines)}


def _get_scores(ranked_ad_ids: list, prod_preds: dict, cosine_scores: dict) -> dict:
    """Get scoring function for ads: prod_prediction if available, cosine fallback."""
    if prod_preds is not None:
        return {int(aid): prod_preds.get(int(aid), cosine_scores.get(int(aid), 0.0))
                for aid in ranked_ad_ids}
    return {int(aid): cosine_scores.get(int(aid), 0.0) for aid in ranked_ad_ids}


def evaluate_pipeline_request(
    ranked_ad_ids: list,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    prod_data_dir: str,
    request_id,
    user_emb: np.ndarray = None,
    ad_embs: np.ndarray = None,
) -> dict:
    """Evaluate a single request through the pipeline lens.

    Simulates PM -> AI -> AF pipeline stages and computes per-stage survival
    rates, cross-stage recall, and rank correlation with prod predictions.
    """
    # Standard recall metrics
    result = evaluate_request(ranked_ad_ids, ad_ids, labels)
    result["request_id"] = int(request_id)

    # Load prod predictions
    prod_preds = _load_prod_predictions(prod_data_dir, request_id)
    result["has_prod_predictions"] = prod_preds is not None

    # Compute cosine fallback scores
    cosine_scores = {}
    if user_emb is not None and ad_embs is not None:
        cosine_scores = _cosine_scores(user_emb, ad_embs, ad_ids)

    # Build label lookup
    id_to_label = dict(zip(ad_ids.tolist(), labels.tolist()))

    # Score lookup for all ranked ads
    scores = _get_scores(ranked_ad_ids, prod_preds, cosine_scores)

    # --- Stage 1: PM truncation ---
    # Take agent's top-500, then re-score by prod_prediction (or cosine), keep top 500
    agent_top_500 = [int(aid) for aid in ranked_ad_ids[:PM_SIZE]]
    pm_scored = sorted(agent_top_500, key=lambda aid: scores.get(aid, 0.0), reverse=True)
    pm_survivors = pm_scored[:PM_SIZE]

    # PM survival rate: fraction of agent's top-100 that survive PM
    agent_top_100 = set(int(aid) for aid in ranked_ad_ids[:100])
    pm_survivor_set = set(pm_survivors)
    pm_survival = len(agent_top_100 & pm_survivor_set) / max(len(agent_top_100), 1)
    result["pm_survival_rate"] = float(pm_survival)

    # --- Stage 2: AI ---
    # From PM survivors, keep top 100 by combined score
    # Combined score: 0.7 * prod_prediction + 0.3 * agent_rank_score
    agent_rank_map = {}
    for i, aid in enumerate(ranked_ad_ids):
        agent_rank_map[int(aid)] = 1.0 - (i / max(len(ranked_ad_ids), 1))

    def combined_score(aid):
        prod_s = scores.get(aid, 0.0)
        agent_s = agent_rank_map.get(aid, 0.0)
        return 0.7 * prod_s + 0.3 * agent_s

    ai_scored = sorted(pm_survivors, key=lambda aid: combined_score(aid), reverse=True)
    ai_survivors = ai_scored[:AI_SIZE]
    ai_survivor_set = set(ai_survivors)

    ai_survival = len(agent_top_100 & ai_survivor_set) / max(len(agent_top_100), 1)
    result["ai_survival_rate"] = float(ai_survival)

    # --- Stage 3: AF ---
    af_scored = sorted(ai_survivors, key=lambda aid: combined_score(aid), reverse=True)
    af_survivors = af_scored[:AF_SIZE]
    af_survivor_set = set(af_survivors)

    af_survival = len(agent_top_100 & af_survivor_set) / max(len(agent_top_100), 1)
    result["af_survival_rate"] = float(af_survival)

    # --- Cross-stage recall@K ---
    # Recall computed only among AF survivors: how many of the 20 AF survivors are positive
    af_positives = sum(1 for aid in af_survivors if id_to_label.get(aid, 0) == 1)
    n_positives = int(labels.sum())
    result["cross_stage_recall@20"] = float(af_positives / n_positives) if n_positives > 0 else 0.0
    result["af_positive_count"] = af_positives
    result["af_positive_rate"] = float(af_positives / AF_SIZE)

    # --- Spearman rank correlation ---
    # Between agent's ranking and prod_prediction ranking for overlapping ads
    if prod_preds is not None:
        overlap_aids = [int(aid) for aid in ranked_ad_ids if int(aid) in prod_preds]
        if len(overlap_aids) >= 3:
            agent_ranks = [ranked_ad_ids.index(aid) for aid in overlap_aids]
            prod_vals = [prod_preds[aid] for aid in overlap_aids]
            corr, pval = spearmanr(agent_ranks, prod_vals)
            # Note: negative correlation expected (lower rank index = higher agent rank,
            # higher prod_prediction = better), so we negate for interpretability
            result["spearman_correlation"] = float(-corr) if not np.isnan(corr) else 0.0
            result["spearman_pvalue"] = float(pval) if not np.isnan(pval) else 1.0
            result["spearman_n_overlap"] = len(overlap_aids)
        else:
            result["spearman_correlation"] = 0.0
            result["spearman_pvalue"] = 1.0
            result["spearman_n_overlap"] = len(overlap_aids)
    else:
        result["spearman_correlation"] = 0.0
        result["spearman_pvalue"] = 1.0
        result["spearman_n_overlap"] = 0

    return result


def evaluate_truncation_robustness(
    ranked_ad_ids: list,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    prod_data_dir: str,
    request_id,
    user_emb: np.ndarray = None,
    ad_embs: np.ndarray = None,
) -> list:
    """Measure how recall degrades at different truncation levels.

    For each truncation_pct, removes the bottom truncation_pct% of the pool
    by prod_prediction (or cosine) and computes recall@100 on remaining ads.
    """
    # Load prod predictions
    prod_preds = _load_prod_predictions(prod_data_dir, request_id)

    # Compute cosine fallback
    cosine_scores = {}
    if user_emb is not None and ad_embs is not None:
        cosine_scores = _cosine_scores(user_emb, ad_embs, ad_ids)

    # Build score for every ad in the pool
    pool_scores = {}
    for aid in ad_ids:
        aid_int = int(aid)
        if prod_preds is not None and aid_int in prod_preds:
            pool_scores[aid_int] = prod_preds[aid_int]
        else:
            pool_scores[aid_int] = cosine_scores.get(aid_int, 0.0)

    # Sort all pool ads by score ascending (worst first)
    sorted_pool = sorted(pool_scores.items(), key=lambda x: x[1])

    id_to_label = dict(zip(ad_ids.tolist(), labels.tolist()))
    n_positives = int(labels.sum())

    results = []
    for trunc_pct in TRUNCATION_PCTS:
        n_remove = int(len(sorted_pool) * trunc_pct / 100)
        removed_ids = set(aid for aid, _ in sorted_pool[:n_remove])
        surviving_pool = set(int(aid) for aid in ad_ids) - removed_ids

        # Filter agent's ranked list to surviving ads
        filtered_ranked = [int(aid) for aid in ranked_ad_ids if int(aid) in surviving_pool]

        # Compute recall@100 on filtered list
        top_100 = filtered_ranked[:100]
        hits = sum(1 for aid in top_100 if id_to_label.get(aid, 0) == 1)
        recall_100 = float(hits / n_positives) if n_positives > 0 else 0.0

        results.append({
            "truncation_pct": trunc_pct,
            "recall_at_100": recall_100,
            "n_remaining": len(surviving_pool),
        })

    return results


def _extract_request_id(output: dict, path: str):
    """Extract request_id from output JSON or filename."""
    request_id = output.get("request_id")
    if request_id is not None:
        return int(request_id)
    basename = os.path.basename(path).replace(".json", "")
    try:
        return int(basename)
    except ValueError:
        try:
            return int(basename.replace("request_", ""))
        except ValueError:
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline-aware evaluation: per-stage recall and consistency metrics"
    )
    parser.add_argument("--run-id", required=True, help="Run ID (subfolder under outputs/)")
    parser.add_argument("--data-dir", default="data/local/model/split", help="Directory with .npz files")
    parser.add_argument("--prod-dir", default="data/local/model/enriched", help="Directory with prod prediction JSONs")
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

    print(f"Pipeline evaluation for run '{args.run_id}' ({len(output_files)} outputs)...")
    t0 = time.time()

    per_request = []
    truncation_results = []
    skipped = 0

    for path in output_files:
        with open(path) as f:
            output = json.load(f)

        ranked_ads = output.get("ranked_ads", [])
        request_id = _extract_request_id(output, path)
        if request_id is None:
            continue  # Skip non-request files like summary.json

        if request_id not in gt_index:
            skipped += 1
            continue

        data = np.load(gt_index[request_id])
        ad_ids = data["ad_ids"]
        labels = data["labels"]
        user_emb = data["user_emb"] if "user_emb" in data else None
        ad_embs = data["ad_embs"] if "ad_embs" in data else None

        # Pipeline evaluation
        result = evaluate_pipeline_request(
            ranked_ads, ad_ids, labels, args.prod_dir, request_id,
            user_emb=user_emb, ad_embs=ad_embs,
        )
        per_request.append(result)

        # Truncation robustness
        trunc = evaluate_truncation_robustness(
            ranked_ads, ad_ids, labels, args.prod_dir, request_id,
            user_emb=user_emb, ad_embs=ad_embs,
        )
        truncation_results.append({
            "request_id": int(request_id),
            "truncation": trunc,
        })

    elapsed = time.time() - t0
    if skipped > 0:
        print(f"  skipped {skipped} outputs (no matching ground truth)")
    print(f"  evaluated {len(per_request)} requests in {elapsed:.1f}s\n")

    if not per_request:
        print("ERROR: No requests evaluated.")
        sys.exit(1)

    # Aggregate standard metrics
    standard_metrics = ["recall", "precision", "ndcg"]
    aggregate = {}
    for metric in standard_metrics:
        for k in K_VALUES:
            key = f"{metric}@{k}"
            values = np.array([r[key] for r in per_request])
            ci_lo, ci_hi = bootstrap_ci(values)
            aggregate[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "ci_95": [ci_lo, ci_hi],
            }

    # Aggregate pipeline metrics
    pipeline_keys = [
        "pm_survival_rate", "ai_survival_rate", "af_survival_rate",
        "cross_stage_recall@20", "af_positive_rate",
        "spearman_correlation",
    ]
    for key in pipeline_keys:
        values = np.array([r[key] for r in per_request])
        ci_lo, ci_hi = bootstrap_ci(values)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "ci_95": [ci_lo, ci_hi],
        }

    # Aggregate truncation robustness
    agg_truncation = []
    for trunc_pct in TRUNCATION_PCTS:
        recalls = []
        n_remaining_list = []
        for tr in truncation_results:
            for entry in tr["truncation"]:
                if entry["truncation_pct"] == trunc_pct:
                    recalls.append(entry["recall_at_100"])
                    n_remaining_list.append(entry["n_remaining"])
        recalls = np.array(recalls)
        agg_truncation.append({
            "truncation_pct": trunc_pct,
            "mean_recall_at_100": float(recalls.mean()),
            "std_recall_at_100": float(recalls.std()),
            "mean_n_remaining": float(np.mean(n_remaining_list)),
        })

    # Assemble output
    output_data = {
        "run_id": args.run_id,
        "n_requests": len(per_request),
        "aggregate": aggregate,
        "aggregate_truncation": agg_truncation,
        "per_request": per_request,
        "truncation_per_request": truncation_results,
    }

    # Save results
    result_path = os.path.join("evaluation", "results", f"{args.run_id}_pipeline.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {result_path}\n")

    # Print standard metrics
    print(f"{'Metric':<24} {'Mean':>8} {'Std':>8}")
    print("-" * 42)
    for metric in standard_metrics:
        for k in K_VALUES:
            key = f"{metric}@{k}"
            a = aggregate[key]
            print(f"{key:<24} {a['mean']:>8.4f} {a['std']:>8.4f}")

    # Print pipeline metrics
    print()
    print(f"{'Pipeline Metric':<24} {'Mean':>8} {'Std':>8}")
    print("-" * 42)
    for key in pipeline_keys:
        a = aggregate[key]
        print(f"{key:<24} {a['mean']:>8.4f} {a['std']:>8.4f}")

    # Print truncation robustness
    print()
    print(f"{'Trunc %':<10} {'Recall@100':>12} {'Std':>8} {'N Remaining':>14}")
    print("-" * 46)
    for entry in agg_truncation:
        print(
            f"{entry['truncation_pct']:<10} "
            f"{entry['mean_recall_at_100']:>12.4f} "
            f"{entry['std_recall_at_100']:>8.4f} "
            f"{entry['mean_n_remaining']:>14.0f}"
        )


if __name__ == "__main__":
    main()
