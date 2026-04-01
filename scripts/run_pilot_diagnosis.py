#!/usr/bin/env python3
"""Pilot diagnosis: exercise all pipeline tools and produce a diagnostic report.

Usage:
    python3 scripts/run_pilot_diagnosis.py [--max-requests 20] [--data-dir data/local/model/split] [--output-dir outputs/pilot_diagnosis]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pipeline_simulator import pipeline_simulator
from tools.hsnn_cluster_scorer import hsnn_cluster_scorer
from tools.ml_reducer import ml_reducer
from tools.parallel_routes_blender import parallel_routes_blender
from tools.pselect_main_route import pselect_main_route
from tools.forced_retrieval import forced_retrieval
from tools.prod_model_ranker import prod_model_ranker
from tools.anti_negative_scorer import anti_negative_scorer
from tools.cluster_explorer import cluster_explorer
from tools.similar_ads import similar_ads_lookup
from evaluation.evaluate import evaluate_request


def load_request(npz_path):
    """Load a request from an NPZ file, using history_labels to avoid leakage."""
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
        rd["has_split"] = True
    else:
        rd["labels"] = data["labels"]
        rd["test_labels"] = data["labels"]
        rd["has_split"] = False
    return rd


# ---------------------------------------------------------------------------
# Q1: Stage Drop-Off Analysis
# ---------------------------------------------------------------------------
def run_q1(rd):
    """Run pipeline_simulator(stage='all') and collect per-stage positive survival rates."""
    result = pipeline_simulator(
        user_emb=rd["user_emb"],
        ad_embs=rd["ad_embs"],
        ad_ids=rd["ad_ids"],
        labels=rd["labels"],
        stage="all",
        request_id=rd["request_id"],
    )
    stages = result["stages"]
    total_pos = stages.get("AP", {}).get("n_positives", 0)

    survival = {}
    survival["AP"] = 1.0  # all positives present at AP stage
    for stage_name in ["PM", "AI", "AF"]:
        if stage_name in stages:
            rate = stages[stage_name].get("positive_survival_rate", 0.0)
            survival[stage_name] = rate
    return survival


# ---------------------------------------------------------------------------
# Q2: Route Uniqueness Analysis
# ---------------------------------------------------------------------------
def run_q2(rd):
    """Run 4 production routes, blend them, and collect route statistics."""
    emb_result = pselect_main_route(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], top_k=100,
    )
    fr_result = forced_retrieval(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=100,
    )
    hsnn_result = hsnn_cluster_scorer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=100,
    )
    pm_result = prod_model_ranker(
        rd["ad_ids"], top_k=100, request_id=rd["request_id"],
    )

    route_results = {
        "embedding": [r["ad_id"] for r in emb_result.get("results", [])],
        "fr_centroid": [r["ad_id"] for r in fr_result.get("results", [])],
        "hsnn": [r["ad_id"] for r in hsnn_result.get("results", [])],
    }
    if pm_result.get("available", False):
        route_results["prod_model"] = [r["ad_id"] for r in pm_result.get("results", [])]

    blend = parallel_routes_blender(
        user_emb=rd["user_emb"],
        ad_embs=rd["ad_embs"],
        ad_ids=rd["ad_ids"],
        labels=rd["labels"],
        route_results=route_results,
    )
    return blend.get("route_statistics", {})


# ---------------------------------------------------------------------------
# Q3: ML Reducer vs Heuristic Truncation
# ---------------------------------------------------------------------------
def run_q3(rd):
    """Compare ml_value vs heuristic_cosine at 50% reduction."""
    ml_result = ml_reducer(
        user_emb=rd["user_emb"],
        ad_embs=rd["ad_embs"],
        ad_ids=rd["ad_ids"],
        labels=rd["labels"],
        reduction_rate=0.5,
        method="ml_value",
        request_id=rd["request_id"],
    )
    heuristic_result = ml_reducer(
        user_emb=rd["user_emb"],
        ad_embs=rd["ad_embs"],
        ad_ids=rd["ad_ids"],
        labels=rd["labels"],
        reduction_rate=0.5,
        method="heuristic_cosine",
        request_id=rd["request_id"],
    )
    return {
        "ml_value": ml_result.get("value_preservation", {}),
        "heuristic_cosine": heuristic_result.get("value_preservation", {}),
    }


# ---------------------------------------------------------------------------
# Q4: HSNN Exploration Budget
# ---------------------------------------------------------------------------
def run_q4(rd):
    """Run hsnn_cluster_scorer with varying expand_top_k_coarse, measure recall."""
    budgets = [2, 3, 5, 8]
    results = []
    for budget in budgets:
        hsnn_result = hsnn_cluster_scorer(
            user_emb=rd["user_emb"],
            ad_embs=rd["ad_embs"],
            ad_ids=rd["ad_ids"],
            labels=rd["labels"],
            expand_top_k_coarse=budget,
            top_k=100,
        )
        ranked_ids = [r["ad_id"] for r in hsnn_result.get("results", [])]
        eval_result = evaluate_request(ranked_ids, rd["ad_ids"], rd["test_labels"])
        results.append({
            "expand_top_k": budget,
            "recall_at_100": eval_result.get("recall@100", 0.0),
            "computational_savings": hsnn_result.get("computational_savings", 0.0),
        })
    return results


# ---------------------------------------------------------------------------
# Q5: Production vs Exploration Route Value
# ---------------------------------------------------------------------------
def run_q5(rd):
    """Measure recall@100 as exploration routes are incrementally added."""
    # Production routes
    emb_result = pselect_main_route(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], top_k=100,
    )
    fr_result = forced_retrieval(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=100,
    )
    hsnn_result = hsnn_cluster_scorer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=100,
    )
    pm_result = prod_model_ranker(
        rd["ad_ids"], top_k=100, request_id=rd["request_id"],
    )

    prod_routes = {
        "embedding": [r["ad_id"] for r in emb_result.get("results", [])],
        "fr_centroid": [r["ad_id"] for r in fr_result.get("results", [])],
        "hsnn": [r["ad_id"] for r in hsnn_result.get("results", [])],
    }
    if pm_result.get("available", False):
        prod_routes["prod_model"] = [r["ad_id"] for r in pm_result.get("results", [])]

    # Production-only blend
    prod_blend = parallel_routes_blender(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
        route_results=dict(prod_routes),
    )
    prod_ranked = [r["ad_id"] for r in prod_blend.get("blended_results", [])]
    prod_eval = evaluate_request(prod_ranked, rd["ad_ids"], rd["test_labels"])
    prod_recall = prod_eval.get("recall@100", 0.0)

    results = [{"config": "prod_only", "recall_at_100": prod_recall}]

    # Exploration routes to add incrementally
    # anti_negative_scorer
    an_result = anti_negative_scorer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=100,
    )
    an_ids = [r["ad_id"] for r in an_result.get("results", [])]

    # cluster_explorer
    cl_result = cluster_explorer(
        rd["ad_embs"], rd["ad_ids"], n_clusters=5, top_k_per_cluster=20,
        labels=rd["labels"],
    )
    cl_ids = [r["ad_id"] for r in cl_result.get("ads", [])]

    # similar_ads_lookup: use top 5 positive ad_ids as references
    pos_mask = rd["labels"] == 1
    pos_ad_ids = rd["ad_ids"][pos_mask].tolist()
    ref_ids = [int(x) for x in pos_ad_ids[:5]]
    if ref_ids:
        sal_result = similar_ads_lookup(
            rd["ad_embs"], rd["ad_ids"], reference_ad_ids=ref_ids, top_k_per_ref=20,
        )
        sal_ids = []
        for group in sal_result:
            for sim in group.get("similar_ads", []):
                sal_ids.append(sim["ad_id"])
    else:
        sal_ids = []

    exploration_routes = [
        ("anti_negative", an_ids),
        ("cluster_explorer", cl_ids),
        ("similar_ads", sal_ids),
    ]

    current_routes = dict(prod_routes)
    for route_name, route_ids in exploration_routes:
        current_routes[route_name] = route_ids
        blend = parallel_routes_blender(
            rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
            route_results=dict(current_routes),
        )
        ranked = [r["ad_id"] for r in blend.get("blended_results", [])]
        ev = evaluate_request(ranked, rd["ad_ids"], rd["test_labels"])
        results.append({
            "config": f"prod + {route_name}",
            "recall_at_100": ev.get("recall@100", 0.0),
        })

    return results, prod_recall


# ---------------------------------------------------------------------------
# Aggregation and report generation
# ---------------------------------------------------------------------------
def aggregate_q1(all_q1):
    """Aggregate per-stage survival rates across requests."""
    stages = ["AP", "PM", "AI", "AF"]
    agg = {}
    for s in stages:
        vals = [q.get(s) for q in all_q1 if q.get(s) is not None]
        if vals:
            agg[s] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        else:
            agg[s] = {"mean": 0.0, "std": 0.0}
    return agg


def aggregate_q2(all_q2):
    """Aggregate route statistics across requests."""
    route_names = set()
    for q in all_q2:
        route_names.update(q.keys())

    agg = {}
    for rname in sorted(route_names):
        candidates = [q[rname]["n_candidates"] for q in all_q2 if rname in q]
        unique = [q[rname]["n_unique"] for q in all_q2 if rname in q]
        unique_pos = [q[rname]["n_unique_positives"] for q in all_q2 if rname in q]
        overlap = [q[rname]["overlap_with_other_routes"] for q in all_q2 if rname in q]
        agg[rname] = {
            "avg_candidates": float(np.mean(candidates)) if candidates else 0.0,
            "avg_unique": float(np.mean(unique)) if unique else 0.0,
            "avg_unique_positives": float(np.mean(unique_pos)) if unique_pos else 0.0,
            "avg_overlap": float(np.mean(overlap)) if overlap else 0.0,
        }
    return agg


def aggregate_q3(all_q3):
    """Aggregate ML reducer comparison across requests."""
    agg = {}
    for method in ["ml_value", "heuristic_cosine"]:
        vals_kept = [q[method].get("total_value_kept", 0.0) for q in all_q3]
        pos_pres = [q[method].get("positive_preservation", 0.0) for q in all_q3]
        agg[method] = {
            "avg_value_preserved": float(np.mean(vals_kept)),
            "avg_positive_preservation": float(np.mean(pos_pres)),
        }
    return agg


def aggregate_q4(all_q4):
    """Aggregate HSNN budget sweep across requests."""
    budgets = [2, 3, 5, 8]
    agg = {}
    for budget in budgets:
        recalls = []
        savings = []
        for q in all_q4:
            for entry in q:
                if entry["expand_top_k"] == budget:
                    recalls.append(entry["recall_at_100"])
                    savings.append(entry["computational_savings"])
        agg[budget] = {
            "avg_recall_at_100": float(np.mean(recalls)) if recalls else 0.0,
            "avg_compute_savings": float(np.mean(savings)) if savings else 0.0,
        }
    return agg


def aggregate_q5(all_q5):
    """Aggregate production vs exploration route value across requests."""
    # Collect all config names in order
    config_names = []
    if all_q5:
        for entry in all_q5[0][0]:
            config_names.append(entry["config"])

    agg = {}
    prod_recalls = [q[1] for q in all_q5]
    avg_prod_recall = float(np.mean(prod_recalls)) if prod_recalls else 0.0

    for config in config_names:
        recalls = []
        for q_results, _ in all_q5:
            for entry in q_results:
                if entry["config"] == config:
                    recalls.append(entry["recall_at_100"])
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        delta = avg_recall - avg_prod_recall
        agg[config] = {
            "avg_recall_at_100": avg_recall,
            "delta_vs_prod_only": delta,
        }
    return agg


def write_report(output_dir, agg_q1, agg_q2, agg_q3, agg_q4, agg_q5,
                 n_requests, data_dir, has_split):
    """Write the markdown diagnostic report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_note = ""
    if not has_split:
        split_note = "\n> **Note**: No train/test split found. Using `labels` for both tool input and evaluation.\n"

    lines = []
    lines.append("# Pilot Diagnosis Report\n")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Requests analyzed: {n_requests}")
    lines.append(f"Data source: {data_dir}")
    lines.append(split_note)

    # Q1
    lines.append("## Q1: Stage Drop-Off Analysis\n")
    lines.append("| Stage | Avg Positive Survival Rate | Std |")
    lines.append("|-------|---------------------------|-----|")
    for stage in ["AP", "PM", "AI", "AF"]:
        if stage in agg_q1:
            m = agg_q1[stage]["mean"]
            s = agg_q1[stage]["std"]
            lines.append(f"| {stage} | {m:.4f} | {s:.4f} |")
    # Key finding
    if "AF" in agg_q1 and "AP" in agg_q1:
        final = agg_q1["AF"]["mean"]
        lines.append(f"\n**Key finding**: On average, {final:.1%} of positive ads survive the full AP->AF pipeline.\n")

    # Q2
    lines.append("## Q2: Route Uniqueness Analysis\n")
    lines.append("| Route | Avg Candidates | Avg Unique | Avg Unique Positives | Avg Overlap |")
    lines.append("|-------|---------------|------------|---------------------|-------------|")
    for rname, stats in agg_q2.items():
        lines.append(
            f"| {rname} | {stats['avg_candidates']:.1f} | {stats['avg_unique']:.1f} "
            f"| {stats['avg_unique_positives']:.2f} | {stats['avg_overlap']:.4f} |"
        )
    # Key finding
    if agg_q2:
        max_unique = max(agg_q2.values(), key=lambda x: x["avg_unique"])
        max_route = [k for k, v in agg_q2.items() if v is max_unique][0]
        lines.append(f"\n**Key finding**: Route `{max_route}` contributes the most unique candidates ({max_unique['avg_unique']:.1f} avg).\n")

    # Q3
    lines.append("## Q3: ML Reducer vs Heuristic Truncation\n")
    lines.append("| Method | Avg Value Preserved | Avg Positive Preservation |")
    lines.append("|--------|--------------------|--------------------------:|")
    for method in ["ml_value", "heuristic_cosine"]:
        if method in agg_q3:
            vp = agg_q3[method]["avg_value_preserved"]
            pp = agg_q3[method]["avg_positive_preservation"]
            lines.append(f"| {method} | {vp:.4f} | {pp:.4f} |")
    # Key finding
    if "ml_value" in agg_q3 and "heuristic_cosine" in agg_q3:
        ml_pp = agg_q3["ml_value"]["avg_positive_preservation"]
        h_pp = agg_q3["heuristic_cosine"]["avg_positive_preservation"]
        diff = ml_pp - h_pp
        better = "ml_value" if diff > 0 else "heuristic_cosine"
        lines.append(f"\n**Key finding**: `{better}` preserves {abs(diff):.1%} more positives at 50% reduction rate.\n")

    # Q4
    lines.append("## Q4: HSNN Exploration Budget\n")
    lines.append("| expand_top_k | Avg Recall@100 | Avg Compute Savings |")
    lines.append("|-------------|---------------|--------------------:|")
    for budget in [2, 3, 5, 8]:
        if budget in agg_q4:
            r = agg_q4[budget]["avg_recall_at_100"]
            s = agg_q4[budget]["avg_compute_savings"]
            lines.append(f"| {budget} | {r:.4f} | {s:.4f} |")
    # Key finding
    if 2 in agg_q4 and 8 in agg_q4:
        r2 = agg_q4[2]["avg_recall_at_100"]
        r8 = agg_q4[8]["avg_recall_at_100"]
        s2 = agg_q4[2]["avg_compute_savings"]
        s8 = agg_q4[8]["avg_compute_savings"]
        lines.append(
            f"\n**Key finding**: Expanding from 2 to 8 coarse clusters improves recall by "
            f"{r8 - r2:+.4f} but reduces compute savings from {s2:.1%} to {s8:.1%}.\n"
        )

    # Q5
    lines.append("## Q5: Production vs Exploration Route Value\n")
    lines.append("| Configuration | Avg Recall@100 | Delta vs Prod-Only |")
    lines.append("|--------------|---------------|--------------------|")
    for config, stats in agg_q5.items():
        r = stats["avg_recall_at_100"]
        d = stats["delta_vs_prod_only"]
        lines.append(f"| {config} | {r:.4f} | {d:+.4f} |")
    # Key finding
    if agg_q5:
        configs = list(agg_q5.keys())
        if len(configs) > 1:
            full_config = configs[-1]
            full_delta = agg_q5[full_config]["delta_vs_prod_only"]
            lines.append(
                f"\n**Key finding**: Adding all exploration routes changes recall@100 by "
                f"{full_delta:+.4f} compared to production-only blend.\n"
            )

    # Recommendations
    lines.append("## Actionable Recommendations\n")
    recs = []

    # From Q1
    if "AF" in agg_q1:
        af_surv = agg_q1["AF"]["mean"]
        if af_surv < 0.5:
            recs.append(f"- **Pipeline drop-off is severe** ({af_surv:.1%} positive survival). Consider widening PM/AI budgets or adding recall-optimized routes before PM.")
        else:
            recs.append(f"- Pipeline positive survival is reasonable ({af_surv:.1%}). Focus on ranking quality rather than recall recovery.")

    # From Q3
    if "ml_value" in agg_q3 and "heuristic_cosine" in agg_q3:
        ml_pp = agg_q3["ml_value"]["avg_positive_preservation"]
        h_pp = agg_q3["heuristic_cosine"]["avg_positive_preservation"]
        if ml_pp > h_pp:
            recs.append(f"- **ML-based truncation outperforms heuristic** ({ml_pp:.1%} vs {h_pp:.1%} positive preservation). Prioritize ML Reducer deployment.")
        else:
            recs.append(f"- Heuristic cosine truncation is competitive ({h_pp:.1%} vs {ml_pp:.1%}). ML Reducer may not justify added complexity.")

    # From Q4
    if 3 in agg_q4 and 5 in agg_q4:
        r3 = agg_q4[3]["avg_recall_at_100"]
        r5 = agg_q4[5]["avg_recall_at_100"]
        s3 = agg_q4[3]["avg_compute_savings"]
        if r5 - r3 < 0.01:
            recs.append(f"- **HSNN expand_top_k=3 is sufficient**: expanding to 5 gains <1% recall while reducing compute savings from {s3:.1%}.")
        else:
            recs.append(f"- Consider HSNN expand_top_k=5 for {r5 - r3:.1%} recall gain at moderate compute cost.")

    # From Q5
    if agg_q5:
        configs = list(agg_q5.keys())
        if len(configs) > 1:
            full_delta = agg_q5[configs[-1]]["delta_vs_prod_only"]
            if full_delta > 0.01:
                recs.append(f"- **Exploration routes add value** ({full_delta:+.4f} recall). Include anti_negative, cluster_explorer, and similar_ads in production blend.")
            elif full_delta < -0.01:
                recs.append(f"- Exploration routes hurt recall ({full_delta:+.4f}). They may add diversity but dilute precision; consider lower blend weights.")
            else:
                recs.append(f"- Exploration routes have marginal recall impact ({full_delta:+.4f}). Their diversity value should be assessed via online experiments.")

    if not recs:
        recs.append("- No clear recommendations; consider increasing request sample size for more reliable estimates.")

    lines.extend(recs)
    lines.append("")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Pilot diagnosis: exercise all pipeline tools")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--data-dir", default="data/local/model/split")
    parser.add_argument("--output-dir", default="outputs/pilot_diagnosis")
    args = parser.parse_args()

    # Resolve data directory with fallback
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir = Path("data/local/model/raw")
        if not data_dir.exists():
            print(f"ERROR: Neither {args.data_dir} nor data/ directory found.", file=sys.stderr)
            sys.exit(1)
        print(f"Falling back to data directory: {data_dir}", file=sys.stderr)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    npz_files = sorted(data_dir.glob("request_*.npz"))[:args.max_requests]
    if not npz_files:
        print(f"ERROR: No request_*.npz files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    n_requests = len(npz_files)
    print(f"Pilot diagnosis: {n_requests} requests from {data_dir}", file=sys.stderr)

    all_q1 = []
    all_q2 = []
    all_q3 = []
    all_q4 = []
    all_q5 = []
    has_split = True

    for i, npz_path in enumerate(npz_files):
        rd = load_request(npz_path)
        if not rd["has_split"]:
            has_split = False

        print(
            f"Processing request {i + 1}/{n_requests}: {rd['request_id']}...",
            file=sys.stderr,
        )

        try:
            q1 = run_q1(rd)
            all_q1.append(q1)
        except Exception as e:
            print(f"  Q1 failed for {rd['request_id']}: {e}", file=sys.stderr)

        try:
            q2 = run_q2(rd)
            all_q2.append(q2)
        except Exception as e:
            print(f"  Q2 failed for {rd['request_id']}: {e}", file=sys.stderr)

        try:
            q3 = run_q3(rd)
            all_q3.append(q3)
        except Exception as e:
            print(f"  Q3 failed for {rd['request_id']}: {e}", file=sys.stderr)

        try:
            q4 = run_q4(rd)
            all_q4.append(q4)
        except Exception as e:
            print(f"  Q4 failed for {rd['request_id']}: {e}", file=sys.stderr)

        try:
            q5 = run_q5(rd)
            all_q5.append(q5)
        except Exception as e:
            print(f"  Q5 failed for {rd['request_id']}: {e}", file=sys.stderr)

    # Aggregate results
    print("Aggregating results...", file=sys.stderr)

    agg_q1 = aggregate_q1(all_q1) if all_q1 else {}
    agg_q2 = aggregate_q2(all_q2) if all_q2 else {}
    agg_q3 = aggregate_q3(all_q3) if all_q3 else {}
    agg_q4 = aggregate_q4(all_q4) if all_q4 else {}
    agg_q5 = aggregate_q5(all_q5) if all_q5 else {}

    # Save raw results as JSON
    raw_results = {
        "n_requests": n_requests,
        "data_dir": str(data_dir),
        "has_split": has_split,
        "q1_stage_dropoff": agg_q1,
        "q2_route_uniqueness": agg_q2,
        "q3_ml_vs_heuristic": agg_q3,
        "q4_hsnn_budget": {str(k): v for k, v in agg_q4.items()} if agg_q4 else {},
        "q5_prod_vs_exploration": agg_q5,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Raw results saved to {results_path}", file=sys.stderr)

    # Write markdown report
    report_path = write_report(
        output_dir, agg_q1, agg_q2, agg_q3, agg_q4, agg_q5,
        n_requests, str(data_dir), has_split,
    )
    print(f"Report saved to {report_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
