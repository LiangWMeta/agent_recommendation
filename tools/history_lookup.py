"""Historical learning system: find past requests with similar signal characteristics."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def lookup_similar_requests(
    similarity_gap: float,
    positive_rate: float,
    n_candidates: int,
    history_path: str = "evaluation/history.json",
) -> Dict[str, Any]:
    """Find past requests with similar signal characteristics.

    Args:
        similarity_gap: Gap between avg positive and negative cosine similarity.
        positive_rate: Fraction of candidates that are positive (engaged).
        n_candidates: Total number of candidate ads.
        history_path: Path to the history JSON file.

    Returns:
        Dict with 'similar_requests' (top 3 matches) and 'pattern_summary'.
    """
    history_file = Path(history_path)
    if not history_file.exists():
        return {
            "similar_requests": [],
            "pattern_summary": "No historical data yet.",
        }

    try:
        with open(history_file) as f:
            history = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "similar_requests": [],
            "pattern_summary": "No historical data yet.",
        }

    if not history:
        return {
            "similar_requests": [],
            "pattern_summary": "No historical data yet.",
        }

    # Filter entries where similarity_gap and positive_rate are within 2x of query
    def within_2x(query_val: float, hist_val: float) -> bool:
        if query_val == 0 and hist_val == 0:
            return True
        if query_val == 0 or hist_val == 0:
            # If one is zero and other isn't, use absolute threshold
            return abs(query_val - hist_val) < 0.01
        ratio = hist_val / query_val
        return 0.5 <= ratio <= 2.0

    candidates = []
    for entry in history:
        h_gap = entry.get("similarity_gap", 0)
        h_rate = entry.get("positive_rate", 0)
        if within_2x(similarity_gap, h_gap) and within_2x(positive_rate, h_rate):
            candidates.append(entry)

    # Normalize features for distance computation
    # Use all history entries to compute normalization ranges
    all_gaps = [e.get("similarity_gap", 0) for e in history]
    all_rates = [e.get("positive_rate", 0) for e in history]
    all_ncands = [e.get("n_candidates", 0) for e in history]

    gap_range = max(all_gaps) - min(all_gaps) if len(all_gaps) > 1 else 1.0
    rate_range = max(all_rates) - min(all_rates) if len(all_rates) > 1 else 1.0
    ncand_range = max(all_ncands) - min(all_ncands) if len(all_ncands) > 1 else 1.0

    gap_range = gap_range if gap_range > 0 else 1.0
    rate_range = rate_range if rate_range > 0 else 1.0
    ncand_range = ncand_range if ncand_range > 0 else 1.0

    def euclidean_distance(entry: Dict) -> float:
        d_gap = (similarity_gap - entry.get("similarity_gap", 0)) / gap_range
        d_rate = (positive_rate - entry.get("positive_rate", 0)) / rate_range
        d_ncand = (n_candidates - entry.get("n_candidates", 0)) / ncand_range
        return math.sqrt(d_gap ** 2 + d_rate ** 2 + d_ncand ** 2)

    # Sort candidates by distance
    candidates.sort(key=euclidean_distance)
    top_3 = candidates[:3]

    similar_requests = []
    for entry in top_3:
        similar_requests.append({
            "request_id": entry.get("request_id"),
            "run_id": entry.get("run_id"),
            "similarity_gap": entry.get("similarity_gap"),
            "positive_rate": entry.get("positive_rate"),
            "n_candidates": entry.get("n_candidates"),
            "strategy": entry.get("strategy"),
            "recall_at_50": entry.get("recall_at_50"),
            "recall_at_100": entry.get("recall_at_100"),
            "distance": round(euclidean_distance(entry), 4),
        })

    # Pattern summary: group by gap buckets and report avg recall
    buckets = {
        "large_gap (>0.05)": [],
        "medium_gap (0.01-0.05)": [],
        "small_gap (<0.01)": [],
    }
    for entry in history:
        gap = entry.get("similarity_gap", 0)
        r100 = entry.get("recall_at_100")
        if r100 is None:
            continue
        if gap > 0.05:
            buckets["large_gap (>0.05)"].append(r100)
        elif gap >= 0.01:
            buckets["medium_gap (0.01-0.05)"].append(r100)
        else:
            buckets["small_gap (<0.01)"].append(r100)

    summary_parts = []
    for bucket_name, recalls in buckets.items():
        if recalls:
            avg = sum(recalls) / len(recalls)
            summary_parts.append(f"{bucket_name}: avg recall@100={avg:.4f} (n={len(recalls)})")
        else:
            summary_parts.append(f"{bucket_name}: no data")

    pattern_summary = "; ".join(summary_parts)

    return {
        "similar_requests": similar_requests,
        "pattern_summary": pattern_summary,
    }
