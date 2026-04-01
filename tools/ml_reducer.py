import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.cluster import KMeans


def ml_reducer(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    candidate_ad_ids: Optional[List[int]] = None,
    target_stage: str = "PM",
    reduction_rate: float = 0.5,
    method: str = "ml_value",
    prod_data_dir: str = "data_enriched",
    request_id: int = None,
) -> Dict:
    """Simulate ML-driven truncation (ML Reducer/Truncator).

    Replaces heuristic bottom-X% removal in the production ads retrieval
    pipeline with an ML-informed scoring and truncation step.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N].
        candidate_ad_ids: If provided, restrict to these ad IDs only.
        target_stage: Which pipeline stage to simulate ("PM" or "AI").
        reduction_rate: Fraction of candidates to remove (0.0 to 1.0).
        method: Scoring method — "ml_value", "heuristic_cosine", or "heuristic_random".
        prod_data_dir: Directory containing prod prediction sidecar files.
        request_id: Request ID for loading prod predictions.

    Returns:
        Dict with survived ads, reduction stats, value preservation, and score stats.
    """
    # --- Filter to candidate_ad_ids if provided ---
    if candidate_ad_ids is not None:
        candidate_set = set(candidate_ad_ids)
        mask = np.array([int(aid) in candidate_set for aid in ad_ids])
        ad_embs = ad_embs[mask]
        ad_ids = ad_ids[mask]
        labels = labels[mask]

    n_input = len(ad_ids)

    # Edge case: no candidates
    if n_input == 0:
        return {
            "survived": [],
            "n_input": 0,
            "n_survived": 0,
            "n_removed": 0,
            "reduction_rate": reduction_rate,
            "method": method,
            "target_stage": target_stage,
            "value_preservation": {
                "total_value_kept": 0.0,
                "positive_preservation": 0.0,
                "n_positives_kept": 0,
                "n_positives_total": 0,
            },
            "score_stats": {
                "survivor_mean": 0.0,
                "survivor_min": 0.0,
                "removed_mean": 0.0,
                "removed_max": 0.0,
                "threshold": 0.0,
            },
        }

    # --- Compute cosine similarities ---
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
    cosine_scores = ad_norms @ user_norm

    # --- Scoring ---
    if method == "heuristic_cosine":
        scores = cosine_scores

    elif method == "heuristic_random":
        scores = np.random.rand(n_input)

    elif method == "ml_value":
        # Cluster engagement rate via KMeans
        n_clusters = min(5, n_input)
        if n_clusters < 2:
            # Cannot cluster with fewer than 2 points; use uniform engagement rate
            cluster_eng_rates = np.array([float(labels.mean()) if n_input > 0 else 0.0] * n_input)
        else:
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
            cluster_labels = kmeans.fit_predict(ad_embs)
            # Compute per-cluster engagement rate
            cluster_eng_map = {}
            for c in range(n_clusters):
                c_mask = cluster_labels == c
                c_labels = labels[c_mask]
                cluster_eng_map[c] = float(c_labels.mean()) if len(c_labels) > 0 else 0.0
            cluster_eng_rates = np.array([cluster_eng_map[c] for c in cluster_labels])

        # Try to load prod predictions
        prod_preds = None
        if request_id is not None:
            prod_path = Path(prod_data_dir) / f"{request_id}_prod.json"
            if prod_path.exists():
                with open(prod_path, "r") as f:
                    prod_data = json.load(f)
                prod_map = {int(item["ad_id"]): float(item["prod_prediction"])
                            for item in prod_data if item.get("prod_prediction") is not None}
                prod_preds = np.array([prod_map.get(int(aid), -1.0) for aid in ad_ids])
                # Check if all predictions were found
                if np.all(prod_preds < 0):
                    prod_preds = None
                else:
                    # Replace missing with mean of available
                    valid_mask = prod_preds >= 0
                    if valid_mask.any():
                        prod_preds[~valid_mask] = prod_preds[valid_mask].mean()

        if prod_preds is not None:
            scores = 0.4 * cosine_scores + 0.3 * cluster_eng_rates + 0.3 * prod_preds
        else:
            scores = 0.5 * cosine_scores + 0.5 * cluster_eng_rates

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ml_value', 'heuristic_cosine', or 'heuristic_random'.")

    # --- Sort and truncate ---
    sorted_indices = np.argsort(scores)[::-1]
    n_survived = max(1, int(np.ceil(n_input * (1.0 - reduction_rate))))
    survivor_indices = sorted_indices[:n_survived]
    removed_indices = sorted_indices[n_survived:]

    survivor_scores = scores[survivor_indices]
    removed_scores = scores[removed_indices] if len(removed_indices) > 0 else np.array([])

    # --- Value preservation ---
    total_score_sum = float(scores.sum())
    survivor_score_sum = float(survivor_scores.sum())
    total_value_kept = survivor_score_sum / total_score_sum if total_score_sum > 0 else 0.0

    n_positives_total = int(labels.sum())
    survivor_labels = labels[survivor_indices]
    n_positives_kept = int(survivor_labels.sum())
    positive_preservation = n_positives_kept / n_positives_total if n_positives_total > 0 else 0.0

    # --- Score stats ---
    threshold = float(survivor_scores[-1]) if len(survivor_scores) > 0 else 0.0

    score_stats = {
        "survivor_mean": float(survivor_scores.mean()) if len(survivor_scores) > 0 else 0.0,
        "survivor_min": float(survivor_scores.min()) if len(survivor_scores) > 0 else 0.0,
        "removed_mean": float(removed_scores.mean()) if len(removed_scores) > 0 else 0.0,
        "removed_max": float(removed_scores.max()) if len(removed_scores) > 0 else 0.0,
        "threshold": threshold,
    }

    # --- Build survived list (top 50 only) ---
    top_n = min(50, len(survivor_indices))
    survived = [
        {
            "ad_id": int(ad_ids[survivor_indices[i]]),
            "predicted_value": float(scores[survivor_indices[i]]),
            "rank": i + 1,
        }
        for i in range(top_n)
    ]

    return {
        "survived": survived,
        "n_input": n_input,
        "n_survived": int(n_survived),
        "n_removed": int(n_input - n_survived),
        "reduction_rate": float(reduction_rate),
        "method": method,
        "target_stage": target_stage,
        "value_preservation": {
            "total_value_kept": float(total_value_kept),
            "positive_preservation": float(positive_preservation),
            "n_positives_kept": n_positives_kept,
            "n_positives_total": n_positives_total,
        },
        "score_stats": score_stats,
    }
