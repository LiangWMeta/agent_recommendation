import numpy as np
from typing import Dict


def anti_negative_scorer(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.3,
    top_k: int = 100,
) -> Dict:
    """Score ads by: sim(ad, pos_centroid) - alpha * sim(ad, neg_centroid).

    This creates a directional signal that pushes ranking toward what
    the user engages with and away from what they ignore.

    Args:
        user_emb: User embedding vector, shape [D] (unused but kept for interface consistency).
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N].
        alpha: Weight for the negative centroid penalty.
        top_k: Number of top results to return.

    Returns:
        Dict with results, alpha, score_range, and centroid similarity.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_centroid = ad_embs[pos_mask].mean(axis=0)
    neg_centroid = ad_embs[neg_mask].mean(axis=0)

    pos_unit = pos_centroid / np.linalg.norm(pos_centroid).clip(1e-8)
    neg_unit = neg_centroid / np.linalg.norm(neg_centroid).clip(1e-8)

    ad_norms_arr = np.linalg.norm(ad_embs, axis=1, keepdims=True).clip(1e-8)
    ad_units = ad_embs / ad_norms_arr

    pos_scores = ad_units @ pos_unit
    neg_scores = ad_units @ neg_unit
    combined = pos_scores - alpha * neg_scores

    top_idx = np.argsort(-combined)[:top_k]
    results = [
        {
            "ad_id": int(ad_ids[i]),
            "score": float(combined[i]),
            "pos_score": float(pos_scores[i]),
            "neg_score": float(neg_scores[i]),
        }
        for i in top_idx
    ]

    return {
        "results": results,
        "alpha": alpha,
        "score_range": [float(combined[top_idx[-1]]), float(combined[top_idx[0]])],
        "pos_neg_centroid_similarity": float(pos_unit @ neg_unit),
    }
