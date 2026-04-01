import numpy as np
from typing import Dict, Optional


def pselect_main_route(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    top_k: int = 100,
    threshold: Optional[float] = None,
) -> Dict:
    """Search ads by cosine similarity to user embedding.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        top_k: Number of top results to return.
        threshold: Optional minimum cosine similarity threshold.

    Returns:
        Dict with "results" (list of {ad_id, score}), "score_range", "score_std", "top_bottom_gap".
    """
    # Normalize
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)

    # Cosine similarities
    scores = ad_norms @ user_norm  # shape [N]

    # Apply threshold
    if threshold is not None:
        mask = scores >= threshold
        scores = scores[mask]
        filtered_ids = ad_ids[mask]
    else:
        filtered_ids = ad_ids

    # Top-k
    k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [
        {"ad_id": int(filtered_ids[i]), "score": float(scores[i])}
        for i in top_indices
    ]

    # Self-diagnostic metadata
    result_scores = np.array([r["score"] for r in results]) if results else np.array([])
    if len(result_scores) > 0:
        score_range = [float(result_scores.min()), float(result_scores.max())]
        score_std = float(result_scores.std())
        n_scores = len(result_scores)
        top_10 = np.sort(result_scores)[-min(10, n_scores):]
        bottom_10 = np.sort(result_scores)[:min(10, n_scores)]
        top_bottom_gap = float(top_10.mean() - bottom_10.mean())
    else:
        score_range = [0.0, 0.0]
        score_std = 0.0
        top_bottom_gap = 0.0

    return {
        "results": results,
        "score_range": score_range,
        "score_std": score_std,
        "top_bottom_gap": top_bottom_gap,
    }
