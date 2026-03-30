import numpy as np
from typing import Dict


def fr_centroid_search(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    top_k: int = 100,
) -> Dict:
    """Search using the centroid of positively-engaged ad embeddings.

    This simulates production Forced Retrieval (FR), which accounts for 85%
    of production impressions. The centroid provides a completely independent
    query vector from user_emb, retrieving different candidates.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N].
        top_k: Number of top results to return.

    Returns:
        Dict with results, centroid_gap, user_emb_gap, correlation, and diagnostics.
    """
    pos_mask = labels == 1
    pos_embs = ad_embs[pos_mask]
    centroid = pos_embs.mean(axis=0)
    # Normalize
    centroid_norm = centroid / np.linalg.norm(centroid).clip(1e-8)
    user_unit = user_emb / np.linalg.norm(user_emb).clip(1e-8)
    ad_norms_arr = np.linalg.norm(ad_embs, axis=1, keepdims=True).clip(1e-8)
    ad_units = ad_embs / ad_norms_arr

    # Score by cosine to centroid
    scores = ad_units @ centroid_norm
    top_idx = np.argsort(-scores)[:top_k]

    # Also compute user_emb scores for comparison
    user_scores = ad_units @ user_unit

    # Diagnostics
    centroid_gap = float(scores[pos_mask].mean() - scores[~pos_mask].mean())
    user_gap = float(user_scores[pos_mask].mean() - user_scores[~pos_mask].mean())
    correlation = float(np.corrcoef(scores, user_scores)[0, 1])

    results = [{"ad_id": int(ad_ids[i]), "score": float(scores[i])} for i in top_idx]

    return {
        "results": results,
        "centroid_gap": centroid_gap,
        "user_emb_gap": user_gap,
        "centroid_vs_user_correlation": correlation,
        "n_positives_used": int(pos_mask.sum()),
        "score_range": [float(scores[top_idx[-1]]), float(scores[top_idx[0]])],
    }
