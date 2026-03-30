import numpy as np
from sklearn.cluster import KMeans
from typing import Dict


def ads_pool_stats(
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    user_emb: np.ndarray,
    n_clusters: int = 5,
) -> Dict:
    """Compute summary statistics for the candidate ad pool.

    Args:
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        user_emb: User embedding vector, shape [D].
        n_clusters: Number of clusters for distribution analysis.

    Returns:
        Dict with total_ads, similarity_stats, and cluster_distribution.
    """
    total_ads = len(ad_ids)

    # Cosine similarities
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
    similarities = ad_norms @ user_norm

    similarity_stats = {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "p25": float(np.percentile(similarities, 25)),
        "p50": float(np.percentile(similarities, 50)),
        "p75": float(np.percentile(similarities, 75)),
        "p95": float(np.percentile(similarities, 95)),
    }

    # Cluster distribution
    n_clusters = min(n_clusters, total_ads)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ad_embs)

    cluster_distribution = []
    for cid in range(n_clusters):
        c_mask = cluster_labels == cid
        c_sims = similarities[c_mask]
        cluster_distribution.append({
            "cluster_id": cid,
            "size": int(c_mask.sum()),
            "avg_similarity": float(c_sims.mean()) if c_mask.sum() > 0 else 0.0,
        })

    return {
        "total_ads": total_ads,
        "similarity_stats": similarity_stats,
        "cluster_distribution": cluster_distribution,
    }
