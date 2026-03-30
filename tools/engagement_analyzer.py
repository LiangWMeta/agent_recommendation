import numpy as np
from sklearn.cluster import KMeans
from typing import Dict


def engagement_pattern_analyzer(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = 5,
) -> Dict:
    """Analyze engagement patterns between positive and negative ads.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N]. 1=positive, 0=negative.
        n_clusters: Number of clusters for distribution analysis.

    Returns:
        Dict with engagement statistics and cluster-level breakdown.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    n_positive = int(pos_mask.sum())
    n_negative = int(neg_mask.sum())

    # Cosine similarities to user
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
    similarities = ad_norms @ user_norm

    avg_pos_sim = float(similarities[pos_mask].mean()) if n_positive > 0 else 0.0
    avg_neg_sim = float(similarities[neg_mask].mean()) if n_negative > 0 else 0.0

    # Cluster analysis
    n_clusters = min(n_clusters, len(ad_embs))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ad_embs)

    positive_clusters = []
    engagement_by_cluster = []

    for cid in range(n_clusters):
        c_mask = cluster_labels == cid
        c_total = int(c_mask.sum())
        c_pos = int((c_mask & pos_mask).sum())
        c_neg = int((c_mask & neg_mask).sum())
        c_rate = c_pos / c_total if c_total > 0 else 0.0

        cluster_info = {
            "cluster_id": cid,
            "total": c_total,
            "positive": c_pos,
            "negative": c_neg,
            "engagement_rate": float(c_rate),
            "avg_user_similarity": float(similarities[c_mask].mean()) if c_total > 0 else 0.0,
        }
        engagement_by_cluster.append(cluster_info)

        if c_pos > 0:
            positive_clusters.append(cid)

    # Self-diagnostic metadata
    similarity_gap = avg_pos_sim - avg_neg_sim
    gap_ratio = similarity_gap / abs(avg_neg_sim) if avg_neg_sim != 0 else 0.0
    positive_sim_std = float(similarities[pos_mask].std()) if n_positive > 1 else 0.0
    negative_sim_std = float(similarities[neg_mask].std()) if n_negative > 1 else 0.0
    overlap_fraction = float((similarities[neg_mask] > avg_pos_sim).mean()) if n_negative > 0 else 0.0

    # Top 10 positive ad_ids by similarity to user
    if n_positive > 0:
        pos_indices = np.where(pos_mask)[0]
        pos_sims = similarities[pos_indices]
        k = min(10, len(pos_indices))
        top_pos_idx = np.argpartition(pos_sims, -k)[-k:]
        top_pos_idx = top_pos_idx[np.argsort(pos_sims[top_pos_idx])[::-1]]
        top_positive_ad_ids = [int(ad_ids[pos_indices[i]]) for i in top_pos_idx]
    else:
        top_positive_ad_ids = []

    # Engagement rate variance and top engaged cluster IDs
    cluster_rates = [c["engagement_rate"] for c in engagement_by_cluster]
    engagement_rate_variance = float(np.var(cluster_rates)) if len(cluster_rates) > 0 else 0.0
    sorted_clusters = sorted(engagement_by_cluster, key=lambda c: c["engagement_rate"], reverse=True)
    top_engaged_cluster_ids = [c["cluster_id"] for c in sorted_clusters[:3]]

    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_clusters": positive_clusters,
        "avg_positive_similarity": avg_pos_sim,
        "avg_negative_similarity": avg_neg_sim,
        "engagement_by_cluster": engagement_by_cluster,
        "similarity_gap": float(similarity_gap),
        "gap_ratio": float(gap_ratio),
        "positive_sim_std": positive_sim_std,
        "negative_sim_std": negative_sim_std,
        "overlap_fraction": overlap_fraction,
        "top_positive_ad_ids": top_positive_ad_ids,
        "engagement_rate_variance": engagement_rate_variance,
        "top_engaged_cluster_ids": top_engaged_cluster_ids,
    }
