import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional


def cluster_explorer(
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    n_clusters: int = 5,
    target_cluster_ids: Optional[List[int]] = None,
    top_k_per_cluster: int = 20,
    labels: Optional[np.ndarray] = None,
) -> Dict:
    """Explore ad embeddings via k-means clustering.

    Args:
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        n_clusters: Number of clusters for k-means.
        target_cluster_ids: If provided, only return ads from these clusters.
        top_k_per_cluster: Max ads to return per cluster (closest to centroid).
        labels: Optional binary engagement labels, shape [N]. 1=positive, 0=negative.

    Returns:
        Dict with "clusters" (metadata per cluster) and "ads" (ad assignments).
    """
    n_clusters = min(n_clusters, len(ad_embs))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ad_embs)
    centroids = kmeans.cluster_centers_

    # Determine which clusters to include
    if target_cluster_ids is not None:
        selected = set(target_cluster_ids)
    else:
        selected = set(range(n_clusters))

    # Build cluster metadata
    clusters_info = []
    for cid in range(n_clusters):
        if cid not in selected:
            continue
        mask = cluster_labels == cid
        cluster_dict = {
            "cluster_id": cid,
            "size": int(mask.sum()),
            "centroid_summary": centroids[cid].tolist(),
        }
        if labels is not None:
            c_total = int(mask.sum())
            c_pos = int((mask & (labels == 1)).sum())
            cluster_dict["positive_count"] = c_pos
            cluster_dict["engagement_rate"] = float(c_pos / c_total) if c_total > 0 else 0.0
        clusters_info.append(cluster_dict)

    # Build ads list: for each selected cluster, return top_k closest to centroid
    ads_list = []
    for cid in sorted(selected):
        mask = cluster_labels == cid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Distances to centroid
        dists = np.linalg.norm(ad_embs[indices] - centroids[cid], axis=1)
        k = min(top_k_per_cluster, len(indices))
        if k >= len(indices):
            top_idx = np.arange(len(indices))
        else:
            top_idx = np.argpartition(dists, k)[:k]
        top_idx = top_idx[np.argsort(dists[top_idx])]

        for i in top_idx:
            ads_list.append({
                "ad_id": int(ad_ids[indices[i]]),
                "cluster_id": cid,
                "distance": float(dists[i]),
            })

    return {"clusters": clusters_info, "ads": ads_list}
