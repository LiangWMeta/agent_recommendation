import numpy as np
from sklearn.cluster import KMeans
from typing import Dict


def hsnn_cluster_scorer(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    n_coarse: int = 10,
    n_fine_per_coarse: int = 5,
    expand_top_k_coarse: int = 3,
    top_k: int = 100,
) -> Dict:
    """Simulate HSNN 2-level hierarchical cluster scoring for sublinear retrieval.

    Level 1 (coarse) clusters the full ad pool, scores centroids against the user
    embedding, and expands only the top-K coarse clusters.  Level 2 (fine)
    sub-clusters each expanded coarse cluster and scores individual ads.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N]. 1=positive, 0=negative.
        n_coarse: Number of coarse-level clusters.
        n_fine_per_coarse: Number of fine sub-clusters within each expanded coarse cluster.
        expand_top_k_coarse: Number of top coarse clusters to expand (rest are pruned).
        top_k: Number of top ads to return.

    Returns:
        Dict with coarse_clusters, fine_clusters, results, computational_savings,
        n_ads_scored, n_ads_pruned, and stage.
    """
    n_ads = len(ad_embs)
    n_coarse = min(n_coarse, n_ads)
    expand_top_k_coarse = min(expand_top_k_coarse, n_coarse)

    # Normalize user embedding
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)

    # --- Level 1: Coarse clustering ---
    coarse_km = KMeans(n_clusters=n_coarse, random_state=42, n_init=10)
    coarse_labels = coarse_km.fit_predict(ad_embs)
    coarse_centroids = coarse_km.cluster_centers_

    # Score coarse centroids by cosine similarity to user embedding
    centroid_norms = coarse_centroids / (
        np.linalg.norm(coarse_centroids, axis=1, keepdims=True) + 1e-10
    )
    coarse_scores = centroid_norms @ user_norm  # shape [n_coarse]

    # Build coarse cluster info
    coarse_order = np.argsort(coarse_scores)[::-1]
    expanded_set = set(coarse_order[:expand_top_k_coarse].tolist())

    coarse_clusters = []
    for cid in coarse_order:
        cid = int(cid)
        mask = coarse_labels == cid
        size = int(mask.sum())
        n_pos = int((mask & (labels == 1)).sum())
        eng_rate = float(n_pos / size) if size > 0 else 0.0
        coarse_clusters.append({
            "cluster_id": cid,
            "centroid_score": float(coarse_scores[cid]),
            "size": size,
            "n_positives": n_pos,
            "engagement_rate": eng_rate,
            "expanded": cid in expanded_set,
        })

    # --- Expansion: collect ads in expanded coarse clusters ---
    expanded_indices = []
    expanded_coarse_ids = []
    for cid in sorted(expanded_set):
        mask = coarse_labels == cid
        indices = np.where(mask)[0]
        expanded_indices.append(indices)
        expanded_coarse_ids.append(cid)

    n_ads_scored = sum(len(idx) for idx in expanded_indices)
    n_ads_pruned = n_ads - n_ads_scored
    computational_savings = float(n_ads_pruned / n_ads) if n_ads > 0 else 0.0

    # --- Level 2: Fine clustering within each expanded coarse cluster ---
    fine_clusters = []
    # Map each ad index to its fine cluster id within its parent coarse cluster
    ad_fine_map = {}  # global_index -> (coarse_id, fine_cluster_id)

    for coarse_id, indices in zip(expanded_coarse_ids, expanded_indices):
        n_in_cluster = len(indices)
        n_fine = min(n_fine_per_coarse, n_in_cluster)

        if n_fine <= 1:
            # All ads in a single fine cluster
            fine_centroid = ad_embs[indices].mean(axis=0)
            fine_centroid_norm = fine_centroid / (np.linalg.norm(fine_centroid) + 1e-10)
            fine_score = float(fine_centroid_norm @ user_norm)
            fine_clusters.append({
                "cluster_id": 0,
                "parent_coarse_id": int(coarse_id),
                "centroid_score": fine_score,
                "size": n_in_cluster,
            })
            for idx in indices:
                ad_fine_map[idx] = (int(coarse_id), 0)
        else:
            fine_km = KMeans(n_clusters=n_fine, random_state=42, n_init=10)
            fine_labels = fine_km.fit_predict(ad_embs[indices])
            fine_centroids = fine_km.cluster_centers_

            fine_centroid_norms = fine_centroids / (
                np.linalg.norm(fine_centroids, axis=1, keepdims=True) + 1e-10
            )
            fine_scores = fine_centroid_norms @ user_norm

            for fid in range(n_fine):
                f_mask = fine_labels == fid
                fine_clusters.append({
                    "cluster_id": int(fid),
                    "parent_coarse_id": int(coarse_id),
                    "centroid_score": float(fine_scores[fid]),
                    "size": int(f_mask.sum()),
                })

            for local_i, global_i in enumerate(indices):
                ad_fine_map[global_i] = (int(coarse_id), int(fine_labels[local_i]))

    # --- Ad scoring: cosine(ad_emb, user_emb) for all ads in expanded clusters ---
    all_expanded = np.concatenate(expanded_indices) if expanded_indices else np.array([], dtype=int)

    if len(all_expanded) == 0:
        return {
            "coarse_clusters": coarse_clusters,
            "fine_clusters": fine_clusters,
            "results": [],
            "computational_savings": computational_savings,
            "n_ads_scored": 0,
            "n_ads_pruned": n_ads_pruned,
            "stage": "AP",
        }

    expanded_embs = ad_embs[all_expanded]
    expanded_norms = expanded_embs / (
        np.linalg.norm(expanded_embs, axis=1, keepdims=True) + 1e-10
    )
    ad_scores = expanded_norms @ user_norm

    # Top-k
    k = min(top_k, len(ad_scores))
    if k >= len(ad_scores):
        top_idx = np.arange(len(ad_scores))
    else:
        top_idx = np.argpartition(ad_scores, -k)[-k:]
    top_idx = top_idx[np.argsort(ad_scores[top_idx])[::-1]]

    results = []
    for i in top_idx:
        global_i = all_expanded[i]
        coarse_id, fine_id = ad_fine_map[global_i]
        results.append({
            "ad_id": int(ad_ids[global_i]),
            "score": float(ad_scores[i]),
            "coarse_cluster": coarse_id,
            "fine_cluster": fine_id,
        })

    return {
        "coarse_clusters": coarse_clusters,
        "fine_clusters": fine_clusters,
        "results": results,
        "computational_savings": computational_savings,
        "n_ads_scored": n_ads_scored,
        "n_ads_pruned": n_ads_pruned,
        "stage": "AP",
    }
