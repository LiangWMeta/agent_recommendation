import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from sklearn.cluster import KMeans


def prod_model_ranker(
    ad_ids: np.ndarray,
    top_k: int = 100,
    mode: str = "rank_all",
    prod_data_dir: str = "data/local/model/enriched",
    request_id: Optional[int] = None,
    scoring: str = "ecpm",
    ad_embs: Optional[np.ndarray] = None,
    user_emb: Optional[np.ndarray] = None,
    n_coarse: int = 10,
    expand_top_k_coarse: int = 3,
) -> Dict:
    """Rank ads by production model scoring.

    Two modes:
    - rank_all: Score every candidate by eCPM (pm_total_value) or pCTR.
    - with_hsnn: Use HSNN hierarchy to select candidates at sublinear cost,
      then score by eCPM. Reports compute savings.

    Two scoring options:
    - ecpm: Use pm_total_value (pCTR × pCVR × bid). Most realistic.
    - ctr: Use prod_prediction (pCTR only). Fallback when eCPM unavailable.

    Args:
        ad_ids: Ad ID array, shape [N].
        top_k: Number of top ads to return.
        mode: "rank_all" or "with_hsnn".
        prod_data_dir: Directory with prod prediction sidecar files.
        request_id: Request ID to load data for.
        scoring: "ecpm" (pm_total_value) or "ctr" (prod_prediction).
        ad_embs: Ad embeddings (required for with_hsnn mode).
        user_emb: User embedding (required for with_hsnn mode).
        n_coarse: HSNN coarse clusters (with_hsnn mode).
        expand_top_k_coarse: HSNN clusters to expand (with_hsnn mode).
    """
    prod_path = Path(prod_data_dir) / f"{request_id}_prod.json"
    if not prod_path.exists():
        return {
            "results": [],
            "error": f"No prod data at {prod_path}. Run extract_prod_predictions.py first.",
            "available": False,
        }

    with open(prod_path) as f:
        prod_data = json.load(f)

    # Build ad_id -> scores mapping
    # eCPM priority: median_pm_tv (RAA) > median_ecpm > pm_total_value > prod_prediction
    score_map = {}
    ecpm_available = False
    for entry in prod_data:
        aid = int(entry.get("ad_id", 0))
        if scoring == "ecpm":
            if entry.get("median_pm_tv") is not None:
                score_map[aid] = float(entry["median_pm_tv"])
                ecpm_available = True
            elif entry.get("median_ecpm") is not None:
                score_map[aid] = float(entry["median_ecpm"])
                ecpm_available = True
            elif entry.get("pm_total_value") is not None:
                score_map[aid] = float(entry["pm_total_value"])
                ecpm_available = True
            elif entry.get("prod_prediction") is not None:
                score_map[aid] = float(entry["prod_prediction"])
        elif entry.get("prod_prediction") is not None:
            score_map[aid] = float(entry["prod_prediction"])

    # Determine actual scoring method used
    actual_scoring = "ecpm" if ecpm_available and scoring == "ecpm" else "ctr"

    # Determine which ads to score
    if mode == "with_hsnn" and ad_embs is not None and user_emb is not None:
        # HSNN: cluster, expand top-K coarse, only score those
        n_ads = len(ad_ids)
        k = min(n_coarse, n_ads)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(ad_embs)

        # Score coarse centroids by cosine to user
        user_unit = user_emb / (np.linalg.norm(user_emb) + 1e-10)
        centroid_scores = km.cluster_centers_ @ user_unit / (
            np.linalg.norm(km.cluster_centers_, axis=1) + 1e-10
        )
        top_clusters = set(np.argsort(centroid_scores)[-expand_top_k_coarse:])

        # Only score ads in expanded clusters
        candidate_mask = np.array([cluster_ids[i] in top_clusters for i in range(n_ads)])
        candidate_ids = ad_ids[candidate_mask]
        n_scored = int(candidate_mask.sum())
        n_pruned = n_ads - n_scored
        compute_savings = n_pruned / n_ads if n_ads > 0 else 0.0
    else:
        # rank_all: score every ad
        candidate_ids = ad_ids
        n_scored = len(ad_ids)
        n_pruned = 0
        compute_savings = 0.0

    # Score candidates
    scored = []
    missing = 0
    for aid in candidate_ids:
        aid_int = int(aid)
        if aid_int in score_map:
            scored.append((aid_int, score_map[aid_int]))
        else:
            missing += 1

    scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_k]

    results = [{"ad_id": aid, "score": score} for aid, score in top_scored]

    scores_arr = np.array([s for _, s in scored]) if scored else np.array([0.0])

    return {
        "results": results,
        "available": True,
        "mode": mode,
        "scoring": actual_scoring,
        "coverage": len(scored) / len(candidate_ids) if len(candidate_ids) > 0 else 0,
        "missing_ads": missing,
        "n_scored": n_scored,
        "n_pruned": n_pruned,
        "compute_savings": compute_savings,
        "score_stats": {
            "mean": float(scores_arr.mean()),
            "std": float(scores_arr.std()),
            "min": float(scores_arr.min()),
            "max": float(scores_arr.max()),
        },
    }
