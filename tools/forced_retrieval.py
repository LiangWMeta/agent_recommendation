import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional


def forced_retrieval(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    top_k: int = 100,
    prod_data_dir: str = "data/local/model/enriched",
    request_id: Optional[int] = None,
) -> Dict:
    """Forced Retrieval route using production flags.

    In production, FR ads are flagged with is_forced_retrieval=true and
    ranked by eCPM (pm_total_value). This tool uses those prod flags when
    available, falling back to centroid-based approximation otherwise.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N].
        top_k: Number of top results to return.
        prod_data_dir: Directory with prod prediction sidecar files.
        request_id: Request ID to load FR flags for.

    Returns:
        Dict with results, source (prod_flags or centroid_fallback), diagnostics.
    """
    # Try to load prod flags
    fr_ads = None
    if request_id is not None:
        prod_path = Path(prod_data_dir) / f"{request_id}_prod.json"
        if prod_path.exists():
            with open(prod_path) as f:
                prod_data = json.load(f)

            # Find FR-flagged ads and their eCPM scores
            fr_entries = []
            for entry in prod_data:
                if entry.get("is_forced_retrieval", False):
                    aid = int(entry["ad_id"])
                    score = entry.get("pm_total_value") or entry.get("prod_prediction") or 0.0
                    fr_entries.append((aid, float(score)))

            if fr_entries:
                # Sort by eCPM descending, take top-K
                fr_entries.sort(key=lambda x: x[1], reverse=True)
                fr_ads = fr_entries[:top_k]

    if fr_ads is not None:
        # Prod-flagged FR
        results = [{"ad_id": aid, "score": score} for aid, score in fr_ads]

        # Compute diagnostics: how do FR ads compare to non-FR
        fr_id_set = {aid for aid, _ in fr_ads}
        ad_id_list = ad_ids.tolist()
        user_unit = user_emb / (np.linalg.norm(user_emb) + 1e-10)
        ad_units = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
        cosines = ad_units @ user_unit

        fr_mask = np.array([int(aid) in fr_id_set for aid in ad_ids])
        fr_cosine_mean = float(cosines[fr_mask].mean()) if fr_mask.any() else 0.0
        non_fr_cosine_mean = float(cosines[~fr_mask].mean()) if (~fr_mask).any() else 0.0

        # Count FR positives
        n_fr_positives = int((labels[fr_mask] == 1).sum()) if fr_mask.any() else 0

        return {
            "results": results,
            "source": "prod_flags",
            "n_fr_ads": len(fr_ads),
            "n_fr_positives": n_fr_positives,
            "fr_cosine_mean": fr_cosine_mean,
            "non_fr_cosine_mean": non_fr_cosine_mean,
            "cosine_gap": fr_cosine_mean - non_fr_cosine_mean,
        }

    # Fallback: centroid-based approximation
    pos_mask = labels == 1
    if pos_mask.sum() == 0:
        return {"results": [], "source": "centroid_fallback", "error": "No positive ads"}

    pos_embs = ad_embs[pos_mask]
    centroid = pos_embs.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
    user_unit = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_units = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)

    scores = ad_units @ centroid_norm
    top_idx = np.argsort(-scores)[:top_k]

    user_scores = ad_units @ user_unit
    centroid_gap = float(scores[pos_mask].mean() - scores[~pos_mask].mean())
    user_gap = float(user_scores[pos_mask].mean() - user_scores[~pos_mask].mean())
    correlation = float(np.corrcoef(scores, user_scores)[0, 1])

    results = [{"ad_id": int(ad_ids[i]), "score": float(scores[i])} for i in top_idx]

    return {
        "results": results,
        "source": "centroid_fallback",
        "centroid_gap": centroid_gap,
        "user_emb_gap": user_gap,
        "centroid_vs_user_correlation": correlation,
        "n_positives_used": int(pos_mask.sum()),
    }
