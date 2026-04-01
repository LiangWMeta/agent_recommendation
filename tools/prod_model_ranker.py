import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional


def prod_model_ranker(
    ad_ids: np.ndarray,
    top_k: int = 100,
    prod_data_dir: str = "data/local/model/enriched",
    request_id: Optional[int] = None,
) -> Dict:
    """Rank ads by production model prediction (calibrated CTR from SlimDSNN).

    This is the strongest per-ad signal — the actual production PM model's
    calibrated prediction of click-through rate. In production, this score
    determines which ads survive PreMatch truncation.

    Args:
        ad_ids: Ad ID array, shape [N].
        top_k: Number of top ads to return.
        prod_data_dir: Directory containing prod prediction sidecar files.
        request_id: Request ID to load prod predictions for.

    Returns:
        Dict with ranked results and diagnostics.
    """
    # Load prod predictions from sidecar JSON
    prod_path = Path(prod_data_dir) / f"{request_id}_prod.json"
    if not prod_path.exists():
        return {
            "results": [],
            "error": f"No prod prediction data at {prod_path}. Run extract_prod_predictions.py first.",
            "available": False,
        }

    with open(prod_path) as f:
        prod_data = json.load(f)

    # Build ad_id -> prod_prediction mapping
    pred_map = {int(entry["ad_id"]): float(entry["prod_prediction"])
                for entry in prod_data if entry.get("prod_prediction") is not None}

    # Score all ads
    scored = []
    missing = 0
    for aid in ad_ids:
        aid_int = int(aid)
        if aid_int in pred_map:
            scored.append((aid_int, pred_map[aid_int]))
        else:
            missing += 1

    # Sort by prod_prediction descending
    scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_k]

    results = [{"ad_id": aid, "prod_prediction": score} for aid, score in top_scored]

    scores_arr = np.array([s for _, s in scored]) if scored else np.array([0.0])

    return {
        "results": results,
        "available": True,
        "coverage": len(scored) / len(ad_ids) if len(ad_ids) > 0 else 0,
        "missing_ads": missing,
        "score_stats": {
            "mean": float(scores_arr.mean()),
            "std": float(scores_arr.std()),
            "min": float(scores_arr.min()),
            "max": float(scores_arr.max()),
        },
    }
