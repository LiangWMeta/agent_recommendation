import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from scipy.stats import spearmanr


def pipeline_simulator(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    stage: str = "all",
    pm_budget: int = 500,
    ai_budget: int = 100,
    af_budget: int = 20,
    prod_data_dir: str = "data_enriched",
    request_id: int = None,
) -> Dict:
    """Simulate the production ads retrieval cascaded pipeline: AP -> PM -> AI -> AF.

    Each stage progressively filters candidates using increasingly sophisticated
    scoring. Tracks positive (engaged) ad survival across stages to identify
    where the funnel loses valuable candidates.

    Args:
        user_emb: User embedding vector, shape [D].
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        labels: Binary engagement labels, shape [N]. 1=positive, 0=negative.
        stage: Which stage to run up to: "all", "AP", "PM", "AI", "AF".
        pm_budget: How many ads survive PreMatch truncation.
        ai_budget: How many ads survive AdInference.
        af_budget: How many ads survive AdFilter (final).
        prod_data_dir: Directory containing prod prediction sidecar files.
        request_id: Request ID to load prod predictions for.

    Returns:
        Dict with per-stage stats, cross-stage rank correlations, and drop-off summary.
    """
    stage_order = ["AP", "PM", "AI", "AF"]
    if stage == "all":
        target_idx = len(stage_order) - 1
    else:
        target_idx = stage_order.index(stage)

    n_total = len(ad_ids)
    total_positives = int(labels.sum())

    # Normalize embeddings for cosine similarity
    user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
    ad_norms_matrix = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
    cosine_scores = ad_norms_matrix @ user_norm  # shape [N]

    # --- Load prod predictions if available ---
    scoring_method = "cosine_fallback"
    pred_map = {}
    if request_id is not None:
        prod_path = Path(prod_data_dir) / f"{request_id}_prod.json"
        if prod_path.exists():
            with open(prod_path) as f:
                prod_data = json.load(f)
            pred_map = {
                int(entry["ad_id"]): float(entry["prod_prediction"])
                for entry in prod_data
                if entry.get("prod_prediction") is not None
            }
            if pred_map:
                scoring_method = "prod_prediction"

    stages_output = {}

    # Track current surviving indices (into original arrays)
    surviving_idx = np.arange(n_total)

    # Score arrays that persist across stages for correlation computation
    pm_scores_for_survivors = None
    ai_scores_for_survivors = None
    af_scores_for_survivors = None

    # ---- AP stage ----
    ap_order = np.argsort(cosine_scores)[::-1]
    ap_ad_ids = ad_ids[ap_order]
    ap_positives = int(labels.sum())

    stages_output["AP"] = {
        "n_candidates": n_total,
        "n_positives": ap_positives,
        "top_ad_ids": [int(x) for x in ap_ad_ids[:20]],
    }

    surviving_idx = ap_order  # all candidates, sorted by cosine

    if target_idx < 1:
        return _build_result(stages_output, stage_order[:target_idx + 1],
                             total_positives, scoring_method)

    # ---- PM stage ----
    if scoring_method == "prod_prediction":
        # Score by prod_prediction
        pm_scores = np.array([
            pred_map.get(int(ad_ids[i]), 0.0) for i in surviving_idx
        ])
    else:
        # Fallback: cosine similarity + small cluster engagement bonus
        from sklearn.cluster import KMeans

        n_clusters = min(10, n_total)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ad_embs)

        # Compute per-cluster engagement rate as bonus
        cluster_eng_rate = np.zeros(n_clusters)
        for cid in range(n_clusters):
            mask = cluster_labels == cid
            c_total = mask.sum()
            if c_total > 0:
                cluster_eng_rate[cid] = labels[mask].sum() / c_total

        bonus = cluster_eng_rate[cluster_labels[surviving_idx]]
        pm_scores = cosine_scores[surviving_idx] + 0.1 * bonus

    pm_order = np.argsort(pm_scores)[::-1]
    pm_keep = min(pm_budget, len(pm_order))
    pm_selected = pm_order[:pm_keep]

    pm_surviving_idx = surviving_idx[pm_selected]
    pm_ad_ids_sorted = ad_ids[pm_surviving_idx]
    pm_positives = int(labels[pm_surviving_idx].sum())
    pm_scores_for_survivors = pm_scores[pm_selected]

    stages_output["PM"] = {
        "n_candidates": len(surviving_idx),
        "n_survived": pm_keep,
        "n_positives_survived": pm_positives,
        "positive_survival_rate": float(pm_positives / total_positives) if total_positives > 0 else 0.0,
        "top_ad_ids": [int(x) for x in pm_ad_ids_sorted[:20]],
    }

    surviving_idx = pm_surviving_idx

    if target_idx < 2:
        return _build_result(stages_output, stage_order[:target_idx + 1],
                             total_positives, scoring_method)

    # ---- AI stage ----
    # Normalize PM scores to [0, 1] for combination
    pm_scores_current = pm_scores_for_survivors.copy()
    pm_min, pm_max = pm_scores_current.min(), pm_scores_current.max()
    if pm_max - pm_min > 1e-10:
        norm_pm = (pm_scores_current - pm_min) / (pm_max - pm_min)
    else:
        norm_pm = np.ones_like(pm_scores_current)

    # Greedy diversity-aware selection
    n_ai = min(ai_budget, len(surviving_idx))
    ai_embs = ad_norms_matrix[surviving_idx]

    selected_mask = np.zeros(len(surviving_idx), dtype=bool)
    ai_combined_scores = np.zeros(len(surviving_idx))
    selection_order = []

    for step in range(n_ai):
        if step == 0:
            # No diversity penalty for first pick — use PM score only
            diversity_bonus = np.zeros(len(surviving_idx))
        else:
            # Diversity bonus = negative of max similarity to already-selected
            selected_embs = ai_embs[selected_mask]
            sim_to_selected = ai_embs @ selected_embs.T  # [remaining, selected]
            max_sim = sim_to_selected.max(axis=1)
            diversity_bonus = -max_sim

        combined = 0.7 * norm_pm + 0.3 * diversity_bonus
        combined[selected_mask] = -np.inf  # exclude already-selected

        best = np.argmax(combined)
        selected_mask[best] = True
        ai_combined_scores[best] = combined[best]
        selection_order.append(best)

    ai_selected = np.array(selection_order)
    ai_surviving_idx = surviving_idx[ai_selected]
    ai_ad_ids_sorted = ad_ids[ai_surviving_idx]
    ai_positives = int(labels[ai_surviving_idx].sum())
    ai_scores_for_survivors = ai_combined_scores[ai_selected]

    stages_output["AI"] = {
        "n_candidates": len(surviving_idx),
        "n_survived": n_ai,
        "n_positives_survived": ai_positives,
        "positive_survival_rate": float(ai_positives / total_positives) if total_positives > 0 else 0.0,
        "top_ad_ids": [int(x) for x in ai_ad_ids_sorted[:20]],
    }

    # PM scores for AI survivors (for correlation)
    pm_scores_ai_survivors = pm_scores_for_survivors[ai_selected]

    surviving_idx = ai_surviving_idx

    if target_idx < 3:
        return _build_result(stages_output, stage_order[:target_idx + 1],
                             total_positives, scoring_method,
                             pm_scores_ai_survivors=pm_scores_ai_survivors,
                             ai_scores_for_survivors=ai_scores_for_survivors)

    # ---- AF stage ----
    # Final ranking by PM score (production full model)
    if scoring_method == "prod_prediction":
        af_scores = np.array([
            pred_map.get(int(ad_ids[i]), 0.0) for i in surviving_idx
        ])
    else:
        af_scores = cosine_scores[surviving_idx]

    af_order = np.argsort(af_scores)[::-1]
    af_keep = min(af_budget, len(af_order))
    af_selected = af_order[:af_keep]

    af_surviving_idx = surviving_idx[af_selected]
    af_ad_ids_sorted = ad_ids[af_surviving_idx]
    af_positives = int(labels[af_surviving_idx].sum())
    af_scores_for_survivors = af_scores[af_selected]

    # AI scores for AF survivors (for correlation)
    ai_scores_af_survivors = ai_scores_for_survivors[af_selected]

    stages_output["AF"] = {
        "n_candidates": len(surviving_idx),
        "n_survived": af_keep,
        "n_positives_survived": af_positives,
        "positive_survival_rate": float(af_positives / total_positives) if total_positives > 0 else 0.0,
        "top_ad_ids": [int(x) for x in af_ad_ids_sorted[:20]],
    }

    return _build_result(
        stages_output, stage_order[:target_idx + 1],
        total_positives, scoring_method,
        pm_scores_ai_survivors=pm_scores_ai_survivors,
        ai_scores_for_survivors=ai_scores_for_survivors,
        ai_scores_af_survivors=ai_scores_af_survivors,
        af_scores_for_survivors=af_scores_for_survivors,
    )


def _build_result(
    stages: Dict,
    stage_list: list,
    total_positives: int,
    scoring_method: str,
    pm_scores_ai_survivors: np.ndarray = None,
    ai_scores_for_survivors: np.ndarray = None,
    ai_scores_af_survivors: np.ndarray = None,
    af_scores_for_survivors: np.ndarray = None,
) -> Dict:
    result = {
        "stages": stages,
        "cross_stage_consistency": {},
        "drop_off_summary": {},
        "scoring_method": scoring_method,
    }

    # Cross-stage rank correlations
    pm_ai_corr = _safe_spearman(pm_scores_ai_survivors, ai_scores_for_survivors)
    ai_af_corr = _safe_spearman(ai_scores_af_survivors, af_scores_for_survivors)
    result["cross_stage_consistency"] = {
        "pm_ai_rank_correlation": pm_ai_corr,
        "ai_af_rank_correlation": ai_af_corr,
    }

    # Drop-off summary
    drop_off = {}
    if total_positives > 0:
        ap_pos = stages.get("AP", {}).get("n_positives", total_positives)
        pm_pos = stages.get("PM", {}).get("n_positives_survived", ap_pos)
        ai_pos = stages.get("AI", {}).get("n_positives_survived", pm_pos)
        af_pos = stages.get("AF", {}).get("n_positives_survived", ai_pos)

        drop_off["ap_to_pm_positive_loss"] = float(1.0 - pm_pos / ap_pos) if "PM" in stages and ap_pos > 0 else 0.0
        drop_off["pm_to_ai_positive_loss"] = float(1.0 - ai_pos / pm_pos) if "AI" in stages and pm_pos > 0 else 0.0
        drop_off["ai_to_af_positive_loss"] = float(1.0 - af_pos / ai_pos) if "AF" in stages and ai_pos > 0 else 0.0

        last_stage = stage_list[-1]
        if last_stage == "AP":
            final_pos = ap_pos
        elif last_stage == "PM":
            final_pos = pm_pos
        elif last_stage == "AI":
            final_pos = ai_pos
        else:
            final_pos = af_pos
        drop_off["total_positive_survival"] = float(final_pos / total_positives)
    else:
        drop_off["ap_to_pm_positive_loss"] = 0.0
        drop_off["pm_to_ai_positive_loss"] = 0.0
        drop_off["ai_to_af_positive_loss"] = 0.0
        drop_off["total_positive_survival"] = 0.0

    result["drop_off_summary"] = drop_off

    return result


def _safe_spearman(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    if len(a) != len(b):
        return 0.0
    corr, _ = spearmanr(a, b)
    if np.isnan(corr):
        return 0.0
    return float(corr)
