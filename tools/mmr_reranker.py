import numpy as np
from typing import Dict, List


def mmr_reranker(
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    candidate_ad_ids: List[int],
    user_emb: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 100,
) -> Dict:
    """Re-rank a set of candidate ads using MMR for diversity.

    MMR_score = lambda * relevance(ad) - (1-lambda) * max_sim(ad, selected)

    Args:
        ad_embs: All ad embeddings [N, D].
        ad_ids: All ad ids [N].
        candidate_ad_ids: List of ad_ids to re-rank (from other tools).
        user_emb: User embedding for relevance.
        lambda_param: Trade-off between relevance and diversity (0.7 = 70% relevance).
        top_k: Number of ads to select.

    Returns:
        Dict with results, n_candidates, n_selected, and lambda.
    """
    # Build index from ad_id to position
    id_to_idx = {int(ad_ids[i]): i for i in range(len(ad_ids))}
    candidate_indices = [id_to_idx[int(aid)] for aid in candidate_ad_ids if int(aid) in id_to_idx]

    if not candidate_indices:
        return {"results": [], "n_candidates": 0}

    cand_embs = ad_embs[candidate_indices]
    cand_ids = [int(ad_ids[i]) for i in candidate_indices]

    # Normalize
    user_unit = user_emb / np.linalg.norm(user_emb).clip(1e-8)
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True).clip(1e-8)
    cand_units = cand_embs / cand_norms

    relevance = cand_units @ user_unit

    # Greedy MMR selection
    selected = []
    selected_embs = []
    remaining = list(range(len(candidate_indices)))

    for _ in range(min(top_k, len(remaining))):
        best_score = -float('inf')
        best_idx = -1

        for i in remaining:
            rel = float(relevance[i])
            if selected_embs:
                sel_arr = np.array(selected_embs)
                max_sim = float(np.max(sel_arr @ cand_units[i]))
            else:
                max_sim = 0.0
            mmr = lambda_param * rel - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i

        selected.append(best_idx)
        selected_embs.append(cand_units[best_idx])
        remaining.remove(best_idx)

    results = [
        {"ad_id": cand_ids[i], "mmr_score": float(relevance[i]), "rank": r + 1}
        for r, i in enumerate(selected)
    ]

    return {
        "results": results,
        "n_candidates": len(candidate_indices),
        "n_selected": len(selected),
        "lambda": lambda_param,
    }
