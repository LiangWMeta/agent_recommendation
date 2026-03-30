import numpy as np
from typing import List, Dict, Optional, Set


def similar_ads_lookup(
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    reference_ad_ids: List[int],
    top_k_per_ref: int = 10,
    exclude_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """Find similar ads for each reference ad by cosine similarity.

    Args:
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        reference_ad_ids: List of ad IDs to find similar ads for.
        top_k_per_ref: Number of similar ads per reference.
        exclude_ids: Ad IDs to exclude from results.

    Returns:
        List of dicts, one per reference ad, each with "reference_ad_id"
        and "similar_ads" list of {"ad_id", "score"}.
    """
    exclude: Set[int] = set(exclude_ids) if exclude_ids else set()

    # Build id-to-index mapping
    id_to_idx = {int(aid): i for i, aid in enumerate(ad_ids)}

    # Normalize all embeddings
    norms = np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10
    normed = ad_embs / norms

    results = []
    for ref_id in reference_ad_ids:
        if ref_id not in id_to_idx:
            results.append({"reference_ad_id": ref_id, "similar_ads": []})
            continue

        ref_idx = id_to_idx[ref_id]
        ref_vec = normed[ref_idx]

        # Cosine similarities to all ads
        sims = normed @ ref_vec  # shape [N]

        # Mask out the reference itself and excluded ids
        mask = np.ones(len(ad_ids), dtype=bool)
        mask[ref_idx] = False
        for eid in exclude:
            if eid in id_to_idx:
                mask[id_to_idx[eid]] = False

        valid_indices = np.where(mask)[0]
        valid_sims = sims[valid_indices]

        k = min(top_k_per_ref, len(valid_indices))
        if k == 0:
            results.append({"reference_ad_id": ref_id, "similar_ads": []})
            continue

        top_local = np.argpartition(valid_sims, -k)[-k:]
        top_local = top_local[np.argsort(valid_sims[top_local])[::-1]]

        similar = [
            {"ad_id": int(ad_ids[valid_indices[i]]), "score": float(valid_sims[i])}
            for i in top_local
        ]
        results.append({"reference_ad_id": ref_id, "similar_ads": similar})

    return results
