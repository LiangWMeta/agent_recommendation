import numpy as np
from typing import List, Dict, Union, Tuple


def feature_filter(
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    user_emb: np.ndarray,
    feature_name: str,
    operator: str,
    value: Union[float, Tuple[float, float]],
    top_k: int = 50,
) -> List[Dict]:
    """Filter ads by embedding-derived features.

    Args:
        ad_embs: Ad embedding matrix, shape [N, D].
        ad_ids: Ad ID array, shape [N].
        user_emb: User embedding vector, shape [D].
        feature_name: One of "cosine_score", "embedding_norm", "embedding_mean".
        operator: One of "gt", "lt", "between".
        value: Scalar for gt/lt, tuple (low, high) for between.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with "ad_id" and the computed feature value.
    """
    # Compute the requested feature
    if feature_name == "cosine_score":
        user_norm = user_emb / (np.linalg.norm(user_emb) + 1e-10)
        ad_norms = ad_embs / (np.linalg.norm(ad_embs, axis=1, keepdims=True) + 1e-10)
        feature_vals = ad_norms @ user_norm
    elif feature_name == "embedding_norm":
        feature_vals = np.linalg.norm(ad_embs, axis=1)
    elif feature_name == "embedding_mean":
        feature_vals = np.mean(ad_embs, axis=1)
    else:
        raise ValueError(
            f"Unknown feature_name '{feature_name}'. "
            "Supported: cosine_score, embedding_norm, embedding_mean"
        )

    # Apply operator
    if operator == "gt":
        mask = feature_vals > value
    elif operator == "lt":
        mask = feature_vals < value
    elif operator == "between":
        low, high = value
        mask = (feature_vals >= low) & (feature_vals <= high)
    else:
        raise ValueError(
            f"Unknown operator '{operator}'. Supported: gt, lt, between"
        )

    filtered_ids = ad_ids[mask]
    filtered_vals = feature_vals[mask]

    # Sort by feature value descending, take top_k
    order = np.argsort(filtered_vals)[::-1][:top_k]

    return [
        {"ad_id": int(filtered_ids[i]), feature_name: float(filtered_vals[i])}
        for i in order
    ]
