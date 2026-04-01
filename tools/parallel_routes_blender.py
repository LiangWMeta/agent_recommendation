import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression


def parallel_routes_blender(
    user_emb: np.ndarray,
    ad_embs: np.ndarray,
    ad_ids: np.ndarray,
    labels: np.ndarray,
    route_results: Dict[str, List[int]],  # {route_name: [ad_id, ad_id, ...]} ordered by route's score
    blending_strategy: str = "rrf",       # "rrf", "ml_blender", "priority"
    target_pool_size: int = 200,
    main_route_weight: float = 0.6,       # for priority mode
) -> Dict:
    """Simulate PRM (Parallel Retrieval Models) + ML Blender.

    Blends candidates from multiple retrieval routes using one of three
    strategies: reciprocal rank fusion, learned ML blending, or priority-based
    allocation.

    Args:
        user_emb: User embedding [D].
        ad_embs: All ad embeddings [N, D].
        ad_ids: All ad ids [N].
        labels: Binary engagement labels [N].
        route_results: Dict mapping route names to ordered lists of ad_ids
            (best first) from each route.
        blending_strategy: One of "rrf", "ml_blender", "priority".
        target_pool_size: Number of ads to keep after blending.
        main_route_weight: Weight for the main route in priority mode.

    Returns:
        Dict with blended_results, route_statistics, and blend metadata.
    """
    # Build lookup from ad_id to index in the global arrays
    id_to_idx = {int(ad_ids[i]): i for i in range(len(ad_ids))}
    label_map = {int(ad_ids[i]): int(labels[i]) for i in range(len(ad_ids))}

    # Filter out empty routes
    route_results = {k: v for k, v in route_results.items() if len(v) > 0}

    if not route_results:
        return {
            "blended_results": [],
            "route_statistics": {},
            "blending_strategy": blending_strategy,
            "n_total_unique": 0,
            "n_in_blended": 0,
            "blend_weights": {},
        }

    route_names = list(route_results.keys())

    # Collect all unique ad_ids and build per-route rank maps
    # rank_maps[route_name][ad_id] = 1-based rank in that route
    rank_maps: Dict[str, Dict[int, int]] = {}
    all_ad_ids_set = set()
    for rname, rlist in route_results.items():
        rank_maps[rname] = {}
        for rank_idx, aid in enumerate(rlist):
            aid = int(aid)
            rank_maps[rname][aid] = rank_idx + 1  # 1-based
            all_ad_ids_set.add(aid)

    all_unique_ads = list(all_ad_ids_set)

    # Determine which routes each ad belongs to
    ad_routes: Dict[int, List[str]] = {}
    for aid in all_unique_ads:
        ad_routes[aid] = [rname for rname in route_names if aid in rank_maps[rname]]

    # ---- Blending ----
    blended_scores: Dict[int, float] = {}
    blend_weights: Dict[str, float] = {}
    actual_strategy = blending_strategy

    if blending_strategy == "rrf":
        blend_weights = {rname: 1.0 for rname in route_names}
        blended_scores = _rrf_scores(all_unique_ads, rank_maps, route_names)

    elif blending_strategy == "ml_blender":
        # Check if we have enough positives to train
        n_positives = sum(1 for aid in all_unique_ads if label_map.get(aid, 0) == 1)
        if n_positives < 10:
            # Fallback to RRF
            actual_strategy = "rrf"
            blend_weights = {rname: 1.0 for rname in route_names}
            blended_scores = _rrf_scores(all_unique_ads, rank_maps, route_names)
        else:
            blended_scores, blend_weights = _ml_blender_scores(
                all_unique_ads, rank_maps, route_names, label_map
            )

    elif blending_strategy == "priority":
        blended_scores, blend_weights = _priority_scores(
            route_results, route_names, target_pool_size, main_route_weight
        )

    else:
        # Unknown strategy, default to RRF
        actual_strategy = "rrf"
        blend_weights = {rname: 1.0 for rname in route_names}
        blended_scores = _rrf_scores(all_unique_ads, rank_maps, route_names)

    # Sort by blended_score descending, take top target_pool_size
    sorted_ads = sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_ads = sorted_ads[:target_pool_size]

    # Build blended_results (cap output at 100 entries)
    blended_results = []
    for rank, (aid, score) in enumerate(sorted_ads[:100], start=1):
        blended_results.append({
            "ad_id": int(aid),
            "blended_score": round(float(score), 6),
            "contributing_routes": ad_routes.get(aid, []),
            "rank": rank,
        })

    # ---- Route statistics ----
    route_statistics = {}
    for rname in route_names:
        route_ads = set(int(a) for a in route_results[rname])
        other_ads = set()
        for other_name in route_names:
            if other_name != rname:
                other_ads.update(int(a) for a in route_results[other_name])

        unique_ads = route_ads - other_ads
        n_positives = sum(1 for aid in route_ads if label_map.get(aid, 0) == 1)
        n_unique_positives = sum(1 for aid in unique_ads if label_map.get(aid, 0) == 1)
        overlap = len(route_ads & other_ads) / max(len(route_ads), 1)

        route_statistics[rname] = {
            "n_candidates": len(route_ads),
            "n_unique": len(unique_ads),
            "n_positives": n_positives,
            "n_unique_positives": n_unique_positives,
            "overlap_with_other_routes": round(float(overlap), 4),
        }

    return {
        "blended_results": blended_results,
        "route_statistics": route_statistics,
        "blending_strategy": actual_strategy,
        "n_total_unique": len(all_unique_ads),
        "n_in_blended": len(sorted_ads),
        "blend_weights": {k: round(float(v), 6) for k, v in blend_weights.items()},
    }


def _rrf_scores(
    all_ads: List[int],
    rank_maps: Dict[str, Dict[int, int]],
    route_names: List[str],
    k: int = 60,
) -> Dict[int, float]:
    """Reciprocal Rank Fusion: score = sum_routes 1/(k + rank)."""
    scores = {}
    for aid in all_ads:
        s = 0.0
        for rname in route_names:
            if aid in rank_maps[rname]:
                s += 1.0 / (k + rank_maps[rname][aid])
        scores[aid] = s
    return scores


def _ml_blender_scores(
    all_ads: List[int],
    rank_maps: Dict[str, Dict[int, int]],
    route_names: List[str],
    label_map: Dict[int, int],
) -> tuple:
    """Learn route weights via LogisticRegression on reciprocal-rank features."""
    n_routes = len(route_names)
    n_ads = len(all_ads)

    # Build feature matrix: each row = [1/rank_in_route1, 1/rank_in_route2, ...]
    X = np.zeros((n_ads, n_routes), dtype=np.float64)
    y = np.zeros(n_ads, dtype=np.int32)

    for i, aid in enumerate(all_ads):
        for j, rname in enumerate(route_names):
            if aid in rank_maps[rname]:
                X[i, j] = 1.0 / rank_maps[rname][aid]
        y[i] = label_map.get(aid, 0)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)

    # Predicted probability as blended score
    probs = model.predict_proba(X)
    # Index of the positive class
    pos_idx = list(model.classes_).index(1)

    scores = {}
    for i, aid in enumerate(all_ads):
        scores[aid] = float(probs[i, pos_idx])

    # Extract learned weights (coefficients)
    coefs = model.coef_[0]
    blend_weights = {rname: float(coefs[j]) for j, rname in enumerate(route_names)}

    return scores, blend_weights


def _priority_scores(
    route_results: Dict[str, List[int]],
    route_names: List[str],
    target_pool_size: int,
    main_route_weight: float,
) -> tuple:
    """Priority-based allocation: main route gets a fixed share, rest split equally."""
    # Determine main route: first key, or the one with most candidates
    main_route = max(route_names, key=lambda r: len(route_results[r]))
    secondary_routes = [r for r in route_names if r != main_route]

    n_secondary = len(secondary_routes)
    if n_secondary > 0:
        secondary_weight = (1.0 - main_route_weight) / n_secondary
    else:
        main_route_weight = 1.0
        secondary_weight = 0.0

    blend_weights = {main_route: main_route_weight}
    for sr in secondary_routes:
        blend_weights[sr] = secondary_weight

    # Allocate slots
    main_slots = int(target_pool_size * main_route_weight)
    secondary_slots_each = (
        int(target_pool_size * secondary_weight) if n_secondary > 0 else 0
    )

    # Fill: main route first, then secondaries interleaved
    selected = []
    seen = set()

    # Main route
    for aid in route_results[main_route]:
        aid = int(aid)
        if aid not in seen and len(selected) < main_slots:
            selected.append(aid)
            seen.add(aid)

    # Secondary routes interleaved
    if secondary_routes:
        max_len = max(len(route_results[sr]) for sr in secondary_routes)
        remaining_slots = target_pool_size - len(selected)
        for idx in range(max_len):
            if remaining_slots <= 0:
                break
            for sr in secondary_routes:
                if idx < len(route_results[sr]):
                    aid = int(route_results[sr][idx])
                    if aid not in seen:
                        selected.append(aid)
                        seen.add(aid)
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break

    # Assign scores based on position (higher = better)
    scores = {}
    n = len(selected)
    for rank_idx, aid in enumerate(selected):
        scores[aid] = (n - rank_idx) / n  # linear decay from 1.0 to 1/n

    return scores, blend_weights
