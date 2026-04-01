from typing import Dict, Any, List

from tools.pselect_main_route import pselect_main_route
from tools.feature_filter import feature_filter
from tools.cluster_explorer import cluster_explorer
from tools.similar_ads import similar_ads_lookup
from tools.engagement_analyzer import engagement_pattern_analyzer
from tools.pool_stats import ads_pool_stats
from tools.history_lookup import lookup_similar_requests
from tools.forced_retrieval import forced_retrieval
from tools.anti_negative_scorer import anti_negative_scorer
from tools.mmr_reranker import mmr_reranker
from tools.pipeline_simulator import pipeline_simulator
from tools.hsnn_cluster_scorer import hsnn_cluster_scorer
from tools.ml_reducer import ml_reducer
from tools.parallel_routes_blender import parallel_routes_blender


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "pselect_main_route",
        "description": (
            "Search ads by cosine similarity to the user embedding. "
            "Returns the top-K most similar ads sorted by score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 100,
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum cosine similarity threshold. If null, no threshold is applied.",
                    "nullable": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "feature_filter",
        "description": (
            "Filter ads by embedding-derived features. Supported features: "
            "cosine_score (similarity to user), embedding_norm, embedding_mean. "
            "Supported operators: gt, lt, between."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "enum": ["cosine_score", "embedding_norm", "embedding_mean"],
                    "description": "The feature to filter on.",
                },
                "operator": {
                    "type": "string",
                    "enum": ["gt", "lt", "between"],
                    "description": "Comparison operator.",
                },
                "value": {
                    "description": (
                        "Threshold value. Scalar for gt/lt, "
                        "array of [low, high] for between."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results.",
                    "default": 50,
                },
            },
            "required": ["feature_name", "operator", "value"],
        },
    },
    {
        "name": "cluster_explorer",
        "description": (
            "Explore ad embeddings via k-means clustering. Returns cluster metadata "
            "and ads closest to each centroid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters.",
                    "default": 5,
                },
                "target_cluster_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "If provided, only return ads from these clusters.",
                    "nullable": True,
                },
                "top_k_per_cluster": {
                    "type": "integer",
                    "description": "Max ads to return per cluster.",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
    {
        "name": "similar_ads_lookup",
        "description": (
            "Find similar ads for each reference ad by cosine similarity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reference_ad_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Ad IDs to find similar ads for.",
                },
                "top_k_per_ref": {
                    "type": "integer",
                    "description": "Number of similar ads per reference.",
                    "default": 10,
                },
                "exclude_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Ad IDs to exclude from results.",
                    "nullable": True,
                },
            },
            "required": ["reference_ad_ids"],
        },
    },
    {
        "name": "engagement_pattern_analyzer",
        "description": (
            "Analyze engagement patterns: compare positive vs negative ads by "
            "cluster distribution, similarity to user, and feature stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters for analysis.",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "ads_pool_stats",
        "description": (
            "Compute summary statistics for the candidate ad pool: total count, "
            "similarity distribution, and cluster sizes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters for distribution analysis.",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "lookup_similar_requests",
        "description": (
            "Find past requests with similar signal characteristics. Returns top 3 "
            "most similar historical requests with their strategies and outcomes, "
            "plus a pattern summary grouped by similarity gap buckets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "similarity_gap": {
                    "type": "number",
                    "description": "Gap between avg positive and negative cosine similarity.",
                },
                "positive_rate": {
                    "type": "number",
                    "description": "Fraction of candidates that are positive (engaged).",
                },
                "n_candidates": {
                    "type": "integer",
                    "description": "Total number of candidate ads.",
                },
            },
            "required": ["similarity_gap", "positive_rate", "n_candidates"],
        },
    },
    {
        "name": "forced_retrieval",
        "description": (
            "Simulates production Forced Retrieval by using the centroid of "
            "positively-engaged ad embeddings as a second query vector. "
            "Provides a completely independent query from user_emb."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 100,
                },
            },
            "required": [],
        },
    },
    {
        "name": "anti_negative_scorer",
        "description": (
            "Directional scoring: pushes toward engaged ads, away from non-engaged. "
            "Scores ads by sim(ad, pos_centroid) - alpha * sim(ad, neg_centroid)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "alpha": {
                    "type": "number",
                    "description": "Weight for the negative centroid penalty.",
                    "default": 0.3,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 100,
                },
            },
            "required": [],
        },
    },
    {
        "name": "mmr_reranker",
        "description": (
            "Re-rank a set of candidate ads using Maximal Marginal Relevance (MMR) "
            "for diversity. Balances relevance to user with diversity among selected ads."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_ad_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Ad IDs to re-rank (from other tools).",
                },
                "lambda_param": {
                    "type": "number",
                    "description": "Trade-off between relevance and diversity (0.7 = 70% relevance).",
                    "default": 0.7,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of ads to select.",
                    "default": 100,
                },
            },
            "required": ["candidate_ad_ids"],
        },
    },
    {
        "name": "pipeline_simulator",
        "description": (
            "Simulate the production cascaded pipeline (AP→PM→AI→AF). "
            "Shows per-stage survival of positive ads, cross-stage rank correlation, "
            "and drop-off analysis. Use to understand where good ads are lost."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": ["all", "AP", "PM", "AI", "AF"],
                    "description": "Which stage(s) to simulate. 'all' runs full cascade.",
                    "default": "all",
                },
                "pm_budget": {
                    "type": "integer",
                    "description": "Number of ads surviving PM truncation.",
                    "default": 500,
                },
                "ai_budget": {
                    "type": "integer",
                    "description": "Number of ads surviving AI stage.",
                    "default": 100,
                },
                "af_budget": {
                    "type": "integer",
                    "description": "Number of ads surviving AF stage.",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
    {
        "name": "hsnn_cluster_scorer",
        "description": (
            "Simulate HSNN 2-level hierarchical cluster scoring for sublinear retrieval. "
            "Clusters ads into coarse groups, scores centroids, expands only top-K clusters, "
            "then sub-clusters for fine scoring. Reports computational savings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_coarse": {
                    "type": "integer",
                    "description": "Number of coarse clusters.",
                    "default": 10,
                },
                "n_fine_per_coarse": {
                    "type": "integer",
                    "description": "Number of fine sub-clusters per expanded coarse cluster.",
                    "default": 5,
                },
                "expand_top_k_coarse": {
                    "type": "integer",
                    "description": "Number of top coarse clusters to expand (rest pruned).",
                    "default": 3,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top ads to return.",
                    "default": 100,
                },
            },
            "required": [],
        },
    },
    {
        "name": "ml_reducer",
        "description": (
            "Simulate ML-driven truncation (ML Reducer). Scores candidates by combined "
            "signals and removes the bottom fraction. Compares against heuristic truncation. "
            "Reports value preservation and positive survival."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_ad_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Ad IDs to reduce. If omitted, uses all ads.",
                    "nullable": True,
                },
                "target_stage": {
                    "type": "string",
                    "enum": ["PM", "AI"],
                    "description": "Which stage this reduction simulates.",
                    "default": "PM",
                },
                "reduction_rate": {
                    "type": "number",
                    "description": "Fraction of ads to remove (0.5 = remove bottom 50%).",
                    "default": 0.5,
                },
                "method": {
                    "type": "string",
                    "enum": ["ml_value", "heuristic_cosine", "heuristic_random"],
                    "description": "Scoring method for truncation.",
                    "default": "ml_value",
                },
            },
            "required": [],
        },
    },
    {
        "name": "parallel_routes_blender",
        "description": (
            "Blend candidates from multiple retrieval routes (PRM + ML Blender). "
            "Supports RRF, learned ML blending, or priority-based allocation. "
            "Reports per-route contribution and overlap analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "route_results": {
                    "type": "object",
                    "description": (
                        "Dict of route_name -> ordered list of ad_ids. "
                        "E.g. {'embedding_search': [id1, id2], 'fr_centroid': [id3, id4]}"
                    ),
                },
                "blending_strategy": {
                    "type": "string",
                    "enum": ["rrf", "ml_blender", "priority"],
                    "description": "Blending strategy.",
                    "default": "rrf",
                },
                "target_pool_size": {
                    "type": "integer",
                    "description": "Target number of blended candidates.",
                    "default": 200,
                },
                "main_route_weight": {
                    "type": "number",
                    "description": "Weight for main route in priority mode.",
                    "default": 0.6,
                },
            },
            "required": ["route_results"],
        },
    },
]


def execute_tool(tool_name: str, args: Dict[str, Any], request_data: Dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate function.

    Args:
        tool_name: Name of the tool to execute.
        args: Arguments from the Claude API tool_use block.
        request_data: Dict with keys: user_emb, ad_embs, ad_ids, labels.

    Returns:
        The tool's return value.
    """
    user_emb = request_data["user_emb"]
    ad_embs = request_data["ad_embs"]
    ad_ids = request_data["ad_ids"]
    labels = request_data["labels"]

    if tool_name == "pselect_main_route":
        return pselect_main_route(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            top_k=args.get("top_k", 100),
            threshold=args.get("threshold"),
        )

    elif tool_name == "feature_filter":
        value = args["value"]
        if args["operator"] == "between" and isinstance(value, list):
            value = tuple(value)
        return feature_filter(
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            user_emb=user_emb,
            feature_name=args["feature_name"],
            operator=args["operator"],
            value=value,
            top_k=args.get("top_k", 50),
        )

    elif tool_name == "cluster_explorer":
        return cluster_explorer(
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            n_clusters=args.get("n_clusters", 5),
            target_cluster_ids=args.get("target_cluster_ids"),
            top_k_per_cluster=args.get("top_k_per_cluster", 20),
            labels=labels,
        )

    elif tool_name == "similar_ads_lookup":
        return similar_ads_lookup(
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            reference_ad_ids=args["reference_ad_ids"],
            top_k_per_ref=args.get("top_k_per_ref", 10),
            exclude_ids=args.get("exclude_ids"),
        )

    elif tool_name == "engagement_pattern_analyzer":
        return engagement_pattern_analyzer(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            n_clusters=args.get("n_clusters", 5),
        )

    elif tool_name == "ads_pool_stats":
        return ads_pool_stats(
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            user_emb=user_emb,
            n_clusters=args.get("n_clusters", 5),
        )

    elif tool_name == "lookup_similar_requests":
        return lookup_similar_requests(
            similarity_gap=args["similarity_gap"],
            positive_rate=args["positive_rate"],
            n_candidates=args["n_candidates"],
        )

    elif tool_name == "forced_retrieval":
        return forced_retrieval(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            top_k=args.get("top_k", 100),
        )

    elif tool_name == "anti_negative_scorer":
        return anti_negative_scorer(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            alpha=args.get("alpha", 0.3),
            top_k=args.get("top_k", 100),
        )

    elif tool_name == "mmr_reranker":
        return mmr_reranker(
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            candidate_ad_ids=args["candidate_ad_ids"],
            user_emb=user_emb,
            lambda_param=args.get("lambda_param", 0.7),
            top_k=args.get("top_k", 100),
        )

    elif tool_name == "pipeline_simulator":
        return pipeline_simulator(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            stage=args.get("stage", "all"),
            pm_budget=args.get("pm_budget", 500),
            ai_budget=args.get("ai_budget", 100),
            af_budget=args.get("af_budget", 20),
            prod_data_dir=request_data.get("prod_data_dir", "data/local/model/enriched"),
            request_id=request_data.get("request_id"),
        )

    elif tool_name == "hsnn_cluster_scorer":
        return hsnn_cluster_scorer(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            n_coarse=args.get("n_coarse", 10),
            n_fine_per_coarse=args.get("n_fine_per_coarse", 5),
            expand_top_k_coarse=args.get("expand_top_k_coarse", 3),
            top_k=args.get("top_k", 100),
        )

    elif tool_name == "ml_reducer":
        return ml_reducer(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            candidate_ad_ids=args.get("candidate_ad_ids"),
            target_stage=args.get("target_stage", "PM"),
            reduction_rate=args.get("reduction_rate", 0.5),
            method=args.get("method", "ml_value"),
            prod_data_dir=request_data.get("prod_data_dir", "data/local/model/enriched"),
            request_id=request_data.get("request_id"),
        )

    elif tool_name == "parallel_routes_blender":
        return parallel_routes_blender(
            user_emb=user_emb,
            ad_embs=ad_embs,
            ad_ids=ad_ids,
            labels=labels,
            route_results=args["route_results"],
            blending_strategy=args.get("blending_strategy", "rrf"),
            target_pool_size=args.get("target_pool_size", 200),
            main_route_weight=args.get("main_route_weight", 0.6),
        )

    else:
        raise ValueError(f"Unknown tool: {tool_name}")
