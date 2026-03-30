from typing import Dict, Any, List

from tools.embedding_search import embedding_similarity_search
from tools.feature_filter import feature_filter
from tools.cluster_explorer import cluster_explorer
from tools.similar_ads import similar_ads_lookup
from tools.engagement_analyzer import engagement_pattern_analyzer
from tools.pool_stats import ads_pool_stats
from tools.history_lookup import lookup_similar_requests
from tools.fr_centroid_search import fr_centroid_search
from tools.anti_negative_scorer import anti_negative_scorer
from tools.mmr_reranker import mmr_reranker


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "embedding_similarity_search",
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
        "name": "fr_centroid_search",
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

    if tool_name == "embedding_similarity_search":
        return embedding_similarity_search(
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

    elif tool_name == "fr_centroid_search":
        return fr_centroid_search(
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

    else:
        raise ValueError(f"Unknown tool: {tool_name}")
