#!/usr/bin/env python3
"""
MCP server that exposes retrieval tools for Claude Code.

Usage:
    python3 tools/mcp_server.py --request-npz data/local/model/raw/request_1005207739.npz

This starts an MCP stdio server that Claude Code can connect to.
The server exposes all 6 retrieval tools operating on the specified request's data.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from tools.prod_model_ranker import prod_model_ranker
from tools.pipeline_simulator import pipeline_simulator
from tools.hsnn_cluster_scorer import hsnn_cluster_scorer
from tools.ml_reducer import ml_reducer
from tools.parallel_routes_blender import parallel_routes_blender


# Global request data (loaded once at startup)
REQUEST_DATA = None


def load_request(npz_path):
    """Load request data from npz file."""
    data = np.load(npz_path)
    return {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
        "labels": data["labels"],
    }


def handle_initialize(msg_id):
    """Handle MCP initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ads-retrieval-tools", "version": "1.0.0"},
        },
    }


def handle_tools_list(msg_id):
    """Handle tools/list request."""
    tools = [
        {
            "name": "embedding_similarity_search",
            "description": "Search for ads similar to the user embedding by cosine similarity. Returns ranked list of (ad_id, score).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "description": "Number of top ads to return", "default": 100},
                    "threshold": {"type": "number", "description": "Minimum cosine similarity threshold (optional)"},
                },
            },
        },
        {
            "name": "feature_filter",
            "description": "Filter ads by computed features. Supported features: cosine_score (similarity to user), embedding_norm, embedding_mean.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "feature_name": {"type": "string", "enum": ["cosine_score", "embedding_norm", "embedding_mean"]},
                    "operator": {"type": "string", "enum": ["gt", "lt", "between"]},
                    "value": {"type": "number", "description": "Threshold value (for gt/lt)"},
                    "value_high": {"type": "number", "description": "Upper bound (for between)"},
                    "top_k": {"type": "integer", "default": 50},
                },
                "required": ["feature_name", "operator", "value"],
            },
        },
        {
            "name": "cluster_explorer",
            "description": "Cluster all candidate ads by embedding using k-means. Returns cluster info and top ads per cluster. Use to discover ad segments and find diverse candidates.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_clusters": {"type": "integer", "default": 5, "description": "Number of clusters"},
                    "target_cluster_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific clusters to return ads from (optional, returns all if omitted)",
                    },
                    "top_k_per_cluster": {"type": "integer", "default": 20, "description": "Ads per cluster"},
                },
            },
        },
        {
            "name": "similar_ads_lookup",
            "description": "Find ads similar to specific reference ads by cosine similarity. Use to expand from known good ads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_ad_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Ad IDs to find similar ads for",
                    },
                    "top_k_per_ref": {"type": "integer", "default": 10},
                    "exclude_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Ad IDs to exclude from results",
                    },
                },
                "required": ["reference_ad_ids"],
            },
        },
        {
            "name": "engagement_pattern_analyzer",
            "description": "Analyze the user's engagement patterns. Shows positive vs negative ad statistics, cluster distribution of engaged ads, and similarity gaps. Call this first to understand the user.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "ads_pool_stats",
            "description": "Get summary statistics of the full candidate ad pool: total count, similarity distribution, cluster sizes. Call this first to understand the landscape.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_clusters": {"type": "integer", "default": 5},
                },
            },
        },
        {
            "name": "lookup_similar_requests",
            "description": "Find past requests with similar signal characteristics (similarity_gap, positive_rate, n_candidates). Returns top 3 most similar historical requests with their strategies and outcomes, plus a pattern summary grouped by gap buckets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "similarity_gap": {"type": "number", "description": "Gap between avg positive and negative cosine similarity"},
                    "positive_rate": {"type": "number", "description": "Fraction of candidates that are positive"},
                    "n_candidates": {"type": "integer", "description": "Total number of candidate ads"},
                },
                "required": ["similarity_gap", "positive_rate", "n_candidates"],
            },
        },
        {
            "name": "fr_centroid_search",
            "description": "Simulates production Forced Retrieval by using the centroid of positively-engaged ad embeddings as a second query vector. Provides a completely independent query from user_emb.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "description": "Number of top results to return", "default": 100},
                },
            },
        },
        {
            "name": "anti_negative_scorer",
            "description": "Directional scoring: pushes toward engaged ads, away from non-engaged. Scores ads by sim(ad, pos_centroid) - alpha * sim(ad, neg_centroid).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alpha": {"type": "number", "description": "Weight for the negative centroid penalty", "default": 0.3},
                    "top_k": {"type": "integer", "description": "Number of top results to return", "default": 100},
                },
            },
        },
        {
            "name": "mmr_reranker",
            "description": "Re-rank a set of candidate ads using Maximal Marginal Relevance (MMR) for diversity. Balances relevance to user with diversity among selected ads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "candidate_ad_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Ad IDs to re-rank (from other tools)",
                    },
                    "lambda_param": {"type": "number", "description": "Trade-off between relevance and diversity", "default": 0.7},
                    "top_k": {"type": "integer", "description": "Number of ads to select", "default": 100},
                },
                "required": ["candidate_ad_ids"],
            },
        },
        {
            "name": "prod_model_ranker",
            "description": "Rank ads by production model prediction (calibrated CTR from SlimDSNN PM stage). This is the strongest per-ad quality signal — what the actual production model thinks. Returns ranked list by prod_prediction score. May not be available for all requests.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "description": "Number of top ads to return", "default": 100},
                },
            },
        },
        {
            "name": "pipeline_simulator",
            "description": "Simulate the production cascaded pipeline (AP→PM→AI→AF). Shows per-stage survival of positive ads, cross-stage rank correlation, and drop-off analysis. Use to understand where good ads are lost in the e2e pipeline.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": ["all", "AP", "PM", "AI", "AF"], "default": "all", "description": "Which stage(s) to simulate"},
                    "pm_budget": {"type": "integer", "default": 500, "description": "Ads surviving PM"},
                    "ai_budget": {"type": "integer", "default": 100, "description": "Ads surviving AI"},
                    "af_budget": {"type": "integer", "default": 20, "description": "Ads surviving AF"},
                },
            },
        },
        {
            "name": "hsnn_cluster_scorer",
            "description": "Simulate HSNN 2-level hierarchical cluster scoring for sublinear retrieval. Clusters ads, scores centroids, expands only top-K clusters. Reports computational savings from pruning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_coarse": {"type": "integer", "default": 10, "description": "Number of coarse clusters"},
                    "n_fine_per_coarse": {"type": "integer", "default": 5, "description": "Fine sub-clusters per coarse"},
                    "expand_top_k_coarse": {"type": "integer", "default": 3, "description": "Top coarse clusters to expand"},
                    "top_k": {"type": "integer", "default": 100, "description": "Top ads to return"},
                },
            },
        },
        {
            "name": "ml_reducer",
            "description": "Simulate ML-driven truncation (ML Reducer). Scores candidates by combined signals and removes the bottom fraction. Reports value preservation and positive survival rate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "candidate_ad_ids": {"type": "array", "items": {"type": "integer"}, "description": "Ad IDs to reduce (all if omitted)"},
                    "target_stage": {"type": "string", "enum": ["PM", "AI"], "default": "PM", "description": "Stage to simulate"},
                    "reduction_rate": {"type": "number", "default": 0.5, "description": "Fraction to remove"},
                    "method": {"type": "string", "enum": ["ml_value", "heuristic_cosine", "heuristic_random"], "default": "ml_value"},
                },
            },
        },
        {
            "name": "parallel_routes_blender",
            "description": "Blend candidates from multiple retrieval routes (PRM + ML Blender). Supports RRF, learned ML blending, or priority-based. Reports per-route contribution and overlap.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "route_results": {"type": "object", "description": "Dict of route_name -> ordered list of ad_ids"},
                    "blending_strategy": {"type": "string", "enum": ["rrf", "ml_blender", "priority"], "default": "rrf"},
                    "target_pool_size": {"type": "integer", "default": 200},
                    "main_route_weight": {"type": "number", "default": 0.6, "description": "Weight for main route in priority mode"},
                },
                "required": ["route_results"],
            },
        },
    ]
    return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}


def handle_tool_call(msg_id, tool_name, arguments):
    """Handle tools/call request."""
    global REQUEST_DATA
    rd = REQUEST_DATA

    try:
        if tool_name == "embedding_similarity_search":
            result = embedding_similarity_search(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"],
                top_k=arguments.get("top_k", 100),
                threshold=arguments.get("threshold"),
            )
        elif tool_name == "feature_filter":
            result = feature_filter(
                rd["ad_embs"], rd["ad_ids"], rd["user_emb"],
                feature_name=arguments["feature_name"],
                operator=arguments["operator"],
                value=arguments["value"],
                top_k=arguments.get("top_k", 50),
            )
        elif tool_name == "cluster_explorer":
            result = cluster_explorer(
                rd["ad_embs"], rd["ad_ids"],
                n_clusters=arguments.get("n_clusters", 5),
                target_cluster_ids=arguments.get("target_cluster_ids"),
                top_k_per_cluster=arguments.get("top_k_per_cluster", 20),
                labels=rd["labels"],
            )
        elif tool_name == "similar_ads_lookup":
            result = similar_ads_lookup(
                rd["ad_embs"], rd["ad_ids"],
                reference_ad_ids=arguments["reference_ad_ids"],
                top_k_per_ref=arguments.get("top_k_per_ref", 10),
                exclude_ids=arguments.get("exclude_ids"),
            )
        elif tool_name == "engagement_pattern_analyzer":
            result = engagement_pattern_analyzer(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
            )
        elif tool_name == "ads_pool_stats":
            result = ads_pool_stats(
                rd["ad_embs"], rd["ad_ids"], rd["user_emb"],
                n_clusters=arguments.get("n_clusters", 5),
            )
        elif tool_name == "lookup_similar_requests":
            result = lookup_similar_requests(
                similarity_gap=arguments["similarity_gap"],
                positive_rate=arguments["positive_rate"],
                n_candidates=arguments["n_candidates"],
            )
        elif tool_name == "fr_centroid_search":
            result = fr_centroid_search(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
                top_k=arguments.get("top_k", 100),
            )
        elif tool_name == "anti_negative_scorer":
            result = anti_negative_scorer(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
                alpha=arguments.get("alpha", 0.3),
                top_k=arguments.get("top_k", 100),
            )
        elif tool_name == "mmr_reranker":
            result = mmr_reranker(
                rd["ad_embs"], rd["ad_ids"],
                candidate_ad_ids=arguments["candidate_ad_ids"],
                user_emb=rd["user_emb"],
                lambda_param=arguments.get("lambda_param", 0.7),
                top_k=arguments.get("top_k", 100),
            )
        elif tool_name == "prod_model_ranker":
            result = prod_model_ranker(
                rd["ad_ids"],
                top_k=arguments.get("top_k", 100),
                request_id=rd["request_id"],
            )
        elif tool_name == "pipeline_simulator":
            result = pipeline_simulator(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
                stage=arguments.get("stage", "all"),
                pm_budget=arguments.get("pm_budget", 500),
                ai_budget=arguments.get("ai_budget", 100),
                af_budget=arguments.get("af_budget", 20),
                request_id=rd["request_id"],
            )
        elif tool_name == "hsnn_cluster_scorer":
            result = hsnn_cluster_scorer(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
                n_coarse=arguments.get("n_coarse", 10),
                n_fine_per_coarse=arguments.get("n_fine_per_coarse", 5),
                expand_top_k_coarse=arguments.get("expand_top_k_coarse", 3),
                top_k=arguments.get("top_k", 100),
            )
        elif tool_name == "ml_reducer":
            result = ml_reducer(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
                candidate_ad_ids=arguments.get("candidate_ad_ids"),
                target_stage=arguments.get("target_stage", "PM"),
                reduction_rate=arguments.get("reduction_rate", 0.5),
                method=arguments.get("method", "ml_value"),
                request_id=rd["request_id"],
            )
        elif tool_name == "parallel_routes_blender":
            result = parallel_routes_blender(
                rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
                route_results=arguments["route_results"],
                blending_strategy=arguments.get("blending_strategy", "rrf"),
                target_pool_size=arguments.get("target_pool_size", 200),
                main_route_weight=arguments.get("main_route_weight", 0.6),
            )
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True},
            }

        result_text = json.dumps(result, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
        # Truncate very long results
        if len(result_text) > 50000:
            result_text = result_text[:50000] + "\n... (truncated)"

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": result_text}]},
        }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True},
        }


def main():
    global REQUEST_DATA

    parser = argparse.ArgumentParser(description="MCP server for ads retrieval tools")
    parser.add_argument("--request-npz", required=True, help="Path to request npz file")
    args = parser.parse_args()

    # Load request data
    REQUEST_DATA = load_request(args.request_npz)
    n_pos = int((REQUEST_DATA["labels"] == 1).sum())
    n_neg = int((REQUEST_DATA["labels"] == 0).sum())
    sys.stderr.write(
        f"Loaded request {REQUEST_DATA['request_id']}: "
        f"{len(REQUEST_DATA['ad_ids'])} ads ({n_pos} positive, {n_neg} negative)\n"
    )

    # MCP stdio loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            response = handle_initialize(msg_id)
        elif method == "notifications/initialized":
            continue  # No response needed
        elif method == "tools/list":
            response = handle_tools_list(msg_id)
        elif method == "tools/call":
            params = msg.get("params", {})
            response = handle_tool_call(msg_id, params.get("name", ""), params.get("arguments", {}))
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
