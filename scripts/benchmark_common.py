"""Shared utilities for benchmark scripts.

Provides common functions for loading requests, pre-computing tool results,
formatting results as markdown, and parsing Claude's ranked_ads output.
"""

import json
import re
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pselect_main_route import pselect_main_route
from tools.forced_retrieval import forced_retrieval
from tools.anti_negative_scorer import anti_negative_scorer
from tools.cluster_explorer import cluster_explorer
from tools.engagement_analyzer import engagement_pattern_analyzer
from tools.pool_stats import ads_pool_stats
from tools.prod_model_ranker import prod_model_ranker


def load_request(npz_path):
    """Load request data from npz file with history/test label split."""
    data = np.load(npz_path)
    rd = {
        "request_id": int(data["request_id"]),
        "user_emb": data["user_emb"],
        "ad_embs": data["ad_embs"],
        "ad_ids": data["ad_ids"],
    }
    if "history_labels" in data:
        rd["labels"] = data["history_labels"]
        rd["test_labels"] = data["test_labels"]
    else:
        rd["labels"] = data["labels"]
        rd["test_labels"] = data["labels"]
    return rd


def precompute_tool_results(rd):
    """Run all retrieval tools and return results dict."""
    results = {}

    results["ads_pool_stats"] = ads_pool_stats(
        rd["ad_embs"], rd["ad_ids"], rd["user_emb"], n_clusters=5
    )
    results["engagement_pattern_analyzer"] = engagement_pattern_analyzer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"]
    )
    results["forced_retrieval"] = forced_retrieval(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"], top_k=150
    )
    results["pselect_main_route"] = pselect_main_route(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], top_k=150
    )
    results["cluster_explorer"] = cluster_explorer(
        rd["ad_embs"], rd["ad_ids"], n_clusters=5, top_k_per_cluster=30,
        labels=rd["labels"]
    )
    results["anti_negative_scorer"] = anti_negative_scorer(
        rd["user_emb"], rd["ad_embs"], rd["ad_ids"], rd["labels"],
        alpha=0.3, top_k=100
    )
    results["prod_model_ranker"] = prod_model_ranker(
        rd["ad_ids"], top_k=100, request_id=rd["request_id"]
    )

    return results


def format_tool_results(rd, results):
    """Format pre-computed tool results as markdown for Claude."""
    sections = []
    request_id = rd["request_id"]

    # Pool stats
    ps = results["ads_pool_stats"]
    sections.append(f"""# Request {request_id}

## ads_pool_stats
- Total ads: {ps['total_ads']}
- Similarity stats: mean={ps['similarity_stats']['mean']:.4f}, std={ps['similarity_stats']['std']:.4f}
- Clusters: {json.dumps([{k: c[k] for k in ['cluster_id', 'size', 'avg_similarity']} for c in ps.get('cluster_distribution', [])], default=str)}""")

    # Engagement analyzer
    ea = results["engagement_pattern_analyzer"]
    sections.append(f"""## engagement_pattern_analyzer
- Positive: {ea['n_positive']}, Negative: {ea['n_negative']}
- similarity_gap: {ea.get('similarity_gap', 0):.6f}
- gap_ratio: {ea.get('gap_ratio', 0):.4f}
- overlap_fraction: {ea.get('overlap_fraction', 0):.4f}
- engagement_rate_variance: {ea.get('engagement_rate_variance', 0):.6f}
- top_engaged_cluster_ids: {ea.get('top_engaged_cluster_ids', [])}
- top_positive_ad_ids: {ea.get('top_positive_ad_ids', [])[:5]}
- Engagement by cluster: {json.dumps(ea.get('engagement_by_cluster', []), default=str)}""")

    # Forced retrieval
    fr = results["forced_retrieval"]
    fr_results = fr.get("results", [])
    sections.append(f"""## forced_retrieval (top_k=150)
- centroid_gap: {fr.get('centroid_gap', 0):.6f}
- user_emb_gap: {fr.get('user_emb_gap', 0):.6f}
- centroid_vs_user_correlation: {fr.get('centroid_vs_user_correlation', 0):.4f}
- n_positives_used: {fr.get('n_positives_used', 0)}
- Top 10 results: {json.dumps(fr_results[:10], default=str)}
- Full result ad_ids ({len(fr_results)} total): {[r['ad_id'] for r in fr_results]}""")

    # PSelect main route
    es = results["pselect_main_route"]
    es_results = es.get("results", [])
    sections.append(f"""## pselect_main_route (top_k=150)
- score_range: {es.get('score_range', [])}
- score_std: {es.get('score_std', 0):.6f}
- top_bottom_gap: {es.get('top_bottom_gap', 0):.6f}
- Top 10 results: {json.dumps(es_results[:10], default=str)}
- Full result ad_ids ({len(es_results)} total): {[r['ad_id'] for r in es_results]}""")

    # Cluster explorer
    ce = results["cluster_explorer"]
    sections.append(f"""## cluster_explorer (n_clusters=5, top_k_per_cluster=30)
- Clusters: {json.dumps(ce.get('clusters', []), default=str)}
- Ads ({len(ce.get('ads', []))} total): {json.dumps(ce.get('ads', [])[:20], default=str)}...""")

    # Anti-negative scorer
    an = results["anti_negative_scorer"]
    an_results = an.get("results", [])
    sections.append(f"""## anti_negative_scorer (alpha=0.3, top_k=100)
- pos_neg_centroid_similarity: {an.get('pos_neg_centroid_similarity', 0):.4f}
- Full result ad_ids ({len(an_results)} total): {[r['ad_id'] for r in an_results]}""")

    # Prod model ranker
    pm = results["prod_model_ranker"]
    pm_results = pm.get("results", [])
    sections.append(f"""## prod_model_ranker (top_k=100)
- available: {pm.get('available', False)}
- coverage: {pm.get('coverage', 0):.2%}
- Full result ad_ids ({len(pm_results)} total): {[r['ad_id'] for r in pm_results]}""")

    return "\n\n".join(sections)


def parse_ranked_ads(text):
    """Parse ranked_ads JSON from Claude's response text."""
    # Try ```json block
    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "ranked_ads" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    # Try raw JSON
    try:
        idx = text.find('{"ranked_ads"')
        if idx < 0:
            idx = text.find('"ranked_ads"')
            if idx > 0:
                idx = text.rfind('{', 0, idx)
        if idx >= 0:
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[idx:i + 1])
    except (json.JSONDecodeError, ValueError):
        pass

    return None
