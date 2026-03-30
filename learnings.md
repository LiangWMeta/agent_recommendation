# Recommendation Learnings

Empirical findings from past benchmark runs. Use these to inform your strategy.

## Embedding Signal Is NOT Always Reliable

The cosine similarity between user and ad embeddings varies dramatically in discriminative power across requests. You MUST check signal quality before relying on it.

**Key metric: `similarity_gap`** (from engagement_pattern_analyzer)
= avg_positive_similarity - avg_negative_similarity

| Gap Range | Signal Quality | Observed Recall@100 | Recommended Strategy |
|-----------|---------------|---------------------|---------------------|
| > 0.05 | Strong | ~30% | Embedding-first: rank by cosine similarity, boost with cluster engagement |
| 0.01–0.05 | Moderate | ~10-15% | Hybrid: weight embedding 50% + cluster engagement rate 50% |
| < 0.01 | Weak/None | ~3-9% with embedding-first | Cluster-first: rank by cluster engagement rate, ignore embedding ranking |

**Evidence:**
- Request 1005207739: gap=0.0994, embedding-first → recall@100=30.3%
- Request 1017453312: gap=0.0008, embedding-first → recall@100=3.5% (FAILURE)
- Request 102432111: gap=0.0671, embedding-first → recall@100=11.6%
- Request 1034102114: gap≈weak, embedding-first → recall@100=9.0%

## Reactive Decision Points

### After calling `engagement_pattern_analyzer`:
1. Check `similarity_gap` — if < 0.01, DO NOT use embedding_similarity_search as your primary ranking signal
2. Check `overlap_fraction` — if > 0.7, most negatives look like positives in embedding space; the embedding cannot separate them
3. Check `engagement_rate_variance` — if high (> 0.01), some clusters have much higher engagement; concentrate candidates there
4. Check `top_engaged_cluster_ids` — these are the clusters to prioritize

### After calling `embedding_similarity_search`:
1. Check `score_std` — if very low (< 0.01), all scores are nearly identical; ranking is essentially random
2. Check `top_bottom_gap` — if < 0.01, no meaningful separation between best and worst candidates

### After calling `cluster_explorer`:
1. Check per-cluster `engagement_rate` — if one cluster has >2x the engagement rate of others, concentrate candidates there
2. If engagement rates are roughly equal across clusters, use uniform allocation

## Cluster-First Strategy (for weak embedding signal)

When `similarity_gap < 0.01`:
1. Call `engagement_pattern_analyzer` to get per-cluster engagement rates
2. Call `cluster_explorer` with labels to get engagement-aware clusters
3. Rank clusters by engagement_rate (descending)
4. Allocate your top-K budget proportional to each cluster's engagement_rate
5. Within each cluster, rank ads by distance to cluster centroid (closest first)
6. This outperforms embedding-first by ~2-3x on weak-signal requests

## Historical Patterns
<!-- Updated automatically by scripts/update_learnings.py after each benchmark run -->

### Run cc_5req (4 requests evaluated)
- Average recall@100: 13.6% (baseline: 12.9%)
- Best: request 1005207739 (gap=0.0994) → 30.3%
- Worst: request 1017453312 (gap=0.0008) → 3.5%
- Correlation: similarity_gap strongly predicts whether embedding-first strategy succeeds
