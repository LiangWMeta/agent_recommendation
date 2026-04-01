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
1. Check `similarity_gap` — if < 0.01, DO NOT use pselect_main_route as your primary ranking signal
2. Check `overlap_fraction` — if > 0.7, most negatives look like positives in embedding space; the embedding cannot separate them
3. Check `engagement_rate_variance` — if high (> 0.01), some clusters have much higher engagement; concentrate candidates there
4. Check `top_engaged_cluster_ids` — these are the clusters to prioritize

### After calling `pselect_main_route`:
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

## Pipeline Diagnosis Findings (2026-04-01, 100 requests)

1. **PM truncation loses 54% of positives** — still the #1 bottleneck, but less severe than the 5-request pilot suggested (70%). At scale, PM survival is 45.9%.
2. **Forced Retrieval is 4.4x more valuable than PSelect** — 14.2 unique positives vs 3.2 per request, with only 7.1% overlap. Irreplaceable as an independent route.
3. **HSNN and PSelect are 73% redundant** — HSNN mostly rediscovers what PSelect already finds.
4. **ML Reducer outperforms heuristic by +9.4%** — preserves 77.6% vs 68.1% of positives at 50% reduction. Stronger at 100 requests than at 5.
5. **Exploration routes add +2.3% recall at scale** — the 10-request pilot showed <0.1%; at 100 requests, `similar_ads_lookup` adds +2.3% and `anti_negative_scorer` adds +1.7%. Small samples underestimate exploration value.
6. **Claude reasoning adds +1.9% recall over fixed-weight RRF** — agent (18.3%) vs RRF (16.4%). Adaptive route selection helps, especially for weak-signal requests.

## Historical Patterns
<!-- Updated automatically by scripts/update_learnings.py after each benchmark run -->

### Run agent_100 (90 requests, Claude-orchestrated, 2026-04-01)
- Recall@100: 18.29% (baseline: 12.85%, +42%)
- Recall@100 vs RRF: 18.29% vs 16.36% (+1.93%)
- Avg ranked ads: 161
- Time: 73.4s/request (batch-5, sonnet)
- 90/100 succeeded (2 batches had parse failures)

### Run cc_5req (4 requests evaluated)
- Average recall@100: 13.6% (baseline: 12.9%)
- Best: request 1005207739 (gap=0.0994) → 30.3%
- Worst: request 1017453312 (gap=0.0008) → 3.5%
- Correlation: similarity_gap strongly predicts whether embedding-first strategy succeeds
