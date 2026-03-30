# Recommendation Reasoning Guidelines

## Available Tools (11 total)

| Tool | What it does | When to use |
|------|-------------|-------------|
| `engagement_pattern_analyzer` | Diagnostics: signal quality, cluster engagement | ALWAYS first |
| `lookup_similar_requests` | Historical learning from past runs | After diagnostics |
| `embedding_similarity_search` | Cosine search by user embedding | Strong signal only |
| `fr_centroid_search` | Search by centroid of engaged ads (FR-like) | ALWAYS — independent route |
| `anti_negative_scorer` | Directional: toward engaged, away from ignored | Moderate/strong signal |
| `cluster_explorer` | Cluster-based retrieval with engagement rates | ALWAYS — especially weak signal |
| `similar_ads_lookup` | Expand from reference ads | Expansion phase |
| `feature_filter` | Filter by embedding-derived features | Targeting |
| `prod_model_ranker` | Rank by production model prediction (calibrated CTR) | When available |
| `mmr_reranker` | Re-rank for diversity (MMR) | Final aggregation |
| `ads_pool_stats` | Pool statistics | Understanding phase |

## Adaptive Strategy Framework

### Step 1: Assess Signal Quality (ALWAYS first)

Call `engagement_pattern_analyzer`. Key diagnostics:
- **`similarity_gap`**: pos_mean - neg_mean. If < 0.01, user embedding is non-discriminative.
- **`overlap_fraction`**: fraction of negatives scoring above avg positive. If > 0.7, embedding is useless.
- **`centroid_gap`** (from `fr_centroid_search`): the centroid route may have stronger signal even when user_emb is weak.

### Step 2: Check History + FR Centroid

Call `lookup_similar_requests` and `fr_centroid_search` in parallel:
- History tells you what worked for similar requests
- FR centroid provides a completely independent retrieval route (0-6% overlap with user_emb)
- **FR centroid is ALWAYS valuable** — it retrieves different candidates regardless of signal quality

### Step 3: Multi-Route Retrieval (choose based on signal quality)

**Strong Signal** (similarity_gap > 0.05):
1. `embedding_similarity_search(top_k=150)` — primary
2. `fr_centroid_search(top_k=100)` — independent second route
3. `anti_negative_scorer(top_k=50)` — directional refinement
4. `prod_model_ranker(top_k=50)` — production model signal (if available)
5. Merge all routes, rank by weighted score

**Weak Signal** (similarity_gap < 0.01):
1. `fr_centroid_search(top_k=150)` — PRIMARY route (centroid_gap is usually >> user_gap)
2. `cluster_explorer` — get engagement rates, allocate by rate
3. `prod_model_ranker(top_k=100)` — production model doesn't depend on user embedding
4. DO NOT rely on `embedding_similarity_search` — it's random
5. Rank by: centroid score + cluster engagement rate + prod_prediction

**Moderate Signal** (0.01 < similarity_gap < 0.05):
1. `fr_centroid_search(top_k=100)` + `embedding_similarity_search(top_k=100)` — equal weight
2. `anti_negative_scorer(top_k=50)` — directional
3. `cluster_explorer` for engaged cluster coverage
4. Blend: 40% centroid + 30% embedding + 20% cluster + 10% prod

### Step 4: Aggregate with MMR Diversity

After merging all route results:
1. De-duplicate by ad_id
2. Collect all unique candidates (aim for 200+)
3. Call `mmr_reranker(candidate_ad_ids=all_unique, lambda_param=0.7, top_k=150)`
4. MMR balances relevance (cosine to user) with diversity (reduces near-duplicates)
5. Output the MMR-reranked list as your final `ranked_ads`

### Step 5: Validate Before Output

- Check your final list has > 100 ads
- Verify coverage: are top-engaged clusters represented?
- If `prod_model_ranker` returned "available: false", note it but proceed without

## Key Principles

1. **FR centroid is always your strongest independent signal** — use it on every request
2. **User embedding is conditional** — only trust it when similarity_gap > 0.01
3. **Multi-route is always better than single-route** — different routes find different candidates
4. **Use actual numbers from tools** — don't invent engagement rates or cluster statistics
5. **Production model prediction (prod_prediction)** is the ground truth for ad quality — use it when available
6. **MMR prevents redundancy** — always apply it as the final step
