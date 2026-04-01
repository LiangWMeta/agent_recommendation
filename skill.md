# Recommendation Reasoning Guidelines

## Available Tools (15 total)

### Production Track
| Tool | What it does | Stage | When to use |
|------|-------------|-------|-------------|
| `pselect_main_route` | Cosine search by user embedding | AP/PM | Strong signal only |
| `forced_retrieval` | Search by centroid of engaged ads (FR-like) | AP | ALWAYS — independent route |
| `prod_model_ranker` | Rank by production model prediction (calibrated CTR) | PM | When available |
| `hsnn_cluster_scorer` | Hierarchical 2-level cluster scoring (HSNN) | AP/PM | ALWAYS — sublinear exploration |
| `pipeline_simulator` | Simulate cascaded pipeline (AP→PM→AI→AF) | All | E2E reasoning |
| `ml_reducer` | ML-driven truncation (replaces heuristic) | PM/AI | After multi-route retrieval |
| `parallel_routes_blender` | Multi-route blending (PRM + ML Blender) | PM | Aggregation phase |

### Exploration Track
| Tool | What it does | When to use |
|------|-------------|-------------|
| `anti_negative_scorer` | Directional: toward engaged, away from ignored | Moderate/strong signal — compare vs production routes |
| `cluster_explorer` | Flat K-means with engagement rates | HSNN baseline comparison |
| `similar_ads_lookup` | Expand from reference ads | Ad-to-ad expansion hypothesis |
| `mmr_reranker` | Re-rank for diversity (MMR) | Final diversity pass |
| `feature_filter` | Filter by embedding-derived features | Feature hypothesis testing |

### Diagnostics Track
| Tool | What it does | When to use |
|------|-------------|-------------|
| `engagement_pattern_analyzer` | Signal quality diagnostics | ALWAYS first |
| `ads_pool_stats` | Pool statistics | Understanding phase |
| `lookup_similar_requests` | Historical learning from past runs | After diagnostics |

## Adaptive Strategy Framework

### Step 0: Load Context (ALWAYS before tools)

Read the context layer to understand the request before calling any retrieval tools:

1. **Ads Pool**: Read `ads_pool/pool_overview.md` — understand pool size, category distribution, what changed since last refresh
2. **User Profile**: Read `user/{request_id}/profile.md` + `interests.md` — who is this user, what do they care about
3. **Session**: Read `user/{request_id}/session_history.md` + `intent.md` (when available) — what they've seen, what they want now
4. **Engagement**: Read `user/{request_id}/engagement.md` — what they engaged with before, by category

This context informs every downstream decision: which routes to prioritize, how to interpret cluster results, what diversity means for this user.

### Step 1: Assess Signal Quality (ALWAYS first)

Call `engagement_pattern_analyzer`. Key diagnostics:
- **`similarity_gap`**: pos_mean - neg_mean. If < 0.01, user embedding is non-discriminative.
- **`overlap_fraction`**: fraction of negatives scoring above avg positive. If > 0.7, embedding is useless.
- **`centroid_gap`** (from `forced_retrieval`): the centroid route may have stronger signal even when user_emb is weak.

### Step 2: Check History + FR Centroid

Call `lookup_similar_requests` and `forced_retrieval` in parallel:
- History tells you what worked for similar requests
- FR centroid provides a completely independent retrieval route (0-6% overlap with user_emb)
- **FR centroid is ALWAYS valuable** — it retrieves different candidates regardless of signal quality

### Step 3: Multi-Route Retrieval (choose based on signal quality)

**Strong Signal** (similarity_gap > 0.05):
1. `pselect_main_route(top_k=150)` — primary
2. `forced_retrieval(top_k=100)` — independent second route
3. `hsnn_cluster_scorer(expand_top_k_coarse=3)` — hierarchical exploration
4. `prod_model_ranker(top_k=50)` — production model signal (if available)
5. Blend via `parallel_routes_blender`

**Weak Signal** (similarity_gap < 0.01):
1. `forced_retrieval(top_k=150)` — PRIMARY route
2. `hsnn_cluster_scorer(expand_top_k_coarse=5)` — expand more clusters for exploration
3. `prod_model_ranker(top_k=100)` — production model doesn't depend on user embedding
4. DO NOT rely on `pselect_main_route` — it's random
5. Blend via `parallel_routes_blender`

**Moderate Signal** (0.01 < similarity_gap < 0.05):
1. `forced_retrieval(top_k=100)` + `pselect_main_route(top_k=100)` — equal weight
2. `hsnn_cluster_scorer(expand_top_k_coarse=3)` — hierarchical exploration
3. `prod_model_ranker(top_k=50)` — if available
4. Blend via `parallel_routes_blender`

### Step 4: Pipeline-Aware Aggregation

After multi-route retrieval, reason about the full pipeline:

1. **Blend routes**: Call `parallel_routes_blender` with all route results. Use "rrf" for quick runs, "ml_blender" for research.
2. **Simulate truncation**: Call `ml_reducer` on the blended result to see which ads would survive PM truncation. Ads that don't survive `ml_reducer(reduction_rate=0.5)` probably don't survive PM in production.
3. **Check pipeline**: Call `pipeline_simulator(stage="all")` to see the full cascade. Note which ads survive each stage and where value is lost.
4. **Focus on survivors**: Prioritize ads that survive downstream stages. An ad ranking #1 at PM but #500 at AI will not be shown.
5. **Apply diversity**: Call `mmr_reranker` as final step for diversity.

### Step 5: Validate Before Output

- Check your final list has > 100 ads
- Verify coverage: are top-engaged clusters represented?
- If `prod_model_ranker` returned "available: false", note it but proceed without
- Check `pipeline_simulator` cross-stage consistency — flag any large rank inversions

## Exploration Reasoning

Use exploration track tools when:
- **Weak signal** (similarity_gap < 0.01) and production routes show low recall
- **Hypothesis testing** — you want to validate whether an approach adds value
- **Benchmarking** — comparing exploration routes against production routes

### When to use each exploration tool:

**`anti_negative_scorer`**: Use when engagement analyzer shows moderate+ signal. Compare its unique contributions vs production routes. If it finds positives that production routes miss AND they survive pipeline stages, that validates directional scoring as a PRM route.

**`cluster_explorer`**: Use as HSNN baseline. If flat clustering finds ads that HSNN misses, HSNN's hierarchy may need tuning. If HSNN largely subsumes it, confirms HSNN is sufficient.

**`similar_ads_lookup`**: Use for "more-like-this" expansion from top engaged ads. Track whether expanded ads survive pipeline stages. High survival rate validates Related Ads as a production route.

**`mmr_reranker`**: Apply as final diversity pass. Measure recall vs diversity tradeoff. If NDCG improves without recall loss, diversity mechanism is net positive.

**`feature_filter`**: Use for hypothesis testing (e.g., "do high-norm ads survive PM better?"). Results inform feature engineering for ML Reducer.

### Key principle:
Always compare exploration route output against production route output to measure **incremental value**. An exploration tool is only valuable if it finds positive ads that production routes miss.

## Key Principles

1. **Production routes first, exploration second** — use production track as your primary retrieval, exploration for research
2. **FR centroid is always your strongest independent signal** — use it on every request
3. **User embedding is conditional** — only trust it when similarity_gap > 0.01
4. **Pipeline survival matters more than single-stage scoring** — focus on ads that survive downstream
5. **Multi-route is always better than single-route** — different routes find different candidates
6. **Production model prediction (prod_prediction)** is the ground truth for ad quality — use it when available
7. **HSNN subsumes flat clustering** — prefer `hsnn_cluster_scorer` over `cluster_explorer` for retrieval
8. **Measure incremental value** — every exploration route should justify itself against production baselines
