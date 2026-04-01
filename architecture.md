# Ads Recommendation System Architecture

## Overview

The ads recommendation system connects users with relevant ads through a multi-stage pipeline: retrieval → ranking → auction → delivery. This framework models the **retrieval stage** — selecting and filtering candidate ads through a cascaded pipeline that progressively narrows the pool using increasingly complex models.

## Production Pipeline: AP → PM → AI → AF

The production ads retrieval system uses a cascaded multi-stage pipeline where each stage scores more candidates than it keeps:

```
AdPublisher (AP)     PreMatch (PM)      AdIndexer (AI)     AdFinder (AF)
~190K candidates  →  ~11K survivors  →  ~709 survivors  →  ~84 survivors
Lightest models      Medium models      Heavy models       Full ranking
(HSNN-AP)            (SlimDSNN PM)      (DSNN, verticals)  (Full model)
```

Each stage applies more expensive scoring and truncates the bottom portion. An ad must survive every stage to be shown. **Cross-stage value consistency** — ensuring ads are ranked similarly across stages — is a persistent production challenge.

### Stage Budgets (2K Pool Simulation)

For our ~2K candidate pool, we simulate production proportions:
- **AP**: All 2K candidates (100%)
- **PM**: Top 500 (~25%, simulating 11K/190K ratio, scaled up for statistical validity)
- **AI**: Top 100 (~5%, simulating 709/190K)
- **AF**: Top 20 (~1%, simulating 84/190K)

## Context Layer: Ads Pool Understanding + User Context

Before the agent calls any retrieval tools, it reads rich context from two sources that feed the entire orchestration flow.

```
ads_pool/                               user/{request_id}/
(refreshed regularly)                   (assembled per request)
├── embeddings.npz                      ├── profile.md
├── catalog.md                          ├── intent.md
├── pool_overview.md                    ├── interests.md
├── semantic_clusters.md                ├── session_history.md
├── pool_changes.md                     ├── engagement.md
                                        └── context.md
        │                                       │
        └──────────────┬────────────────────────┘
                       ▼
               Agent Orchestrator
        (reads both before calling tools)
```

### Ads Pool Understanding (`ads_pool/`)

A **living knowledge base** of the candidate ads pool. Refreshed regularly as ads update (new creatives, budget changes, targeting changes, expirations).

| File | Content | Refresh Frequency |
|------|---------|-------------------|
| `embeddings.npz` | Ad embeddings (32d PSelect) + ad_ids + labels | Per data pipeline run |
| `catalog.md` | All ads summarized: creative description, product category, advertiser objective, targeting criteria, budget, historical CTR, run duration | Daily or on ad updates |
| `pool_overview.md` | Pool-level summary: total ads, category distribution, budget ranges, top categories by count and spend | On each refresh |
| `semantic_clusters.md` | HSNN cluster → human-readable label mapping, per-cluster dominant category, avg CTR, size, example ads | On each refresh |
| `pool_changes.md` | Delta log: new ads added, ads expired, creative updates, budget changes, cluster membership shifts since last refresh | On each refresh |

**Current state**: `embeddings.npz` exists (from data pipeline). Rich metadata fields (creative description, product category, advertiser objective) are **schema-defined but not yet populated** — to be implemented when external metadata sources are integrated. Today, cluster labels and pool statistics are derived from embeddings.

**Schema for `catalog.md` (per-ad entry)**:
```markdown
### Ad {ad_id}
- **Category**: {product_category}          # e.g., fitness_app, ecommerce, gaming
- **Objective**: {advertiser_objective}      # awareness | consideration | conversion
- **Creative**: {creative_description}       # text summary of ad creative
- **Targeting**: {targeting_summary}         # age, gender, interest, geo
- **Budget**: ${daily_budget}/day, running {days_active} days
- **Performance**: CTR {ctr}%, {impressions} impressions
- **Cluster**: coarse_{cluster_id} / fine_{sub_cluster_id}
- **Embedding norm**: {norm}, cosine to avg user: {avg_cosine}
```

### User Context Folder (`user/{request_id}/`)

A **rich user profile** assembled per request. Provides the agent with semantic understanding of who the user is, what they want, and their engagement history — enabling reasoning beyond pure embedding similarity.

| File | Content | Source |
|------|---------|--------|
| `profile.md` | User demographics, device, platform, placement, embedding summary (32d PSelect, norm, top components) | User profile store + embeddings |
| `intent.md` | Current session intent: recent search queries, page context, inferred intent category (browse/search/purchase) | Session signals |
| `interests.md` | Stable interest profile: interest clusters from engagement history, top categories by engagement rate, cross-surface signals (FB + IG + Messenger) | Long-term engagement data |
| `session_history.md` | Ads shown this session, click/skip sequence, session duration, scroll depth, fatigue signals (repeated exposures, declining CTR) | Session logs |
| `engagement.md` | Engagement history enriched with ad semantics: top engaged ads with descriptions, engagement by category/time/creative type | Engagement logs + ads catalog |
| `context.md` | Request-level: pool size, similarity landscape, signal quality, placement type, auction constraints | Request metadata |

**Current state**: `profile.md`, `engagement.md`, `interests.md` (as `interest_clusters.md`), and `context.md` exist today — derived from embeddings and labels via `prepare_contexts.py`. `intent.md` and `session_history.md` are **schema-defined but not yet populated** — to be implemented when session-level signals are available.

### How Context Feeds the Agent Flow

1. **Before diagnostics**: Agent reads `pool_overview.md` + `profile.md` to understand the landscape
2. **During signal assessment**: Agent reads `engagement.md` + `interests.md` to contextualize similarity_gap
3. **During retrieval**: Agent reads `semantic_clusters.md` to interpret HSNN cluster results with semantic labels
4. **During pipeline reasoning**: Agent reads `intent.md` + `session_history.md` to assess fatigue and novelty
5. **During ranking**: Agent reads `catalog.md` to understand what categories are represented in the final list

## Tool Organization

All retrieval tools are organized into three tracks.

### Production Retrieval Track

Tools that directly simulate production retrieval components. These form the core e2e pipeline.

| Tool | Production Analog | Stage | Description |
|------|-------------------|-------|-------------|
| `embedding_similarity_search` | Main Route (TTSN PM / PSelect) | AP/PM | Cosine similarity between user and ad embeddings in 32d PSelect space |
| `fr_centroid_search` | Forced Retrieval (85% of impressions) | AP | Uses centroid of positively-engaged ads as independent query vector |
| `prod_model_ranker` | SlimDSNN PM scoring | PM | Production model calibrated CTR prediction — strongest per-ad quality signal |
| `hsnn_cluster_scorer` | HSNN (88% AP, 36% PM adoption) | AP/PM | 2-level hierarchical cluster scoring for sublinear-cost retrieval |
| `pipeline_simulator` | Cascaded pipeline (AP→PM→AI→AF) | All | Full cascade simulation with per-stage survival and cross-stage consistency |
| `ml_reducer` | ML Reducer/Truncator | PM/AI | ML-driven truncation replacing heuristic bottom-X% removal |
| `parallel_routes_blender` | PRM + ML Blender | PM | Multi-route blending with learned or fixed weights |

**Key production concepts modeled:**
- **HSNN**: Hierarchical Structured Neural Networks — clusters ads into hierarchy, scores cluster centroids first (sublinear cost), only expands promising clusters
- **PRM**: Parallel Retrieval Models — multiple parallel routes with different optimization objectives, replacing heuristic sources like ForceRetrieval and related ads
- **ML Blender**: Learned integration layer that determines optimal candidate mix from multiple routes
- **ML Reducer**: ML-driven filtering that predicts which candidates are unlikely to survive downstream stages

### Exploration Track

Experimental retrieval strategies not yet directly modeled in production. These serve as research tools and baselines for validating production investments.

| Tool | What It Explores | Research Value |
|------|------------------|----------------|
| `anti_negative_scorer` | Directional scoring: push toward positive centroid, away from negative | Tests whether explicit negative signal improves recall beyond simple cosine |
| `cluster_explorer` | Flat K-means clustering with per-cluster engagement rates | Baseline for HSNN comparison — measures value of flat vs hierarchical clustering |
| `similar_ads_lookup` | Ad-to-ad cosine expansion from known good ads | Tests "more-like-this" expansion; candidate for PRM route design |
| `mmr_reranker` | Maximal Marginal Relevance diversity re-ranking | Quantifies recall/diversity tradeoff; informs diversity mechanism design |
| `feature_filter` | Embedding-derived feature filtering (norms, means) | Tests whether embedding features carry signal beyond cosine similarity |

### Diagnostics Track

Analysis tools that inform strategy selection but don't retrieve candidates.

| Tool | Purpose |
|------|---------|
| `engagement_pattern_analyzer` | Assess signal quality: similarity gap, overlap fraction, cluster engagement rates |
| `ads_pool_stats` | Pool-level statistics: similarity distribution, cluster sizes |
| `lookup_similar_requests` | Historical learning from past requests with similar signal characteristics |

## Exploration → Production Graduation Path

Exploration tools can validate production investments:

| If exploration tool... | Then production implication... |
|---|---|
| `similar_ads_lookup` consistently finds unique positives that survive AF | Validates Related Ads / UNAGI A2A as a PRM route |
| `cluster_explorer` results largely overlap `hsnn_cluster_scorer` | Confirms HSNN subsumes flat clustering — no need for separate flat route |
| `anti_negative_scorer` adds recall beyond production routes | Directional scoring should be considered as a PRM route |
| `mmr_reranker` improves NDCG without hurting recall | Diversity mechanism should be integrated into ML Blender |
| `feature_filter` identifies features that predict stage survival | Features should be added to ML Reducer's scoring model |

## Data Available

For each ad request:
- User embedding (32d PSelect/TTSN)
- All candidate ad embeddings (32d)
- Engagement labels (positive = user engaged, negative = user didn't engage)
- Train/test label split (history_labels for tools, test_labels for evaluation)
- Production model predictions (calibrated CTR from SlimDSNN, when available)
- Embedding-derived features (cosine score, norms, cluster assignments)

## Key Metrics

### Per-Stage Metrics
- **Positive survival rate**: fraction of engaged ads surviving each stage
- **Value preservation**: fraction of total score value kept after truncation
- **Stage rank correlation**: Spearman correlation between adjacent stages' rankings

### Cross-Stage Metrics
- **Total positive survival**: fraction of all positives reaching AF
- **Cross-stage consistency**: rank correlation between PM and AI, AI and AF
- **Drop-off bottleneck**: which stage loses the most positives

### Retrieval Quality
- **Recall@K**: fraction of positives in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Route contribution**: unique positives per retrieval route
