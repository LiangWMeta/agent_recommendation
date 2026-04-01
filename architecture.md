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
