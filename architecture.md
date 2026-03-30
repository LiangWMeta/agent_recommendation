# Ads Recommendation System Architecture

## Overview

The ads recommendation system connects users with relevant ads through a multi-stage pipeline: retrieval → ranking → auction → delivery. This framework focuses on the **retrieval stage** — selecting the initial set of candidate ads from a large pool.

## Current Retrieval System

The production retrieval system uses **dot-product (cosine) similarity** between:
- **User tower embedding** (32d): Encodes user interests, demographics, and engagement history via a Two-Tower Scoring Network (TTSN)
- **Ad tower embedding** (32d): Encodes ad content, targeting, and historical performance

Retrieval selects the top-K ads by cosine(user_emb, ad_emb). This is fast and effective for warm users with clear behavioral patterns, but has known gaps:

### Retrieval Gaps
1. **Cold-start ads**: New ads with few impressions have poorly calibrated embeddings
2. **Interest diversity**: Users have multiple interest clusters but retrieval over-indexes on the dominant one
3. **Semantic mismatch**: Behavioral embeddings capture click patterns, not semantic meaning — two ads about the same product with different creatives can have distant embeddings
4. **Exploration deficit**: Without diversity mechanisms, retrieval converges to a narrow set of "safe" candidates

## Three Semantic Domains

### User Semantics
- **Intent**: What the user wants now (volatile, session-dependent)
- **Interests**: Deep, stable preferences across categories
- **Cross-surface context**: Behavior across FB, IG, Messenger synthesized into coherent representation

### Ad Semantics
- **Creative meaning**: What the ad communicates visually and textually
- **Product attributes**: Category, price range, brand, use case
- **Advertiser objectives**: Awareness vs. consideration vs. conversion

### Interaction Semantics
- **Engagement patterns**: Why users engage, not just that they engage
- **Session trajectory**: How previous ad exposures influence subsequent engagement
- **Fatigue and novelty**: Diminishing returns vs. value of new discoveries

## The Opportunity

An LLM orchestrator can improve retrieval by:
1. **Multi-route retrieval**: Combining embedding similarity with cluster-based exploration and feature-based filtering
2. **Adaptive strategy**: Adjusting the retrieval mix based on user patterns (explore more for diverse users, exploit more for focused users)
3. **Gap detection**: Identifying candidate types that pure dot-product retrieval systematically misses

## Data Available

For each ad request, you have:
- User embedding (32d PSelect/TTSN)
- All candidate ad embeddings (32d)
- Engagement labels (positive = user engaged, negative = user didn't engage)
- Embedding-derived features (cosine score, norms, cluster assignments)
