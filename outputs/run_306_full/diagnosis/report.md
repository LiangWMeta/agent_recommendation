# Pilot Diagnosis Report

Generated: 2026-04-01 22:24:56
Requests analyzed: 306
Data source: data/local/model/split

## Q1: Stage Drop-Off Analysis

| Stage | Avg Positive Survival Rate | Std |
|-------|---------------------------|-----|
| AP | 1.0000 | 0.0000 |
| PM | 0.3591 | 0.1502 |
| AI | 0.0843 | 0.0500 |
| AF | 0.0223 | 0.0203 |

**Key finding**: On average, 2.2% of positive ads survive the full AP->AF pipeline.

## Q2: Route Uniqueness Analysis

| Route | Avg Candidates | Avg Unique | Avg Unique Positives | Avg Overlap |
|-------|---------------|------------|---------------------|-------------|
| forced_retrieval | 64.3 | 57.5 | 4.21 | 0.1108 |
| prod_model | 100.0 | 90.8 | 6.32 | 0.0925 |
| pselect | 100.0 | 86.2 | 9.53 | 0.1382 |

**Key finding**: Route `prod_model` contributes the most unique candidates (90.8 avg).

## Q3: ML Reducer vs Heuristic Truncation

| Method | Avg Value Preserved | Avg Positive Preservation |
|--------|--------------------|--------------------------:|
| ml_value | 0.0133 | 0.7328 |
| heuristic_cosine | 0.0000 | 0.6851 |

**Key finding**: `ml_value` preserves 4.8% more positives at 50% reduction rate.

## Q4: HSNN Exploration Budget

| expand_top_k | Avg Recall@100 | Avg Compute Savings |
|-------------|---------------|--------------------:|
| 2 | 0.1155 | 0.8100 |
| 3 | 0.1183 | 0.7077 |
| 5 | 0.1234 | 0.5001 |
| 8 | 0.1286 | 0.1950 |

**Key finding**: Expanding from 2 to 8 coarse clusters improves recall by +0.0131 but reduces compute savings from 81.0% to 19.5%.

## Q5: Production vs Exploration Route Value

| Configuration | Avg Recall@100 | Delta vs Prod-Only |
|--------------|---------------|--------------------|
| prod_only | 0.1220 | +0.0000 |
| prod + hsnn_cluster | 0.1289 | +0.0069 |
| prod + anti_negative | 0.1451 | +0.0232 |
| prod + cluster_explorer | 0.1453 | +0.0233 |
| prod + similar_ads | 0.1604 | +0.0384 |

**Key finding**: Adding all exploration routes changes recall@100 by +0.0384 compared to production-only blend.

## Actionable Recommendations

- **Pipeline drop-off is severe** (2.2% positive survival). Consider widening PM/AI budgets or adding recall-optimized routes before PM.
- **ML-based truncation outperforms heuristic** (73.3% vs 68.5% positive preservation). Prioritize ML Reducer deployment.
- **HSNN expand_top_k=3 is sufficient**: expanding to 5 gains <1% recall while reducing compute savings from 70.8%.
- **Exploration routes add value** (+0.0384 recall). Include anti_negative, cluster_explorer, and similar_ads in production blend.
