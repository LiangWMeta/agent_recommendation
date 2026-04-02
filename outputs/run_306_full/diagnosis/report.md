# Pilot Diagnosis Report

Generated: 2026-04-01 20:10:55
Requests analyzed: 306
Data source: data/local/model/split

## Q1: Stage Drop-Off Analysis

| Stage | Avg Positive Survival Rate | Std |
|-------|---------------------------|-----|
| AP | 1.0000 | 0.0000 |
| PM | 0.4145 | 0.1762 |
| AI | 0.1055 | 0.0706 |
| AF | 0.0271 | 0.0240 |

**Key finding**: On average, 2.7% of positive ads survive the full AP->AF pipeline.

## Q2: Route Uniqueness Analysis

| Route | Avg Candidates | Avg Unique | Avg Unique Positives | Avg Overlap |
|-------|---------------|------------|---------------------|-------------|
| forced_retrieval | 82.0 | 75.8 | 9.19 | 0.0857 |
| prod_model | 100.0 | 90.0 | 6.53 | 0.0996 |
| pselect | 100.0 | 90.0 | 9.91 | 0.0998 |

**Key finding**: Route `prod_model` contributes the most unique candidates (90.0 avg).

## Q3: ML Reducer vs Heuristic Truncation

| Method | Avg Value Preserved | Avg Positive Preservation |
|--------|--------------------|--------------------------:|
| ml_value | 0.0133 | 0.7553 |
| heuristic_cosine | 0.0000 | 0.6851 |

**Key finding**: `ml_value` preserves 7.0% more positives at 50% reduction rate.

## Q4: HSNN Exploration Budget

| expand_top_k | Avg Recall@100 | Avg Compute Savings |
|-------------|---------------|--------------------:|
| 2 | 0.1160 | 0.8099 |
| 3 | 0.1183 | 0.7077 |
| 5 | 0.1234 | 0.5001 |
| 8 | 0.1286 | 0.1950 |

**Key finding**: Expanding from 2 to 8 coarse clusters improves recall by +0.0126 but reduces compute savings from 81.0% to 19.5%.

## Q5: Production vs Exploration Route Value

| Configuration | Avg Recall@100 | Delta vs Prod-Only |
|--------------|---------------|--------------------|
| prod_only | 0.1395 | +0.0000 |
| prod + hsnn_cluster | 0.1374 | -0.0022 |
| prod + anti_negative | 0.1537 | +0.0142 |
| prod + cluster_explorer | 0.1524 | +0.0129 |
| prod + similar_ads | 0.1634 | +0.0239 |

**Key finding**: Adding all exploration routes changes recall@100 by +0.0239 compared to production-only blend.

## Actionable Recommendations

- **Pipeline drop-off is severe** (2.7% positive survival). Consider widening PM/AI budgets or adding recall-optimized routes before PM.
- **ML-based truncation outperforms heuristic** (75.5% vs 68.5% positive preservation). Prioritize ML Reducer deployment.
- **HSNN expand_top_k=3 is sufficient**: expanding to 5 gains <1% recall while reducing compute savings from 70.8%.
- **Exploration routes add value** (+0.0239 recall). Include anti_negative, cluster_explorer, and similar_ads in production blend.
