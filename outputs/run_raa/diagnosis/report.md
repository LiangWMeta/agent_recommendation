# Pilot Diagnosis Report

Generated: 2026-04-01 18:38:14
Requests analyzed: 100
Data source: data/local/model/split

## Q1: Stage Drop-Off Analysis

| Stage | Avg Positive Survival Rate | Std |
|-------|---------------------------|-----|
| AP | 1.0000 | 0.0000 |
| PM | 0.3612 | 0.1473 |
| AI | 0.0819 | 0.0454 |
| AF | 0.0227 | 0.0196 |

**Key finding**: On average, 2.3% of positive ads survive the full AP->AF pipeline.

## Q2: Route Uniqueness Analysis

| Route | Avg Candidates | Avg Unique | Avg Unique Positives | Avg Overlap |
|-------|---------------|------------|---------------------|-------------|
| forced_retrieval | 64.0 | 56.8 | 3.98 | 0.1209 |
| prod_model | 100.0 | 90.5 | 6.61 | 0.0946 |
| pselect | 100.0 | 85.6 | 9.51 | 0.1436 |

**Key finding**: Route `prod_model` contributes the most unique candidates (90.5 avg).

## Q3: ML Reducer vs Heuristic Truncation

| Method | Avg Value Preserved | Avg Positive Preservation |
|--------|--------------------|--------------------------:|
| ml_value | 0.0407 | 0.7377 |
| heuristic_cosine | 0.0000 | 0.6815 |

**Key finding**: `ml_value` preserves 5.6% more positives at 50% reduction rate.

## Q4: HSNN Exploration Budget

| expand_top_k | Avg Recall@100 | Avg Compute Savings |
|-------------|---------------|--------------------:|
| 2 | 0.1120 | 0.8112 |
| 3 | 0.1141 | 0.7076 |
| 5 | 0.1220 | 0.5019 |
| 8 | 0.1269 | 0.1976 |

**Key finding**: Expanding from 2 to 8 coarse clusters improves recall by +0.0149 but reduces compute savings from 81.1% to 19.8%.

## Q5: Production vs Exploration Route Value

| Configuration | Avg Recall@100 | Delta vs Prod-Only |
|--------------|---------------|--------------------|
| prod_only | 0.1248 | +0.0000 |
| prod + hsnn_cluster | 0.1289 | +0.0042 |
| prod + anti_negative | 0.1429 | +0.0182 |
| prod + cluster_explorer | 0.1415 | +0.0168 |
| prod + similar_ads | 0.1565 | +0.0318 |

**Key finding**: Adding all exploration routes changes recall@100 by +0.0318 compared to production-only blend.

## Actionable Recommendations

- **Pipeline drop-off is severe** (2.3% positive survival). Consider widening PM/AI budgets or adding recall-optimized routes before PM.
- **ML-based truncation outperforms heuristic** (73.8% vs 68.1% positive preservation). Prioritize ML Reducer deployment.
- **HSNN expand_top_k=3 is sufficient**: expanding to 5 gains <1% recall while reducing compute savings from 70.8%.
- **Exploration routes add value** (+0.0318 recall). Include anti_negative, cluster_explorer, and similar_ads in production blend.
