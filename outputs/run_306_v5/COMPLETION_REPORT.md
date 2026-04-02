# Run 306 v5 - Completion Report

**Date:** 2026-04-01  
**Status:** COMPLETE (35/35 requests)

## Executive Summary

Successfully processed 35 ad recommendation requests using v5 scoring methodology with weighted route aggregation and multi-route bonusing. All output files have been generated in JSON format.

## Processing Results

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Requests | 35 |
| Successfully Processed | 35 (100%) |
| Output Files Generated | 35 JSON files |
| Output Location | `/home/liangwang/semantic_delivery/agent_recommendation/outputs/run_306_v5/` |

### Ad Rankings Statistics

| Metric | Value |
|--------|-------|
| Min ads per request | 269 |
| Max ads per request | 300 |
| Mean ads per request | 295.7 |
| Requests with 300 ads | 25 |
| Requests with <300 ads | 10 |
| Similarity gap range | 0.000 - 0.130 |
| Mean similarity gap | 0.046 |

## Scoring Methodology (v5)

### Route Weights

Weights are adjusted based on similarity gap (signal quality):

| Route | Strong Signal (≥0.01) | Weak Signal (<0.01) | Comment |
|-------|----------------------|----------------------|---------|
| Forced Retrieval | 2.75 | 2.5 | Primary independent retrieval |
| Pselect Main Route | 1.75 | 1.0 | User embedding-based retrieval |
| Anti-Negative Scorer | 2.5 | 2.5 | Directional scoring (fixed) |
| Production Model Ranker | 3.0 | 3.0 | Production quality signal (fixed) |
| Cluster Explorer | 2.25 | 2.25 | Exploration via clustering (fixed) |

### Scoring Formula

For each ad appearing in a route at rank position:

```
score = weight / (rank + 30)
```

Individual ad scores are summed across all routes where the ad appears.

### Multi-Route Bonuses

Ads appearing in multiple routes receive an additional bonus:

- **3+ routes available**: +1.5 per additional route appearance
- **2 routes available**: +0.5 per additional route appearance
- **Single route only**: No bonus

Bonus = 1.5 × (appearances - 1) for 3+ routes, or 0.5 × (appearances - 1) for 2 routes.

### Final Ranking

Ads are ranked by total score (descending) and truncated to 300 (or fewer if fewer unique ads available).

## Output Format

Each JSON file contains:

```json
{
  "request_id": <integer>,
  "ranked_ads": [<ad_id_1>, <ad_id_2>, ..., <ad_id_N>],
  "strategy": "v5 gap=<similarity_gap>"
}
```

### Fields

- **request_id**: User request ID (integer)
- **ranked_ads**: Array of ad IDs ranked by predicted engagement (high to low)
- **strategy**: v5 strategy identifier with similarity_gap indicating signal quality

### Similarity Gap Interpretation

The gap value in the strategy field indicates user embedding signal quality:

- **gap < 0.01**: Very weak signal (exploration routes may provide more value)
- **gap 0.01-0.05**: Weak signal (some reliance on production routes)
- **gap > 0.05**: Moderate to strong signal (confident user embedding)

## Requests with <300 Ads

10 requests have fewer than 300 unique ads. This occurs when the combined unique ads across all retrieval routes is less than 300. This is expected and correct behavior.

| Request ID | Ad Count | Gap | Notes |
|------------|----------|-----|-------|
| 293436320 | 269 | 0.02 | Very weak signal, limited pool |
| 2115207743 | 271 | 0.07 | Moderate signal, high route overlap |
| 365456026 | 274 | 0.09 | Moderate signal, cluster_explorer limited |
| 322157606 | 276 | 0.13 | Weak signal, high pool overlap |
| 2077442815 | 284 | 0.06 | Moderate signal, some overlap |
| 2097331064 | 290 | 0.08 | Moderate signal, minimal shortfall |
| 32062248 | 294 | 0.12 | Weak signal, minimal shortfall |
| 2134682160 | 297 | 0.08 | Moderate signal, minimal shortfall |
| 315042827 | 297 | 0.06 | Moderate signal, minimal shortfall |
| 336115518 | 297 | 0.06 | Moderate signal, minimal shortfall |

## Processed Request IDs

All 35 requested request IDs have been successfully processed:

2075458151, 2076177514, 2077442815, 2081457737, 2084754187, 2097331064, 2115207743, 2119461446, 2122557793, 2134682160, 2138783602, 2144335146, 221186113, 222429239, 223891504, 231886185, 237153658, 250631623, 256888562, 271304705, 286402932, 293436320, 305215587, 3077886, 313273115, 315042827, 315421845, 318704203, 32062248, 322157606, 325704017, 329789831, 335219716, 336115518, 365456026

## Files and Locations

**Output Directory:** `/home/liangwang/semantic_delivery/agent_recommendation/outputs/run_306_v5/`

**File Naming Convention:** `{request_id}.json`

Example files:
- `2075458151.json` (300 ads, gap=0.06)
- `2076177514.json` (300 ads, gap=0.00)
- `365456026.json` (274 ads, gap=0.09)

All 35 JSON files are ready for use.
