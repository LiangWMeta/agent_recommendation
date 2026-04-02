---
name: AF Feature Analysis for PM Scoring Improvement
description: Deep analysis of AF model input features, their availability at earlier pipeline stages, and which features best predict engagement — used to guide PM scorer and retrieval improvements
type: project
---

## AF Model Feature Analysis (2026-04-02)

### Key Finding: CVR and CTR at PM Stage Are the Strongest Engagement Predictors

Feature separation analysis (positive vs negative ads):

| Feature | Positive | Negative | Separation | Stage |
|---------|---------|---------|-----------|-------|
| ai_ecvr (AI CVR) | 0.044 | 0.013 | **3.37x** | AI |
| pm_ectr (PM CTR) | -0.242 | -0.568 | **2.35x** | PM |
| pm_ecvr (PM CVR) | -0.206 | -0.464 | **2.25x** | PM |
| pm_adv_val | 0.176 | 0.083 | **2.12x** | PM |
| pm_rda | 5.28 | 2.87 | **1.84x** | PM |
| ai_ectr | 0.0038 | 0.0021 | 1.81x | AI |
| eco_bid | 0.0017 | 0.0011 | 1.59x | PM |
| pm_pacing | 0.277 | 0.512 | 0.54x (inverse) | PM |
| ai_quality | -0.532 | -0.182 | 2.93x (anti) | AI |

**Why:** pm_ecvr and pm_ectr are available at PM stage but our retrieval ignores them. Using them in a new retrieval route could find the 63% of positives that embedding-only retrieval misses.

**How to apply:** Build PM-feature-based retrieval route using pm_ectr + pm_ecvr + pm_adv_val from `fct_raa_consideration_data.pm_stage_struct`.

### AF Model Architecture
- MTML SparseNN (300-500 features, 65ms budget)
- 4 tasks: CTR, iCVR, Quality, Teacher distillation
- Feature pipeline: AP (ad features) → AF 1st (user features) → AI (cross features) → AF 2nd (full scoring)

### Key Feature Categories
1. **User-side**: embeddings (SUM, FIRST), demographics, engagement history
2. **Ad-side**: campaign metadata, bid info, creative features (AF-only)
3. **Cross features**: user×ad interaction (SOURCE_USER_AD), id-matching
4. **Contextual**: page_type, device, time

### PM vs AF Feature Gap
- PM (SlimDSNN): precomputed embeddings + I2 features only
- AF (MTML SparseNN): full user-ad interactions + creative + quality features
- The gap = user-ad cross features + creative features + quality adjustments

### Data Sources
- `fct_raa_consideration_data` — pm_stage_struct, ai_stage_struct, af_stage_struct with per-component breakdowns
- `adcandidate_features_unpacked_ecpm_bid` — 80 unpacked features including prematch_*, posting_list_*, conversion_lift_*
- float_features index: 1308=af_ectr, 1312=af_ecvr, 3281=ai_total_value, 3398=af_total_value, 5166=pm_total_value
