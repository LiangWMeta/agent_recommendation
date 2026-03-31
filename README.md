# Agent Recommendation Framework

LLM-orchestrated multi-route ads retrieval system that uses Claude Code as a hybrid orchestrator — calling retrieval tools, reasoning over aggregated results, and producing ranked ad lists evaluated against production recall metrics.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Claude Code (Orchestrator)                           │
│  - Reads: ads pool profile, user engagement, route    │
│    diagnostics, cross-route patterns, learnings       │
│  - Reasons: signal quality, route conflicts, cluster  │
│    blind spots, consensus candidates                  │
│  - Decides: strategy, route weights, ranking          │
└────────┬──────────────────────────────┬───────────────┘
         │ Tool results (aggregated)    │
   ┌─────▼─────┐  ┌─────────┐  ┌──────▼─────┐  ┌──────────┐
   │ Embedding  │  │ FR      │  │ Cluster    │  │ Prod     │
   │ Similarity │  │ Centroid│  │ Explorer   │  │ Model    │
   │ Search     │  │ Search  │  │            │  │ Ranker   │
   └────────────┘  └─────────┘  └────────────┘  └──────────┘
   + anti_negative_scorer, similar_ads, mmr_reranker,
     feature_filter, engagement_analyzer, pool_stats,
     history_lookup (11 tools total)
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  Production Recall Evaluation (FBLearner Flow)        │
│  calculate_hard_recall / calculate_soft_recall         │
│  PM (190K→11K) → AI (11K→709) → AF (709→84)          │
│  Per-stage non-leaking signal evaluation              │
└──────────────────────────────────────────────────────┘
```

## Tools (11 total, mapped to production routes)

| Tool | Production Analog | Purpose |
|------|-------------------|---------|
| `embedding_similarity_search` | Main Route (TTSN PM) | Cosine search by user embedding |
| `fr_centroid_search` | Forced Retrieval (85% of impressions) | Search by centroid of engaged ads |
| `anti_negative_scorer` | Directional scoring | Push toward engaged, away from ignored |
| `cluster_explorer` | HSNN-AP CER (10K clusters) | Cluster-based retrieval with engagement rates |
| `similar_ads_lookup` | Related Ads / UNAGI A2A | Find ads similar to reference ads |
| `prod_model_ranker` | SlimDSNN PM scoring | Rank by production calibrated CTR |
| `mmr_reranker` | Diversity mechanisms | MMR diversity re-ranking |
| `engagement_pattern_analyzer` | — | Signal diagnostics (gap, overlap, clusters) |
| `ads_pool_stats` | — | Candidate pool statistics |
| `feature_filter` | — | Filter by embedding features |
| `lookup_similar_requests` | — | Historical learning from past runs |

## Results

### Local Recall (100 requests, 2K candidates, clean RAA labels, train/test split)

| Metric | Dot-Product | Weighted RRF | Improvement |
|--------|------------|-------------|-------------|
| Recall@50 | 7.1% | **8.4%** | **+18%** |
| Recall@100 | 12.7% | **15.1%** | **+19%** |
| Recall@200 | 21.9% | **27.1%** | **+24%** |

### Production Recall (50 requests, 190K candidates, FBLearner flow, non-leaking)

| Metric | Baseline (PSelect) | RRF (non-leaking) | Improvement |
|--------|-------------------|-------------------|-------------|
| AI Soft Recall @20% | 70.5% | **71.7%** | **+1.8%** |
| AI Soft Recall @50% | 69.8% | **72.0%** | **+3.1%** |
| AF Soft Recall @15% | 51.9% | **52.8%** | **+1.6%** |
| AF Soft Recall @20% | 52.0% | **52.7%** | **+1.4%** |

### Data Leakage Analysis

A critical finding: several tool signals (FR centroid, anti-negative, cluster engagement) use engagement labels that are also the evaluation target. Without proper controls, this inflates results by 30-50%.

**Mitigations implemented:**
- Train/test split on labels (50/50) — tools see history, evaluation uses held-out test set
- Per-stage signal constraints — each stage only uses prior-stage orderings
- PM recall uses PSelect only (no prior stage available)
- AI recall uses PSelect + PM signals (PM is prior stage)
- AF recall uses PSelect + PM + AI signals (all prior stages)

## Benchmark Approaches

| Approach | Speed | Coverage | Use Case |
|----------|-------|----------|----------|
| `run_benchmark_cc.py` | ~3 min/req | Claude reasons per-request | Interactive tool-call debugging |
| `run_benchmark_fast.py` | ~3 min/req | Pre-computed tools, 1 Claude call | Single-request evaluation |
| `run_benchmark_batch.py` | ~80s/req | 5 requests per Claude call | Batch evaluation (100+ requests) |
| `run_baseline_weighted.py` | ~0.3s/req | Fixed-weight RRF, no LLM | Baseline comparison |
| FBLearner recall workflow | ~15 min/50 req | Full 190K pool, production metrics | Production-aligned evaluation |

## Quick Start

```bash
# 1. Create train/test split data (no leakage)
python3 scripts/create_split_data.py --data-dir data --max-requests 100

# 2. Generate user context files
python3 scripts/prepare_contexts.py --data-dir data_split --output-dir requests_split --max-requests 100

# 3. Run fixed-weight baseline (instant, no LLM)
python3 scripts/run_baseline_weighted.py --run-id baseline --data-dir data_split --max-requests 100

# 4. Run agent benchmark (batch of 5 per Claude call)
python3 scripts/run_benchmark_batch.py --run-id agent_run --data-dir data_split --batch-size 5 --max-requests 20

# 5. Evaluate (against test_labels only)
python3 evaluation/evaluate.py --run-id agent_run --baseline evaluation/results/baseline.json

# 6. Production recall (FBLearner flow, requires entitlement)
cd /data/users/$USER/fbsource
flow-cli canary-locally \
  --run-as-secure-group oncall_ads_ranking_multi_stage \
  --entitlement ads_global_short_term_exploration \
  scripts.liangwang.agent_recall_workflow.agent_recall_workflow \
  --buck-target fbcode//scripts/liangwang:workflow \
  --parameters-file /tmp/recall_params.json
```

## Directory Structure

```
agent_recommendation/
├── CLAUDE.md                     # Orchestration instructions for Claude
├── architecture.md               # Ads rec system context
├── skill.md                      # Adaptive strategy framework
├── learnings.md                  # Empirical findings from past runs
│
├── tools/                        # 11 MCP retrieval tools
│   ├── mcp_server.py             # MCP stdio server for claude -p
│   ├── tool_registry.py          # Tool schemas + dispatch
│   ├── embedding_search.py       # Cosine similarity search
│   ├── fr_centroid_search.py     # FR centroid retrieval
│   ├── anti_negative_scorer.py   # Directional scoring
│   ├── cluster_explorer.py       # K-means cluster exploration
│   ├── similar_ads.py            # Ad-to-ad similarity
│   ├── prod_model_ranker.py      # Production model scores
│   ├── mmr_reranker.py           # MMR diversity re-ranking
│   ├── engagement_analyzer.py    # Signal diagnostics
│   ├── pool_stats.py             # Pool statistics
│   ├── feature_filter.py         # Feature filtering
│   └── history_lookup.py         # Historical learning
│
├── evaluation/
│   ├── baseline.py               # Dot-product baseline (recall@100 = 12.7%)
│   ├── evaluate.py               # Recall/precision/NDCG evaluator
│   ├── compare_runs.py           # Run comparison
│   └── prod_recall.py            # Production recall utilities
│
├── scripts/
│   ├── create_split_data.py      # Train/test split (no leakage)
│   ├── prepare_contexts.py       # Generate per-user context folders
│   ├── run_benchmark_cc.py       # Claude Code MCP benchmark
│   ├── run_benchmark_fast.py     # Pre-computed tools benchmark
│   ├── run_benchmark_batch.py    # Batch benchmark (5 per call)
│   ├── run_baseline_weighted.py  # Fixed-weight RRF baseline
│   ├── run_recall_pipeline.py    # Production recall SQL
│   ├── submit_recall_flow.py     # FBLearner flow submission
│   ├── upload_agent_scores.py    # Upload scores to Hive
│   ├── update_history.py         # Accumulate results to history
│   ├── extract_prod_predictions.py  # Extract prod_prediction
│   └── extract_bulk_eval_candidates.py  # Extract from bulk eval
│
├── data/ → ../mvp/data/real_data_light/  # 306 requests (symlink)
├── data_split/                   # Train/test split data
├── data_enriched/                # Prod predictions per request
├── data_bulk_eval/               # Extracted from bulk eval (2K/request)
├── data_full_pool/               # Full 190K pool (in progress)
├── requests/                     # Per-user context folders
└── outputs/                      # Benchmark outputs per run

FBLearner workflows (in fbsource):
├── fbcode/scripts/liangwang/
│   ├── agent_recall_workflow.py      # Production recall with per-stage RRF
│   ├── extract_candidates_workflow.py # Extract candidates from bulk eval
│   └── blended_recall_workflow.py    # Rank divergence recall baseline
```

## Data Sources

| Dataset | Requests | Candidates/Req | Labels | Use |
|---------|---------|----------------|--------|-----|
| `data/` (real_data_light) | 306 | ~2,000 | RAA engagement | Local evaluation |
| `data_split/` | 100 | ~2,000 | Train/test split | Leakage-free eval |
| `data_bulk_eval/` | 20 | ~2,000 | AI top-709 | Production-scale pilot |
| `data_full_pool/` | 3 (WIP) | ~190,000 | Train/test split | Full coverage eval |
| Bulk eval aggregate | 50+ | ~190,000 | PM/AI/AF orderings | FBLearner recall |

## Production Recall Pipeline

Uses `calculate_hard_recall` / `calculate_soft_recall` from production `recall_util`:

```
All 190K candidates ordered by prediction
→ Truncate bottom X% (0%, 5%, 10%, ..., 50%)
→ PM stage: |remaining ∩ PM_top_11119| / 11119
→ AI stage: |PM_survivors ∩ AI_top_709| / 709
→ AF stage: |AI_survivors ∩ AF_top_84| / 84
```

Per-stage non-leaking RRF: each stage's prediction ordering only uses signals from prior stages.

## Key Learnings

1. **Data leakage is pervasive** — FR centroid, anti-negative, cluster engagement all use labels that are the evaluation target. Train/test split is essential.
2. **Simple beats complex** — 3-signal RRF outperforms 9-signal. More signals add noise.
3. **Scale matters** — Agent's 200 ads are invisible in 190K pool. Full-pool RRF shows real impact.
4. **Soft recall improves, hard recall doesn't** — RRF surfaces higher-VALUE ads but not more ads by count.
5. **Per-stage signals matter** — Using AF ordering to predict AF recall is leakage; only prior-stage signals are valid.
6. **AI signals predict AF survival** — PM+AI RRF improves AF soft recall by +1.6% at 15% truncation.

## FBLearner Flow Runs

| Run ID | Type | Requests | Key Result |
|--------|------|----------|------------|
| f1057460760 | Blended recall baseline | 50 | Production recall validated |
| f1058397243 | Agent recall (first test) | 20 | Pipeline validated, delta~0 (scale) |
| f1058774858 | Non-leaking per-stage RRF | 50 | AF soft +0.85pp @15%, AI soft +2.1pp @50% |
| f1058778823 | AI-dominant AF weights | 50 | AF soft +0.63pp @20% |
| f1058420052 | Candidate extraction | 20 | 2K candidates with embeddings |
