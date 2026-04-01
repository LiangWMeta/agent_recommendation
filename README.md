# Agent Recommendation Framework

LLM-orchestrated multi-route ads retrieval system that uses Claude Code as a hybrid orchestrator — calling retrieval tools, reasoning over aggregated results, and producing ranked ad lists evaluated against production recall metrics.

Models the production **cascaded pipeline (AP → PM → AI → AF)** to enable e2e reasoning about stage-by-stage filtering, cross-stage consistency, and retrieval route contributions.

## Claude Code Skills

The system is operated through 4 slash commands within Claude Code:

| Command | Purpose |
|---------|---------|
| `/recommend` | Run agent recommendation on a batch of requests via MCP tools |
| `/analyze` | Run diagnosis on requests, reason about results, suggest improvements |
| `/modify` | Modify the system via natural language prompt (add route, add data, change algo) |
| `/data-setup` | Pull and prepare all data needed to run the agent |

**Typical workflow:**
```
/data-setup                              # Prepare data (first time only)
/recommend --requests 20                 # Run agent on 20 requests
/analyze --requests 20                   # Analyze system behavior
/modify "add a route that scores by ad recency"  # Evolve the system
/analyze --requests 20                   # Verify improvement
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  CONTEXT LAYER (read before tools)                                   │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐ │
│  │ Ads Pool Understanding   │  │ User Context Folder              │ │
│  │ ads_pool/                │  │ user/{request_id}/               │ │
│  │ - embeddings.npz         │  │ - profile.md (demographics+emb) │ │
│  │ - catalog.md (all ads)   │  │ - intent.md (session intent)    │ │
│  │ - pool_overview.md       │  │ - interests.md (stable prefs)   │ │
│  │ - semantic_clusters.md   │  │ - session_history.md (fatigue)  │ │
│  │ - pool_changes.md        │  │ - engagement.md (enriched)      │ │
│  │ (refreshed regularly)    │  │ - context.md (request meta)     │ │
│  └──────────────────────────┘  └──────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Claude Code (Orchestrator)                                          │
│  - Reads ads pool + user context to understand landscape & user      │
│  - Assesses signal quality (similarity_gap → strong/moderate/weak)   │
│  - Selects strategy: production-first, exploration for research      │
│  - Reasons about pipeline survival: AP → PM → AI → AF               │
└───────┬──────────────────────────────────────────────────┬───────────┘
        │ Tool results (aggregated)                       │
  ┌─────▼──────────────────────────────────────────────────▼──────┐
  │  PRODUCTION TRACK (simulate real pipeline components)         │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
  │  │Embedding │ │FR        │ │HSNN      │ │Prod Model        ││
  │  │Similarity│ │Centroid  │ │Cluster   │ │Ranker            ││
  │  │Search    │ │Search    │ │Scorer    │ │(SlimDSNN CTR)    ││
  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
  │  ┌──────────────────┐ ┌──────────┐ ┌──────────────────────┐ │
  │  │Parallel Routes   │ │ML        │ │Pipeline              │ │
  │  │Blender (PRM+ML)  │ │Reducer   │ │Simulator (AP→AF)     │ │
  │  └──────────────────┘ └──────────┘ └──────────────────────┘ │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │  EXPLORATION TRACK (research / hypothesis testing)           │
  │  anti_negative_scorer, cluster_explorer, similar_ads_lookup, │
  │  mmr_reranker, feature_filter                                │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │  DIAGNOSTICS TRACK                                           │
  │  engagement_pattern_analyzer, ads_pool_stats,                │
  │  lookup_similar_requests                                     │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Evaluation                                                      │
│  Standard: Recall/Precision/NDCG @K (evaluate.py)                │
│  Pipeline: Per-stage survival, truncation robustness             │
│            (evaluate_pipeline.py)                                │
│  Pilot:    5-question diagnosis report (run_pilot_diagnosis.py)  │
│  Prod:     FBLearner flow, 190K candidates, per-stage recall     │
└──────────────────────────────────────────────────────────────────┘
```

## Tools (15 total)

### Production Track (simulate production components)

| Tool | Production Analog | Stage | Purpose |
|------|-------------------|-------|---------|
| `embedding_similarity_search` | Main Route (TTSN PM / PSelect) | AP/PM | Cosine search by user embedding |
| `fr_centroid_search` | Forced Retrieval (85% of impressions) | AP | Search by centroid of engaged ads |
| `prod_model_ranker` | SlimDSNN PM scoring | PM | Rank by production calibrated CTR |
| `hsnn_cluster_scorer` | HSNN (88% AP, 36% PM adoption) | AP/PM | 2-level hierarchical cluster scoring |
| `pipeline_simulator` | Cascaded pipeline (AP→PM→AI→AF) | All | Full cascade with per-stage survival |
| `ml_reducer` | ML Reducer/Truncator | PM/AI | ML-driven truncation vs heuristic |
| `parallel_routes_blender` | PRM + ML Blender | PM | Multi-route blending (RRF/learned/priority) |

### Exploration Track (research / hypothesis testing)

| Tool | What It Explores | Research Value |
|------|------------------|----------------|
| `anti_negative_scorer` | Directional scoring toward/away from centroids | Tests explicit negative signal vs cosine |
| `cluster_explorer` | Flat K-means with engagement rates | Baseline for HSNN comparison |
| `similar_ads_lookup` | Ad-to-ad expansion from engaged ads | Candidate for PRM route design |
| `mmr_reranker` | Maximal Marginal Relevance diversity | Recall/diversity tradeoff |
| `feature_filter` | Embedding feature filtering | Feature hypothesis testing |

### Diagnostics Track

| Tool | Purpose |
|------|---------|
| `engagement_pattern_analyzer` | Signal quality: similarity gap, overlap, clusters |
| `ads_pool_stats` | Pool statistics and distribution |
| `lookup_similar_requests` | Historical learning from past runs |

## Pilot Diagnosis Results (10 requests)

The pilot (`scripts/run_pilot_diagnosis.py`) exercises all tools and answers 5 questions:

### Q1: Stage Drop-Off — Where do good ads die?

| Stage | Positive Survival Rate | Loss |
|-------|----------------------|------|
| AP | 100% | — |
| PM | 27.4% | 72.6% lost (biggest bottleneck) |
| AI | 3.5% | 87.2% of PM survivors lost |
| AF | 0.5% | Only 0.5% survive full cascade |

### Q2: Route Uniqueness — Which routes contribute unique value?

| Route | Unique Ads | Unique Positives | Overlap |
|-------|-----------|-----------------|---------|
| FR Centroid | 87.9 | **13.7** | 12.1% |
| Prod Model | 88.7 | 3.5 | 11.3% |
| Embedding | 28.3 | 3.6 | 71.7% |
| HSNN | 27.4 | 1.3 | 72.6% |

### Q3: ML Reducer vs Heuristic Truncation (50% reduction)

| Method | Positive Preservation | Advantage |
|--------|---------------------|-----------|
| ML Reducer (combined signals) | **75.9%** | **+6.8%** |
| Heuristic (cosine) | 69.1% | baseline |

### Q4: HSNN Exploration Budget

| expand_top_k | Recall@100 | Compute Savings |
|-------------|-----------|----------------|
| 2 | 15.3% | 80.4% |
| **3** | **14.0%** | **72.6% (sweet spot)** |
| 5 | 14.5% | 50.5% |
| 8 | 14.5% | 20.7% |

### Q5: Production vs Exploration Routes

| Configuration | Recall@100 | Delta |
|--------------|-----------|-------|
| Production routes only | **17.66%** | baseline |
| + anti_negative | 17.76% | +0.10% |
| + cluster_explorer | 17.16% | -0.50% |
| + similar_ads | 17.26% | -0.40% |

**Key insight**: Exploration routes add <0.1% recall — production routes dominate. Exploration value is in diversity, not recall.

## Earlier Results

### Local Recall (100 requests, 2K candidates, train/test split)

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

Several tool signals (FR centroid, anti-negative, cluster engagement) use engagement labels that are also the evaluation target. Without proper controls, this inflates results by 30-50%.

**Mitigations implemented:**
- Train/test split on labels (50/50) — tools see history, evaluation uses held-out test set
- Per-stage signal constraints — each stage only uses prior-stage orderings

## Quick Start

### Using Claude Code Skills (recommended)

```bash
/data-setup                        # Prepare all data (first time)
/recommend --requests 20           # Run agent on 20 requests
/analyze --requests 20             # Analyze system behavior + suggest improvements
/modify "add a new route"          # Modify the system
```

### Using Scripts Directly

See `data/datasets.md` for the full dataset registry, schemas, and loading patterns.

```bash
# 1. Create train/test split data (no leakage)
python3 scripts/create_split_data.py --data-dir data/local/model/raw --max-requests 100

# 2. Generate user context files
python3 scripts/prepare_contexts.py --data-dir data/local/model/split --output-dir user/

# 3. Run pilot diagnosis (exercises all pipeline tools, ~10s)
python3 scripts/run_pilot_diagnosis.py --max-requests 20

# 4. Run agent benchmark (batch of 5 per Claude call)
python3 scripts/run_benchmark_batch.py --run-id agent_run --data-dir data/local/model/split --batch-size 5 --max-requests 20

# 5. Evaluate (against test_labels only)
python3 evaluation/evaluate.py --run-id agent_run --baseline evaluation/results/baseline.json

# 6. Pipeline evaluation (per-stage survival metrics)
python3 evaluation/evaluate_pipeline.py --run-id agent_run
```

## Directory Structure

```
agent_recommendation/
├── CLAUDE.md                     # Agent instructions (context + tools + strategy)
├── architecture.md               # Pipeline architecture + context layer + tools
├── skill.md                      # Adaptive strategy + pipeline reasoning
├── learnings.md                  # Empirical findings from past runs
│
├── .claude/skills/               # Claude Code skills (slash commands)
│   ├── recommend/SKILL.md        # /recommend — run agent on requests
│   ├── analyze/SKILL.md          # /analyze — diagnosis + reasoning
│   ├── modify/SKILL.md           # /modify — system modification via prompt
│   └── data-setup/SKILL.md       # /data-setup — data preparation
│
├── ads_pool/                     # Ads Pool Understanding (refreshed regularly)
│   ├── embeddings.npz            # Ad embeddings (32d) + ad_ids + labels
│   ├── catalog.md                # All ads: category, objective, creative, targeting, CTR
│   ├── pool_overview.md          # Pool summary: size, categories, budget distribution
│   ├── semantic_clusters.md      # HSNN cluster → semantic label mapping
│   └── pool_changes.md           # Delta log: new/expired/changed since last refresh
│
├── user/                         # User Context Folders (per request, rich)
│   └── {request_id}/
│       ├── profile.md            # Demographics, device, embedding summary
│       ├── intent.md             # Session intent, inferred category
│       ├── interests.md          # Stable interest clusters, top categories
│       ├── session_history.md    # Ads seen, click/skip sequence, fatigue
│       ├── engagement.md         # Enriched engagement history by category
│       └── context.md            # Pool size, signal quality, placement
│
├── data/
│   ├── datasets.md               # Dataset registry: schemas, quickstart, lineage
│   └── local/
│       ├── model/                 # Model inputs (embeddings, predictions)
│       │   ├── raw/              # 306 requests, raw embeddings + labels
│       │   ├── split/            # 100 requests, history + test (PRIMARY)
│       │   ├── enriched/         # Prod prediction sidecars (SlimDSNN CTR)
│       │   └── full_pool/        # 190K candidate pool (WIP)
│       └── eval/                  # Evaluation inputs
│           └── bulk_eval/        # 20 prod pipeline extracts
│
├── tools/                        # 15 MCP retrieval tools
│   ├── mcp_server.py             # MCP stdio server
│   ├── tool_registry.py          # Tool schemas + dispatch
│   │  Production: embedding_search, fr_centroid_search, prod_model_ranker,
│   │    hsnn_cluster_scorer, pipeline_simulator, ml_reducer, parallel_routes_blender
│   │  Exploration: anti_negative_scorer, cluster_explorer, similar_ads,
│   │    mmr_reranker, feature_filter
│   │  Diagnostics: engagement_analyzer, pool_stats, history_lookup
│
├── evaluation/
│   ├── evaluate.py               # Recall/Precision/NDCG evaluator
│   ├── evaluate_pipeline.py      # Per-stage survival + truncation robustness
│   ├── baseline.py               # Dot-product baseline
│   └── prod_recall.py            # Production recall utilities
│
├── scripts/
│   ├── run_pilot_diagnosis.py    # 5-question e2e diagnosis
│   ├── create_split_data.py      # Train/test split (no leakage)
│   ├── prepare_contexts.py       # Generate user context folders
│   ├── run_benchmark_batch.py    # Batch benchmark (5 per call)
│   ├── run_baseline_weighted.py  # Fixed-weight RRF baseline
│   └── ...                       # FBLearner flow scripts
│
└── outputs/                      # Benchmark outputs per run
    ├── pilot_diagnosis/          # Pilot report + dashboard + results
    └── system_architecture.html  # E2E architecture visualization
```

## Data Sources

| Dataset | Requests | Candidates/Req | Labels | Use |
|---------|---------|----------------|--------|-----|
| `data/local/model/raw/` | 306 | ~2,000 | RAA engagement | Source data |
| `data/local/model/split/` | 100 | ~2,000 | Train/test split | **PRIMARY** — all local eval |
| `data/local/model/enriched/` | 39 | — | Prod predictions (JSON) | PM-stage scoring |
| `data/local/model/full_pool/` | 1 (WIP) | ~190,000 | Train/test split | Full coverage eval |
| `data/local/eval/bulk_eval/` | 20 | ~2,000 | AI top-709 | Production-scale pilot |
| Bulk eval aggregate | 50+ | ~190,000 | PM/AI/AF orderings | FBLearner recall |

## Key Learnings

1. **PM truncation is the #1 bottleneck** — 72.6% of positive ads lost at PM stage. Improving PM scoring or ML Reducer has highest impact.
2. **ML Reducer > Heuristic** — ML-driven truncation preserves +6.8% more positives than cosine heuristic at 50% reduction.
3. **FR Centroid is irreplaceable** — 13.7 unique positives per request at 12% overlap with other routes.
4. **HSNN sweet spot at k=3** — 73% compute savings at near-peak recall; expanding beyond 5 has diminishing returns.
5. **Production routes dominate** — Exploration routes add <0.1% recall; their value is in diversity, not recall.
6. **Data leakage inflates by 30-50%** — Train/test split essential for realistic evaluation.
7. **Simple beats complex** — 3-signal RRF outperforms 9-signal. More signals add noise.
8. **Soft recall improves, hard recall doesn't** — RRF surfaces higher-VALUE ads but not more ads by count.
9. **Per-stage signals matter** — Only prior-stage signals are valid (PM→AI, not AI→AI).
