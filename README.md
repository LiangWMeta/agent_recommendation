# Agent Recommendation Framework

LLM-orchestrated ads retrieval system that uses Claude Code to reason about the production **cascaded pipeline (AP → PM → AI → AF)** — calling retrieval tools, analyzing stage-by-stage filtering, and producing ranked ad lists with e2e pipeline awareness.

## Claude Code Skills

| Command | Purpose |
|---------|---------|
| `/recommend` | Run agent recommendation on a batch of requests via MCP tools |
| `/analyze` | Run diagnosis on requests, reason about results, suggest improvements |
| `/modify` | Modify the system via natural language prompt (add route, add data, change algo) |
| `/data-setup` | Pull and prepare all data needed to run the agent |

**Typical workflow:**
```
/data-setup                              # Prepare data (first time only)
/recommend --requests 20 --run-id my_run # Run agent on 20 requests
/analyze --run-id my_run                 # Analyze those results + suggest improvements
/modify "add a route that scores by ad recency"  # Evolve the system
/analyze --requests 20                   # Re-analyze with the new route
```

`/analyze` links to `/recommend` via `--run-id` — it evaluates the exact same requests and also runs pipeline diagnosis. Without `--run-id`, it auto-discovers the most recent run or does fresh diagnosis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CONTEXT LAYER (read before tools)                              │
│  ads_pool/              user/{request_id}/                      │
│  - catalog.md           - profile.md, intent.md, interests.md   │
│  - pool_overview.md     - session_history.md, engagement.md     │
│  - semantic_clusters.md - context.md                            │
│  (refreshed regularly)  (assembled per request)                 │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Claude Code Orchestrator                                       │
│  Signal assessment → Adaptive routing → Pipeline reasoning      │
└──────┬──────────────────────────────────────────────┬───────────┘
       ▼                                              ▼
  PRODUCTION TRACK (7 tools)          EXPLORATION TRACK (5 tools)
  pselect_main_route, forced_retrieval,      anti_negative, cluster_explorer,
  hsnn_cluster, prod_model,           similar_ads, mmr_reranker,
  pipeline_simulator, ml_reducer,     feature_filter
  parallel_routes_blender
       │                                              │
       └──────────────────┬───────────────────────────┘
                          ▼
                   DIAGNOSTICS (3 tools)
                   engagement_analyzer, pool_stats, history_lookup
                          │
                          ▼
                   Evaluation (per-stage recall, pipeline metrics)
```

**Production pipeline simulated**: AP (190K) → PM (11K) → AI (709) → AF (84)

## Tools (15 total)

### Production Track

| Tool | Production Component | Stage |
|------|---------------------|-------|
| `pselect_main_route` | PSelect / TTSN Main Route (primary ANN retrieval) | AP/PM |
| `forced_retrieval` | Forced Retrieval (85% of impressions, independent route) | AP |
| `prod_model_ranker` | SlimDSNN PM (calibrated CTR prediction) | PM |
| `hsnn_cluster_scorer` | HSNN (hierarchical cluster scoring, sublinear cost) | AP/PM |
| `pipeline_simulator` | Cascaded pipeline (AP→PM→AI→AF simulation) | All |
| `ml_reducer` | ML Reducer/Truncator (replaces heuristic truncation) | PM/AI |
| `parallel_routes_blender` | PRM + ML Blender (multi-route integration) | PM |

### Exploration Track

| Tool | Purpose |
|------|---------|
| `anti_negative_scorer` | Directional scoring: toward positive, away from negative centroids |
| `cluster_explorer` | Flat K-means clustering (HSNN baseline) |
| `similar_ads_lookup` | Ad-to-ad expansion from engaged ads |
| `mmr_reranker` | Maximal Marginal Relevance diversity re-ranking |
| `feature_filter` | Embedding feature filtering |

### Diagnostics: `engagement_pattern_analyzer`, `ads_pool_stats`, `lookup_similar_requests`

## Results (100 requests, Claude-orchestrated)

### Agent Performance

| Metric | Agent (Claude) | RRF (fixed-weight) | Baseline (cosine) | vs Baseline |
|--------|---------------|--------------------|--------------------|-------------|
| Recall@50 | 10.09% | 8.89% | 7.13% | **+41%** |
| Recall@100 | **18.29%** | 16.36% | 12.85% | **+42%** |
| NDCG@100 | 36.81% | 33.11% | — | — |

Claude's adaptive reasoning adds **+1.93%** recall@100 over fixed-weight RRF.

### Pipeline Diagnosis (100 requests)

| Finding | Numbers | Implication |
|---------|---------|-------------|
| PM truncation is #1 bottleneck | 54.1% of positives lost | Improve PM scoring or widen budget |
| ML Reducer > Heuristic | **+9.4%** positive preservation | Validates ML Reducer investment |
| Forced Retrieval irreplaceable | 14.2 unique positives, 7% overlap | Must remain as dedicated PRM route |
| HSNN sweet spot at k=3 | 71% compute savings, near-peak recall | Expanding beyond 5 has diminishing returns |
| Exploration routes add value | **+2.3%** recall at scale | `similar_ads_lookup` and `anti_negative` should be included |
| Full pipeline survival | 3.1% of positives reach AF | Cross-stage consistency is critical |

## Quick Start

```bash
# Using Claude Code skills (recommended)
/data-setup                                # Prepare all data
/recommend --requests 20 --run-id my_run   # Run agent on 20 requests
/analyze --run-id my_run                   # Analyze results + suggest improvements
/modify "add a new retrieval route"        # Evolve the system

# Using scripts directly (see data/datasets.md for full registry)
python3 scripts/create_split_data.py --data-dir data/local/model/raw
python3 scripts/run_baseline_weighted.py --run-id my_run --data-dir data/local/model/split --max-requests 20
python3 scripts/run_pilot_diagnosis.py --max-requests 20
python3 evaluation/evaluate.py --run-id my_run
```

## Directory Structure

```
agent_recommendation/
├── .claude/skills/               # Claude Code skills
│   ├── recommend/SKILL.md        # /recommend
│   ├── analyze/SKILL.md          # /analyze
│   ├── modify/SKILL.md           # /modify
│   └── data-setup/SKILL.md       # /data-setup
│
├── ads_pool/                     # Ads Pool Understanding (refreshed regularly)
│   ├── catalog.md                # All ads: category, objective, creative, CTR
│   ├── pool_overview.md          # Pool summary
│   └── semantic_clusters.md      # HSNN cluster → semantic labels
│
├── user/{request_id}/            # User Context (per request)
│   ├── profile.md, intent.md, interests.md
│   ├── session_history.md, engagement.md, context.md
│
├── data/
│   ├── datasets.md               # Dataset registry: schemas, quickstart, lineage
│   └── local/
│       ├── model/                 # raw/, split/ (PRIMARY), enriched/, full_pool/
│       └── eval/bulk_eval/        # Production pipeline extracts
│
├── tools/                        # 15 MCP retrieval tools (7 prod + 5 explore + 3 diag)
├── evaluation/                   # evaluate.py, evaluate_pipeline.py
├── scripts/                      # Benchmarks, data prep, pilot diagnosis
└── outputs/                      # Results, reports, visualizations
```

## Key Learnings

1. **Claude reasoning adds +1.9% recall over RRF** — adaptive route selection helps, especially for weak-signal requests.
2. **PM truncation is the #1 bottleneck** — 54% of positive ads lost at PM. Improving PM scoring or ML Reducer has highest impact.
3. **ML Reducer > Heuristic** — +9.4% more positives preserved at 50% reduction.
4. **Forced Retrieval is irreplaceable** — 14.2 unique positives per request at 7% overlap with other routes.
5. **Exploration routes add +2.3% recall at scale** — `similar_ads_lookup` and `anti_negative` should be included in production blend.
6. **HSNN sweet spot at k=3** — 71% compute savings at near-peak recall.
7. **Small samples underestimate** — 10-request pilot showed exploration adds <0.1%; 100 requests revealed +2.3%.
8. **Data leakage inflates by 30-50%** — Train/test split essential.
