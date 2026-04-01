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
/recommend --requests 20                 # Run agent on 20 requests
/analyze --requests 20                   # Analyze system behavior
/modify "add a route that scores by ad recency"  # Evolve the system
/analyze --requests 20                   # Verify improvement
```

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
  embedding_search, fr_centroid,      anti_negative, cluster_explorer,
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

## Pilot Results (10 requests)

| Finding | Numbers | Implication |
|---------|---------|-------------|
| PM truncation is #1 bottleneck | 72.6% of positives lost | Improve PM scoring or widen budget |
| ML Reducer > Heuristic | +6.8% positive preservation | Validates ML Reducer investment |
| FR Centroid irreplaceable | 13.7 unique positives, 12% overlap | Must remain as dedicated PRM route |
| HSNN sweet spot at k=3 | 73% compute savings, near-peak recall | Expanding beyond 5 has diminishing returns |
| Production routes dominate | Exploration adds <0.1% recall | Exploration value is in diversity, not recall |
| Full pipeline survival | Only 0.5% of positives reach AF | Cross-stage consistency is critical |

## Quick Start

```bash
# Using Claude Code skills (recommended)
/data-setup                        # Prepare all data
/recommend --requests 20           # Run agent
/analyze --requests 20             # Analyze + suggest improvements

# Using scripts directly (see data/datasets.md for full registry)
python3 scripts/create_split_data.py --data-dir data/local/model/raw
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

1. **PM truncation is the #1 bottleneck** — 72.6% of positive ads lost at PM. Improving PM scoring or ML Reducer has highest impact.
2. **ML Reducer > Heuristic** — +6.8% more positives preserved at 50% reduction.
3. **FR Centroid is irreplaceable** — 13.7 unique positives per request at 12% overlap with other routes.
4. **HSNN sweet spot at k=3** — 73% compute savings at near-peak recall.
5. **Production routes dominate** — Exploration routes add <0.1% recall.
6. **Data leakage inflates by 30-50%** — Train/test split essential.
7. **Per-stage signals matter** — Only prior-stage signals are valid (PM→AI, not AI→AI).
