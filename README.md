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

## Tools (16 total)

### Production Track (three parallel flows + blender)

```
Flow 1: PSelect (ANN) ─────────────────────┐
Flow 2: Forced Retrieval (flagged, eCPM) ──┤→ Blender → Final
Flow 3: ML Truncator → Prod Model ────────┘
         (rank_all or with_hsnn, eCPM)
```

**Flow 1: PSelect**
| `pselect_main_route` | PSelect / TTSN Main Route (ANN retrieval) | AP |

**Flow 2: Forced Retrieval**
| `forced_retrieval` | FR-flagged ads ranked by eCPM (centroid fallback) | AP |

**Flow 3: Main Flow**
| `ml_reducer` | ML Truncator (reduce candidates before heavy scoring) | PM |
| `prod_model_ranker` | eCPM scoring: rank_all or with_hsnn mode | PM |

**Aggregation**
| `parallel_routes_blender` | Blend all 3 flows | PM |
| `pipeline_simulator` | Simulate AP→PM→AI→AF cascade | All |

### Exploration Track

| Tool | Purpose |
|------|---------|
| `fr_centroid_search` | Centroid-based FR approximation (research baseline) |
| `hsnn_cluster_scorer` | Standalone HSNN hierarchy study |
| `anti_negative_scorer` | Directional scoring toward/away from centroids |
| `cluster_explorer` | Flat K-means clustering |
| `similar_ads_lookup` | Ad-to-ad expansion from engaged ads |
| `mmr_reranker` | MMR diversity re-ranking |
| `feature_filter` | Embedding feature filtering |

### Diagnostics: `engagement_pattern_analyzer`, `ads_pool_stats`, `lookup_similar_requests`

## Results (100 requests, Claude-orchestrated)

### Agent Performance

| Metric | Agent (Haiku) | 3-Flow RRF Baseline | Cosine-Only | vs RRF | vs Cosine |
|--------|--------------|--------------------|----|--------|-----------|
| Recall@10 | 2.19% | 1.99% | 1.77% | +0.20% | +0.42% |
| Recall@20 | 3.78% | 3.72% | 3.30% | +0.06% | +0.48% |
| Recall@50 | 8.86% | 8.46% | 7.13% | +0.40% | +1.73% |
| **Recall@100** | **16.41%** | **15.58%** | 12.85% | **+0.83%** | +3.56% |
| NDCG@100 | 33.83% | 31.54% | — | +2.29% | — |

Claude's adaptive reasoning adds **+0.83% recall@100** over fixed-weight 3-flow RRF baseline.

### Pipeline Diagnosis (100 requests)

| Finding | Numbers | Implication |
|---------|---------|-------------|
| PM truncation is #1 bottleneck | 54.1% of positives lost | Improve PM scoring or widen budget |
| ML Reducer > Heuristic | **+9.4%** positive preservation | Validates ML Reducer investment |
| Forced Retrieval irreplaceable | 14.2 unique positives, 7% overlap | Must remain as dedicated route |
| HSNN sweet spot at k=3 | 71% compute savings, near-peak recall | Expanding beyond 5 has diminishing returns |
| Exploration routes add value | **+2.3%** recall at scale | `similar_ads_lookup` and `anti_negative` should be included |
| Full pipeline survival | Only 3.1% of positives reach AF | Cross-stage consistency is critical |
| Full pipeline survival | 3.1% of positives reach AF | Cross-stage consistency is critical |

## Quick Start

```bash
# Using Claude Code skills (recommended)
/data-setup                                # Prepare all data
/recommend --requests 20 --run-id my_run   # Run agent on 20 requests
/analyze --run-id my_run                   # Analyze results + suggest improvements
/modify "add a new retrieval route"        # Evolve the system
```

`/recommend` uses a two-phase approach:
1. **Pre-compute** all tool results via Python (fast, ~3s/request)
2. **Spawn parallel Agent subagents** to reason over results (no subprocess startup overhead)

```bash
# Or use scripts directly (see data/datasets.md for full registry)
python3 scripts/precompute_tool_results.py --max-requests 20   # Phase 1: pre-compute
python3 scripts/run_baseline_weighted.py --run-id my_run --max-requests 20  # Fixed-weight RRF
python3 evaluation/evaluate.py --run-id my_run                 # Evaluate
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

1. **Agent adds +3.6% recall@100 over baseline** (+28%) — adaptive route selection based on signal quality.
2. **Three parallel flows** — PSelect + Forced Retrieval + Main Flow (ML Truncator → Prod Model). HSNN is a mode within prod model, not a separate route.
3. **PM truncation is the #1 bottleneck** — 54% of positive ads lost. eCPM scoring (pCTR × pCVR × bid) via `pm_total_value` is more realistic than pCTR alone.
4. **ML Reducer > Heuristic** — +9.4% more positives preserved at 50% reduction.
5. **Forced Retrieval is irreplaceable** — 14.2 unique positives per request at 7% overlap.
6. **Exploration routes add +2.3% recall at scale** — `similar_ads_lookup` and `anti_negative` should be included.
7. **Agent Tool approach is 5x faster** — 6 parallel Haiku agents finish 100 requests in ~20 min vs 107 min with `claude -p` subprocess.
8. **Data leakage inflates by 30-50%** — Train/test split essential.
