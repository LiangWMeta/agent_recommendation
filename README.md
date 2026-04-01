# Agent Recommendation Framework

LLM-orchestrated ads retrieval system that models the production **cascaded pipeline (AP → PM → AI → AF)** with three parallel retrieval flows, pipeline-aware reasoning, and e2e diagnosis.

## Claude Code Skills

| Command | Purpose |
|---------|---------|
| `/run` | Full e2e pipeline: data check → precompute → recommend → evaluate → analyze → report |
| `/recommend` | Run agent recommendation on a batch of requests |
| `/analyze` | Run diagnosis + reason about results + suggest improvements |
| `/modify` | Modify system via natural language (add route, add data, change algo) |
| `/data-setup` | Prepare all data (quickstart chain) |

**Typical workflow:**
```
/run --requests 100 --run-id my_run        # Full e2e (recommended)

# Or step by step:
/data-setup                                # Prepare data (first time)
/recommend --requests 20 --run-id my_run   # Run agent
/analyze --run-id my_run                   # Analyze results
/modify "add a route that..."              # Evolve the system
```

## Architecture

Three parallel retrieval flows → blender → pipeline simulation:

```
┌─── Context Layer ──────────────────────────────────────────┐
│  ads_pool/ (catalog, clusters, pool overview)               │
│  user/{id}/ (profile, intent, interests, engagement)        │
└────────────────────────────┬───────────────────────────────┘
                             ▼
┌─── Claude Code Orchestrator ───────────────────────────────┐
│  Signal assessment → Adaptive routing → Pipeline reasoning  │
└──────┬─────────────────────────────────────────────────────┘
       ▼
┌─── Three Parallel Production Flows ───────────────────────┐
│                                                            │
│  Flow 1: PSelect ──────────────────────┐                   │
│    ANN retrieval (two-tower embeddings) │                   │
│                                        │                   │
│  Flow 2: Forced Retrieval ────────────┤→ Blender → Final  │
│    FR-flagged ads ranked by eCPM       │                   │
│                                        │                   │
│  Flow 3: Main Flow ───────────────────┘                   │
│    ML Truncator → Prod Model                               │
│    (rank_all or with_hsnn, eCPM)                           │
│                                                            │
├─── Exploration Track ─────────────────────────────────────┤
│  fr_centroid_search, hsnn_cluster_scorer, anti_negative,   │
│  cluster_explorer, similar_ads, mmr_reranker, feature_filter│
│                                                            │
├─── Diagnostics ───────────────────────────────────────────┤
│  engagement_analyzer, pool_stats, history_lookup           │
└────────────────────────────────────────────────────────────┘
       │
       ▼
┌─── Evaluation ─────────────────────────────────────────────┐
│  evaluate.py (Recall/NDCG) + evaluate_pipeline.py (stages) │
│  + run_pilot_diagnosis.py (5-question report)               │
└────────────────────────────────────────────────────────────┘
```

**Production pipeline simulated**: AP (190K) → PM (11K) → AI (709) → AF (84)

## Results (100 requests)

### Agent vs Baselines

| Metric | Agent (Haiku) | 3-Flow RRF | Cosine-Only | Agent vs RRF | Agent vs Cosine |
|--------|--------------|-----------|-------------|-------------|-----------------|
| Recall@50 | 8.86% | 8.46% | 7.13% | +0.40% | +1.73% |
| **Recall@100** | **16.41%** | **15.58%** | 12.85% | **+0.83%** | **+3.56%** |
| NDCG@100 | 33.83% | 31.54% | — | +2.29% | — |

- **Multi-route blending** (3-flow RRF vs cosine) adds **+2.73%** recall@100
- **Claude's reasoning** adds **+0.83%** on top of RRF (adaptive signal-quality routing)

### Pipeline Diagnosis (100 requests)

| Finding | Numbers | Implication |
|---------|---------|-------------|
| PM truncation is #1 bottleneck | 54.1% of positives lost | Improve PM scoring or widen budget |
| PSelect & FR equally unique | 95.2 unique each, 4.8% overlap | Complementary, both essential |
| FR finds more positives | 14.9 vs 11.2 unique positives | FR is highest-value route |
| ML Reducer > Heuristic | +9.4% positive preservation | Validates ML Reducer investment |
| Exploration routes | +1.3% (similar_ads best) | HSNN hurts (-1.0%), similar_ads helps (+1.3%) |
| Full pipeline survival | Only 3.1% reach AF | Cross-stage consistency critical |

## Tools (16 total)

### Production Track

**Flow 1**: `pselect_main_route` — PSelect / TTSN ANN retrieval
**Flow 2**: `forced_retrieval` — FR-flagged ads by eCPM (centroid fallback)
**Flow 3**: `ml_reducer` → `prod_model_ranker` — ML truncation then eCPM scoring (rank_all or with_hsnn)
**Aggregation**: `parallel_routes_blender`, `pipeline_simulator`

### Exploration Track

`fr_centroid_search`, `hsnn_cluster_scorer`, `anti_negative_scorer`, `cluster_explorer`, `similar_ads_lookup`, `mmr_reranker`, `feature_filter`

### Diagnostics

`engagement_pattern_analyzer`, `ads_pool_stats`, `lookup_similar_requests`

## Quick Start

```bash
# Full e2e (recommended)
/run --requests 20 --run-id my_run

# Or scripts directly
python3 scripts/precompute_tool_results.py --max-requests 20
python3 scripts/run_baseline_weighted.py --run-id baseline --max-requests 20
python3 scripts/run_pilot_diagnosis.py --max-requests 20
python3 evaluation/evaluate.py --run-id baseline
```

`/recommend` pre-computes tool results in Python (~3s/req), then spawns parallel Agent subagents for reasoning (~20 min for 100 requests with 6 Haiku agents).

## Directory Structure

```
agent_recommendation/
├── .claude/skills/               # /run, /recommend, /analyze, /modify, /data-setup
├── ads_pool/                     # Ads Pool Understanding (catalog, clusters)
├── user/{request_id}/            # User Context (profile, intent, engagement)
├── data/
│   ├── datasets.md               # Dataset registry + quickstart
│   └── local/model/              # raw, split (PRIMARY), enriched, full_pool
├── tools/                        # 16 tools (6 prod + 7 explore + 3 diag)
├── evaluation/                   # evaluate.py, evaluate_pipeline.py
├── scripts/                      # Benchmarks, precompute, diagnosis
└── outputs/                      # Per-run results + reports
```

## Key Learnings

1. **Multi-route blending is the biggest win** — 3-flow RRF adds +2.73% recall@100 over cosine-only.
2. **Claude reasoning adds +0.83% on top** — adaptive signal-quality routing helps weak-signal requests most.
3. **Three parallel flows** — PSelect + FR + Main Flow (ML Truncator → Prod Model). HSNN is a mode within prod model.
4. **PM truncation loses 54%** of positives — #1 bottleneck. eCPM scoring (pCTR × pCVR × bid) is more realistic.
5. **PSelect and FR are complementary** — 95.2 unique ads each, only 4.8% overlap, both essential.
6. **ML Reducer > Heuristic** — +9.4% more positives preserved at 50% reduction.
7. **HSNN as separate route hurts (-1.0%)** — confirms it should be a prod model mode, not independent.
8. **similar_ads_lookup is best exploration** — +1.3% recall, consistently finds unique positives.
9. **Agent Tool is 5x faster** — 6 parallel Haiku agents finish 100 requests in ~20 min vs 107 min subprocess.
10. **Data leakage inflates 30-50%** — train/test split essential for realistic evaluation.
