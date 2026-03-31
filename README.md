# Agent Recommendation Framework

LLM-orchestrated multi-route ads retrieval system that uses Claude Code as a hybrid orchestrator — calling retrieval tools, reasoning over results, and producing ranked ad lists evaluated against production RAA ground truth.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Claude Code (Orchestrator)                       │
│  Reads: user context, architecture, skill guide,  │
│         learnings from past runs                  │
│  Decides: which tools to call, strategy to use    │
│  Adapts: signal quality → strategy selection      │
└────────┬───────────────────────────┬──────────────┘
         │ MCP tool calls            │
   ┌─────▼─────┐  ┌─────────┐  ┌───▼──────┐  ┌──────────┐
   │ Embedding  │  │ FR      │  │ Cluster  │  │ Prod     │
   │ Similarity │  │ Centroid│  │ Explorer │  │ Model    │
   │ Search     │  │ Search  │  │          │  │ Ranker   │
   └────────────┘  └─────────┘  └──────────┘  └──────────┘
   + anti_negative_scorer, similar_ads, mmr_reranker,
     feature_filter, engagement_analyzer, pool_stats,
     history_lookup (11 tools total)
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

## Key Results

| Metric | Baseline | Best Agent | Improvement |
|--------|----------|------------|-------------|
| Recall@100 (all) | 12.9% | **18.7%** | **+45%** |
| Recall@100 (weak signal) | 6.6% | **16.8%** | **+155%** |
| Recall@100 (strong signal) | 17.3% | **26.8%** | **+55%** |

### Adaptive Strategy
- **Strong signal** (similarity_gap > 0.05): Embedding-first + multi-route expansion
- **Weak signal** (similarity_gap < 0.01): FR centroid + cluster engagement rates (embedding abandoned)
- **Moderate**: Hybrid blending of all signals

## Quick Start

```bash
# 1. Generate user context files
python3 scripts/prepare_contexts.py --max-requests 10

# 2. Run agent benchmark (Claude Code headless)
python3 scripts/run_benchmark_cc.py --run-id my_run --max-requests 5 --model sonnet

# 3. Evaluate
python3 evaluation/evaluate.py --run-id my_run --baseline evaluation/results/baseline.json

# 4. Production-aligned recall (via FBLearner flow)
python3 scripts/run_recall_pipeline.py --run-id my_run --ds 2026-03-19 --dry-run
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
│   ├── baseline.py               # Dot-product baseline
│   ├── evaluate.py               # Recall/precision/NDCG evaluator
│   ├── compare_runs.py           # Run comparison
│   ├── prod_recall.py            # Production-aligned recall utilities
│   └── results/                  # Evaluation results per run
│
├── scripts/
│   ├── prepare_contexts.py       # Generate per-user context folders
│   ├── run_benchmark_cc.py       # Claude Code headless benchmark runner
│   ├── run_benchmark.py          # Claude API benchmark runner
│   ├── run_recall_pipeline.py    # Production recall SQL pipeline
│   ├── upload_agent_scores.py    # Upload scores to Hive
│   ├── update_history.py         # Accumulate results to history
│   └── extract_prod_predictions.py  # Extract prod_prediction from Hive
│
├── data/ → ../mvp/data/real_data_light/  # 306 requests (symlink)
├── data_enriched/                # Prod predictions per request
├── requests/                     # Per-user context folders
└── outputs/                      # Benchmark outputs per run
```

## Data

- **306 requests** from RAA (`raa_final_with_labels`, ds=2026-03-19)
- Each request: ~200 positive (engaged) ads + ~2000 negative ads
- 32d PSelect embeddings (user + ad towers)
- Production model predictions (59% coverage from `gr_p_select_bulk_eval_input_table`)

## Production Recall Pipeline

The recall computation follows the exact same logic as production PSelect/GR workflows (`workflows_pselect_recall.py`):

```
PM rank → blend (main + agent route at proportion X)
→ FM truncation (11,119 for MF)
→ Post-FM dedup (ee_key, DCO cap, campaign cap, account conv cap)
→ AI ranking (top 709)
→ FULL OUTER JOIN prod vs RAA
→ soft_recall, hard_recall, winsorized_soft_recall
```

FBLearner workflow at: `fbcode/fblearner/flow/projects/ads/agent_recommendation/workflows_agent_recall.py`
