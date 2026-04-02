---
name: run
description: Run the full e2e agent recommendation pipeline — data check, precompute, recommend, evaluate, analyze, report
---

# Run E2E Pipeline

Single command to run the entire agent recommendation flow: verify data → pre-compute tool results → spawn parallel agents for reasoning → evaluate against ground truth → run diagnosis → present findings.

## Arguments

- First arg or `--requests N`: Number of requests to process (default: 20)
- `--run-id ID`: Run identifier (default: `run_YYYYMMDD_HHMMSS`)
- `--data-dir PATH`: Data directory (default: `data/local/model/split`)
- `--skip-setup`: Skip data verification step
- `--wave-size N`: Agents per wave (default: 5)

## Workflow

### Step 1: Data Check

Verify data is prepared. If not, run setup steps.

```bash
echo "=== Data Check ==="
echo "split:     $(ls data/local/model/split/*.npz 2>/dev/null | wc -l) files"
echo "enriched:  $(ls data/local/model/enriched/*.json 2>/dev/null | wc -l) files"
echo "user ctx:  $(ls -d user/*/ 2>/dev/null | wc -l) folders"
```

If split data is empty:
```bash
python3 scripts/create_split_data.py --data-dir data/local/model/raw --max-requests {N}
```

If user context folders are empty:
```bash
python3 scripts/prepare_contexts.py --data-dir data/local/model/split --output-dir user/
```

If `--skip-setup` is provided, skip this step entirely.

### Step 2: Pre-compute Tool Results

Run all retrieval tools via Python (no LLM needed, ~3s/request):

```bash
python3 scripts/precompute_tool_results.py \
  --data-dir {data_dir} \
  --output-dir /tmp/tool_results \
  --max-requests {N}
```

Read `/tmp/tool_results/manifest.json` to get the list of request_ids.

### Step 3: Create Output Directory

```bash
mkdir -p outputs/{run_id}
```

Set `run_id` to provided value or generate: `run_$(date +%Y%m%d_%H%M%S)`.

### Step 4: Recommend via Parallel Agents

Process requests in waves of `wave_size` (default 5). For each wave, spawn agents **in a single message** with `run_in_background: true`.

**Agent prompt for each request:**

```
You are an ads recommendation agent. Your job is to produce the best possible
ranked list of 200 ad IDs by combining pre-computed retrieval results.

## Input
Read /tmp/tool_results/{request_id}.md (pre-computed tool results) and
user/{request_id}/ context files (profile.md, engagement.md, interest_clusters.md).

## Algorithm: Weighted Rank Fusion with Cluster Awareness

### Phase 1: Assess Signal Quality
Read engagement_pattern_analyzer from the tool results:
- similarity_gap: measures how well user embedding separates positive from negative ads
- centroid_gap: measures FR centroid signal strength
- overlap_fraction: >0.5 means embedding is weak

Determine signal regime:
- STRONG: similarity_gap > 0.05 AND overlap_fraction < 0.3
- WEAK: similarity_gap < 0.01 OR overlap_fraction > 0.5
- MODERATE: everything else

### Phase 2: Score Every Ad
For each ad that appears in ANY route, compute a combined score using
WEIGHTED RECIPROCAL RANK FUSION:

  score(ad) = sum over routes of: weight[route] / rank_in_route(ad)

Route weights by signal regime:

| Route              | STRONG | MODERATE | WEAK  |
|--------------------|--------|----------|-------|
| prod_model_ranker  | 3.0    | 3.0      | 3.0   |
| forced_retrieval   | 2.0    | 2.5      | 3.0   |
| pselect_main_route | 2.5    | 1.5      | 0.5   |
| anti_negative      | 1.5    | 1.5      | 2.0   |
| cluster_explorer   | 1.0    | 1.0      | 1.5   |

Bonus: if an ad appears in 3+ routes, add +1.0 to its score (consensus bonus).
Bonus: if an ad is in top_positive_ad_ids from engagement_pattern_analyzer, add +2.0.

### Phase 3: Cluster-Aware Diversity
Read cluster engagement rates from engagement_pattern_analyzer.
After scoring, ensure the top-100 covers all high-engagement clusters (rate > 5%):
- For each high-engagement cluster, at least 10 ads from that cluster
  should appear in the top 100.
- If a cluster is under-represented, promote its highest-scoring ads up.
- Cap any single cluster at 40% of the top-100 to avoid over-concentration.

### Phase 4: Final Ranking
1. Sort all ads by combined score (descending)
2. Apply cluster diversity adjustment from Phase 3
3. Output the top 200 ads

## Output
Write JSON to outputs/{run_id}/{request_id}.json:
{
  "request_id": {request_id},
  "ranked_ads": [id1, id2, ...],
  "strategy": "signal={STRONG|MODERATE|WEAK} gap=X.XX, top_routes=[...], n_consensus=N, cluster_coverage=[...]"
}

Include exactly 200 ranked ad IDs. This is critical — the more ads you rank,
the higher the recall. Parse ALL ad IDs from the "Full result ad_ids" lines
in the tool results, not just the top-10 summaries.
```

**Agent settings:** `mode: "auto"`, `run_in_background: true`

Wait for each wave to complete before spawning the next.

### Step 5: Evaluate

After all agents complete, run evaluation:

```bash
python3 evaluation/evaluate.py --run-id {run_id} --data-dir {data_dir}
```

Read `evaluation/results/{run_id}.json` for recall@K metrics.

### Step 6: Analyze (Pilot Diagnosis)

Run diagnosis on the same requests:

```bash
python3 scripts/run_pilot_diagnosis.py \
  --max-requests {N} \
  --data-dir {data_dir} \
  --output-dir outputs/{run_id}/diagnosis
```

Read `outputs/{run_id}/diagnosis/report.md` and `results.json`.

### Step 7: Write Metadata

Write `outputs/{run_id}/meta.json`:
```json
{
  "run_id": "{run_id}",
  "data_dir": "{data_dir}",
  "n_requests": N,
  "request_ids": [...],
  "timestamp": "..."
}
```

### Step 8: Present Report

Combine evaluation + diagnosis into a final report. Present:

**Agent Performance:**
- Recall@K table (agent vs baseline)
- NDCG@K and Precision@K

**Pipeline Diagnosis (from pilot):**
- Q1: Stage drop-off (AP→PM→AI→AF positive survival rates)
- Q2: Route uniqueness (which routes contribute unique positives)
- Q3: ML Reducer vs heuristic truncation
- Q4: HSNN exploration budget (recall vs compute savings)
- Q5: Production vs exploration route value

**Key Findings:**
- Top 3-5 actionable insights with evidence
- Suggested improvements (what to change, expected impact, how to implement)

**Summary line:**
```
Run {run_id} complete: {N} requests, recall@100={X}% (+Y% vs baseline),
{Z} findings. Run /analyze --run-id {run_id} for deeper analysis.
```
