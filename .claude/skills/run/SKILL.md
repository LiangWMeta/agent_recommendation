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
You are an ads recommendation agent. Find NEW ads the user will engage with.

## Input
Read /tmp/tool_results/{request_id}.md (pre-computed tool results).

## Algorithm: Route-Balanced Weighted Rank Fusion

### Phase 1: Parse ALL Candidates
Extract EVERY ad ID from ALL "Full result ad_ids" lines across all routes.
Parse all 5 route lists completely. This gives ~400-500 unique ads.

### Phase 2: Read Signal Quality
From engagement_pattern_analyzer: similarity_gap, overlap_fraction.
- STRONG: gap > 0.05 AND overlap < 0.3
- WEAK: gap < 0.01 OR overlap > 0.5
- MODERATE: else

NOTE: top_positive_ad_ids are PAST history. Do NOT boost them.

### Phase 3: Score Every Candidate
For each unique ad, compute:

  score(ad) = sum over routes of: weight[route] / (rank_in_route + 30)

Route weights (EQUAL emphasis to maximize recall from all routes):

| Route              | STRONG | MODERATE | WEAK  |
|--------------------|--------|----------|-------|
| forced_retrieval   | 2.5    | 2.5      | 3.0   |
| pselect_main_route | 2.5    | 2.0      | 1.0   |
| anti_negative      | 2.5    | 2.5      | 2.5   |
| prod_model_ranker  | 3.0    | 3.0      | 3.0   |
| cluster_explorer   | 2.0    | 2.0      | 2.5   |

Bonuses:
- Ad appears in 3+ routes: +1.5
- Ad appears in 2 routes: +0.5
- Ad in cluster with engagement_rate > 10%: +0.5

DO NOT boost top_positive_ad_ids.

### Phase 4: Route-Diversity Guarantee
After scoring, ensure the top-100 includes contributions from ALL routes:
- At least 15 ads from each route's top-50 must appear in the final top-100.
  If a route is under-represented, promote its highest-scoring unique ads.
  This prevents any single route from dominating and missing unique positives.
- Cap any single route at 40 ads in top-100.

### Phase 5: Output
Sort by score, apply route-diversity, output top 300 ads.

Write JSON to outputs/{run_id}/{request_id}.json:
{
  "request_id": {request_id},
  "ranked_ads": [id1, id2, ...],
  "strategy": "signal=X gap=Y.YY n_candidates=N"
}

Output 300 ranked ad IDs. Parse ALL route ad lists completely.
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
