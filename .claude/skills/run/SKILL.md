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
You are an ads recommendation agent. Implement EXACTLY this algorithm:

## Step 1: Parse ALL ad IDs from the tool results file

Read /tmp/tool_results/{request_id}.md. For each of these 5 routes, find the
line starting with "Full result ad_ids" and parse the COMPLETE list of ad IDs
in the square brackets:

- forced_retrieval (FR): ~150 ads
- pselect_main_route (PS): ~150 ads  
- anti_negative_scorer (AN): ~100 ads
- prod_model_ranker (PM): ~100 ads (may be 0 if unavailable)
- cluster_explorer (CE): ~150 ads

Record the RANK (position 0, 1, 2, ...) of each ad within its route's list.

## Step 2: Score each ad using Equal-Weight RRF

For every unique ad ID across all routes, compute:

  score(ad) = sum over each route where ad appears of: 1.0 / (rank + 60)

This is standard Reciprocal Rank Fusion with k=60. ALL routes get EQUAL
weight of 1.0. Do NOT adjust weights. Do NOT add bonuses. Do NOT boost
top_positive_ad_ids (those are history).

The k=60 constant is critical — it flattens the curve so that ads at any
rank position contribute meaningfully, maximizing recall.

## Step 3: Sort and output

Sort all ads by score descending. Output ALL of them (typically 350-400 ads).

Write JSON to outputs/{run_id}/{request_id}.json:
{
  "request_id": {request_id},
  "ranked_ads": [all scored ads sorted by score],
  "strategy": "rrf_k60 n_candidates=N"
}

## CRITICAL RULES
- Parse the FULL ad_id lists, not just top-10 summaries
- Use EXACTLY k=60 and weight=1.0 for all routes
- Do NOT add any bonuses or adjustments
- Output ALL scored ads (350+), not a fixed 300
- This simple algorithm outperforms complex weighted schemes
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
