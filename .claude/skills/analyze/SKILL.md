---
name: analyze
description: Run diagnosis on requests, reason about system behavior, and suggest improvements
---

# Analyze

Run the pilot diagnosis across N requests, then reason about the results to identify bottlenecks, evaluate route contributions, and suggest concrete improvements.

## Arguments

- First arg or `--requests N`: Number of requests for diagnosis (default: 20)
- `--run-id ID`: Evaluate outputs from a specific `/recommend` run (reads from `outputs/{run_id}/`)
- `--data-dir PATH`: Data directory (default: `data/local/model/split`)

## Workflow

### 1. Determine what to analyze

**If `--run-id` is provided:**
- Read `outputs/{run_id}/meta.json` to find which requests were processed and which data_dir was used
- If `meta.json` doesn't exist, list `outputs/{run_id}/*.json` to find request IDs (extract from filenames)
- Use those same request IDs and data_dir for the diagnosis
- This ensures analysis covers the exact same requests that `/recommend` processed

**If no `--run-id`:**
- Find the most recent run: `ls -td outputs/run_* 2>/dev/null | head -1`
- If a recent run exists, offer to analyze it: "Found run {run_id} with N requests. Analyzing this run."
- If no runs exist, run fresh diagnosis on `--requests N` requests from data_dir

### 2. Run pilot diagnosis

Run diagnosis on the identified requests:

```bash
python3 scripts/run_pilot_diagnosis.py \
  --max-requests {N} \
  --data-dir {data_dir} \
  --output-dir outputs/pilot_diagnosis
```

This answers 5 questions:
- Q1: Stage drop-off (AP→PM→AI→AF positive survival)
- Q2: Route uniqueness (which routes contribute unique positives)
- Q3: ML Reducer vs heuristic truncation
- Q4: HSNN exploration budget (recall vs compute)
- Q5: Production vs exploration route value

### 3. Read results

Read these files:
- `outputs/pilot_diagnosis/results.json` — raw numbers
- `outputs/pilot_diagnosis/report.md` — formatted report

### 4. If --run-id provided, also evaluate the recommend outputs

Evaluate the agent's ranked_ads against ground truth:

```bash
python3 evaluation/evaluate.py --run-id {run_id} --data-dir {data_dir}
python3 evaluation/evaluate_pipeline.py --run-id {run_id}
```

Read the evaluation results from `evaluation/results/{run_id}.json` and `evaluation/results/{run_id}_pipeline.json`.

This adds per-request recall@K, NDCG, and pipeline survival metrics for the agent's actual recommendations (vs the system-level diagnosis from the pilot).

### 5. Read system context

Read these for background:
- `architecture.md` — pipeline architecture, tool organization
- `skill.md` — current reasoning strategy
- `learnings.md` — previous findings

### 6. Reason and present findings

Analyze the diagnosis results and present insights on:

**Stage Drop-Off Analysis**:
- Which stage loses the most positives? Why?
- Is PM truncation too aggressive? Could widening the budget help?
- How does cross-stage rank correlation look?

**Route Value Analysis**:
- Which routes contribute the most unique positive ads?
- Which routes have high overlap (redundant)?
- Are exploration routes adding value beyond production routes?

**Signal Quality Distribution**:
- What fraction of requests have strong/moderate/weak signal?
- Do weak-signal requests need a different strategy?

**Truncation & Efficiency**:
- Does ML Reducer outperform heuristic? By how much?
- What's the HSNN sweet spot for compute/recall tradeoff?

**If recommend results available** (--run-id):
- Per-request recall@K breakdown
- Which requests performed well/poorly and why?
- Pipeline survival rates for the agent's ranked lists

### 7. Suggest improvements

Based on evidence, suggest 3-5 concrete improvements. Each should include:
- What to change
- Expected impact (with numbers from the analysis)
- How to implement (which files to modify, or use `/modify`)
- Priority (high/medium/low)

### 8. Update learnings

Append new findings to `learnings.md` with the date and evidence. Only add genuinely new insights — don't repeat what's already there.
