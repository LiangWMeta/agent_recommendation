---
name: analyze
description: Run diagnosis on requests, reason about system behavior, and suggest improvements
---

# Analyze

Run the pilot diagnosis across N requests, then reason about the results to identify bottlenecks, evaluate route contributions, and suggest concrete improvements.

## Arguments

- First arg or `--requests N`: Number of requests for diagnosis (default: 20)
- `--run-id ID`: Also evaluate outputs from a specific `/recommend` run
- `--data-dir PATH`: Data directory (default: `data/local/model/split`)

## Workflow

### 1. Run pilot diagnosis

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

### 2. Read results

Read these files:
- `outputs/pilot_diagnosis/results.json` — raw numbers
- `outputs/pilot_diagnosis/report.md` — formatted report

### 3. If --run-id provided, evaluate recommend outputs

```bash
python3 evaluation/evaluate.py --run-id {run_id} --data-dir {data_dir}
python3 evaluation/evaluate_pipeline.py --run-id {run_id}
```

Read the evaluation results from `evaluation/results/{run_id}.json` and `evaluation/results/{run_id}_pipeline.json`.

### 4. Read system context

Read these for background:
- `architecture.md` — pipeline architecture, tool organization
- `skill.md` — current reasoning strategy
- `learnings.md` — previous findings

### 5. Reason and present findings

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

**If recommend results available**:
- Per-request recall@K breakdown
- Which requests performed well/poorly and why?
- Pipeline survival rates for the agent's ranked lists

### 6. Suggest improvements

Based on evidence, suggest 3-5 concrete improvements. Each should include:
- What to change
- Expected impact (with numbers from the analysis)
- How to implement (which files to modify)
- Priority (high/medium/low)

Example: "Widen PM budget from 500 to 700 — pilot shows 72.6% positive loss at PM. Even a 10% wider budget could recover ~7% of lost positives. Modify `pipeline_simulator.py` default `pm_budget`."

### 7. Update learnings

Append new findings to `learnings.md` with the date and evidence. Only add genuinely new insights — don't repeat what's already there.
