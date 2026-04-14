# Dataset Registry

Single source of truth for all datasets used by the agent recommendation framework — local files, remote sources, loading patterns, and leakage prevention rules.

## Quickstart: Pull Data to Run the Agent

The fastest way — `setup_data.py` checks what exists, builds what's missing:

```bash
python3 scripts/setup_data.py              # check + build all missing stages
python3 scripts/setup_data.py --check-only  # just report status
python3 scripts/setup_data.py --force       # rebuild everything
```

Or run individual steps manually:

```bash
# Step 1: Pull RAA source data (embeddings + labels from Hive)
python3 scripts/extract_raa.py --output-dir data/local/model/raw --max-requests 300

# Step 2: Create train/test split (prevents label leakage)
python3 scripts/create_split_data.py \
  --data-dir data/local/model/raw \
  --output-dir data/local/model/split \
  --max-requests 100

# Step 3: Extract production predictions (optional, improves PM scoring)
python3 scripts/extract_prod_predictions.py \
  --data-dir data/local/model/raw \
  --output-dir data/local/model/enriched

# Step 4: Generate user context folders
python3 scripts/prepare_contexts.py \
  --data-dir data/local/model/split \
  --output-dir user/

# Step 5: Generate ads pool understanding
python3 ads_pool/refresh.py --data-dir data/local/model/split

# Step 6: Run the agent
python3 scripts/run_pilot_diagnosis.py --data-dir data/local/model/split
```

**Dependency chain**: Step 1 → Step 2 → Steps 3,4,5 (parallel) → Step 6

## Quick Reference

| Dataset | Path | Requests | Ads/Req | Schema Keys | Primary Use |
|---------|------|---------|---------|-------------|-------------|
| Raw source | `data/local/model/raw/` | 306 | ~2K | user_emb, ad_embs, ad_ids, labels | Source data (don't use directly for eval) |
| Train/test split | `data/local/model/split/` | 100 | ~2K | + history_labels, test_labels | **PRIMARY** — all local eval |
| Prod predictions | `data/local/model/enriched/` | 39 | — | ad_id, prod_prediction (JSON) | PM-stage scoring signal |
| Full pool | `data/local/model/full_pool/` | 1 (WIP) | ~190K | Same as split | Full coverage evaluation |
| Bulk eval extract | `data/local/eval/bulk_eval/` | 20 | ~2K | Same as raw | Production pipeline pilot |

## Local Datasets

### `data/local/model/raw/` — Source Data

**Source**: RAA (Rank All Ads) engagement pipeline via Hive extraction.
**Contents**: 306 request NPZ files, each with ~2K candidate ads.

```python
# Schema
data = np.load("data/local/model/raw/request_{rid}.npz")
data["request_id"]   # scalar int — unique request identifier
data["user_emb"]     # shape [32] — user embedding (PSelect/TTSN, 32d)
data["ad_embs"]      # shape [N, 32] — candidate ad embeddings
data["ad_ids"]       # shape [N] — ad identifiers
data["labels"]       # shape [N] — binary engagement (1=engaged, 0=not)
```

**When to use**: As input to `create_split_data.py` and `extract_prod_predictions.py`. Do NOT use directly for evaluation — labels are not split, causing leakage.

### `data/local/model/split/` — Train/Test Split (PRIMARY)

**Source**: Derived from `raw/` via `scripts/create_split_data.py` with 50/50 split on positive labels.
**Contents**: 100 request NPZ files with leakage-free label split.

```python
# Schema (extends raw with split labels)
data = np.load("data/local/model/split/request_{rid}.npz")
data["history_labels"]  # shape [N] — labels tools can see (50% of positives)
data["test_labels"]     # shape [N] — labels for evaluation only (other 50%)
data["labels"]          # shape [N] — all labels (for reference, don't use in tools+eval)

# Standard loading pattern
labels_for_tools = data["history_labels"]   # pass to retrieval tools
labels_for_eval  = data["test_labels"]      # pass to evaluate_request()
```

**When to use**: All local benchmarks, pilot diagnosis, tool development. This is the **primary dataset** for everything.

### `data/local/model/enriched/` — Production Predictions

**Source**: Extracted from Hive via `scripts/extract_prod_predictions.py`.
**Contents**: 39 JSON sidecar files with SlimDSNN calibrated CTR predictions.

```python
# Schema: JSON list of per-ad predictions
# File: data/local/model/enriched/{request_id}_prod.json
[
    {"ad_id": 120241598724250381, "prod_prediction": 0.0547},
    {"ad_id": 120245678901234567, "prod_prediction": null},  # missing prediction
    ...
]

# Loading pattern (used by prod_model_ranker, pipeline_simulator, ml_reducer)
import json
with open(f"data/local/model/enriched/{request_id}_prod.json") as f:
    prod_data = json.load(f)
pred_map = {int(item["ad_id"]): float(item["prod_prediction"])
            for item in prod_data if item.get("prod_prediction") is not None}
```

**Coverage**: Not all requests have sidecars (39 of 306). Some ads within a request may have `null` predictions. Tools handle missing data gracefully with cosine fallback.

### `data/local/model/full_pool/` — Full 190K Candidate Pool

**Source**: Extracted from bulk eval aggregate via FBLearner flow.
**Contents**: 1 NPZ file (WIP) with ~190K candidates per request.
**Status**: Work in progress — expanding to more requests.

**When to use**: Full-coverage evaluation matching production scale. Used with FBLearner recall workflows.

### `data/local/eval/bulk_eval/` — Production Pipeline Extract

**Source**: Extracted from production bulk eval via `scripts/extract_bulk_eval_candidates.py` (FBLearner `extract_candidates_workflow`).
**Contents**: 20 NPZ files with ~2K candidates each, extracted from the AI top-709 production ranking.

**When to use**: Production-scale pilot testing. Used by `run_benchmark_fast.py`.

## Remote Datasets (Hive / Feature Store)

These are the upstream data sources. Local datasets are extracted from them.

| Name | Hive Table / Location | Schema | Extracted To | Script |
|------|----------------------|--------|-------------|--------|
| RAA engagement | `gr_p_select_bulk_eval_input_table` | request_id, ad_id, user_emb (32d), ad_emb (32d), label | `data/local/model/raw/` | `scripts/extract_raa.py` |
| Prod predictions | Same table, `prod_prediction` column | request_id, ad_id, prod_prediction (calibrated CTR) | `data/local/model/enriched/` | `scripts/extract_prod_predictions.py` |
| Bulk eval aggregate | FBLearner flow output | request_id, ad_id, pm_total_value, ai_rank, af_rank, features | `data/local/eval/bulk_eval/` | FBLearner `extract_candidates_workflow` |
| PSelect embeddings | Feature Store (TTSN PSelect group) | user_id → 32d float, ad_id → 32d float | Embedded in RAA table | — |

## Local ↔ Remote Lineage

```
Hive: gr_p_select_bulk_eval_input_table
  ├──→ data/local/model/raw/          (extract_raa.py: embeddings + labels)
  │       └──→ data/local/model/split/    (create_split_data.py: 50/50 label split)
  ├──→ data/local/model/enriched/     (extract_prod_predictions.py: CTR sidecars)
  └──→ data/local/model/full_pool/    (FBLearner flow: full 190K pool)

FBLearner: extract_candidates_workflow
  └──→ data/local/eval/bulk_eval/     (top-2K from AI stage)
```

## Context Data (separate from data/)

Per-ad and per-user context are maintained separately — see `architecture.md` for full documentation.

| Module | Path | Purpose | Generated By |
|--------|------|---------|-------------|
| Ads Pool Understanding | `ads_pool/` | Pool catalog, semantic clusters, change log | `ads_pool/refresh.py` |
| User Context | `user/{request_id}/` | Profile, intent, interests, engagement, session | `scripts/prepare_contexts.py` |

## Leakage Prevention Rules

**Critical**: Several tools (Forced Retrieval, anti-negative scorer, cluster engagement) use engagement labels that are also the evaluation target. Without proper controls, this inflates results by 30-50%.

### Rules:
1. **Tools see `history_labels`** — 50% of positive labels, simulating past clicks
2. **Evaluation uses `test_labels`** — the other 50%, simulating future clicks
3. **Never mix** — a tool must never access `test_labels`
4. **Per-stage constraints** (production recall):
   - PM recall: PSelect ordering only (no prior stage)
   - AI recall: PSelect + PM signals (PM is prior stage)
   - AF recall: PSelect + PM + AI signals (all prior stages)

### How to verify:
```bash
# Check no tool imports or accesses test_labels
grep -r "test_labels" tools/  # should return nothing
```

## Usage by Script/Tool

| Script/Tool | Dataset Used | Labels | Enrichment |
|------------|-------------|--------|-----------|
| `run_pilot_diagnosis.py` | `data/local/model/split` | history→tools, test→eval | enriched (if available) |
| `run_baseline_weighted.py` | `data/local/model/split` | history→tools, test→eval | — |
| `run_benchmark_batch.py` | `data/local/model/split` | history→tools, test→eval | — |
| `run_benchmark_fast.py` | `data/local/eval/bulk_eval` | labels | — |
| `evaluate.py` | `data/local/model/raw` or `split` | labels or test_labels | — |
| `evaluate_pipeline.py` | `data/local/model/split` | test_labels | enriched |
| `prod_model_ranker` | — | — | `data/local/model/enriched` |
| `pipeline_simulator` | — | history_labels | `data/local/model/enriched` |
| `ml_reducer` | — | history_labels | `data/local/model/enriched` |
| `hsnn_cluster_scorer` | — | history_labels | — |
| `forced_retrieval` | — | history_labels | — |
| `anti_negative_scorer` | — | history_labels | — |
