---
name: data-setup
description: Pull and prepare all data needed to run the agent recommendation system
---

# Data Setup

Prepare all data needed to run the agent. Follows the quickstart chain from `data/datasets.md`: extract source data → create train/test split → extract prod predictions → generate user context → generate ads pool understanding.

## Arguments

- `--max-requests N`: Maximum requests to process (default: 100)
- `--skip-existing`: Skip steps where data already exists

## Workflow

### 1. Check existing data

```bash
echo "=== Checking data ==="
echo "raw:       $(ls data/local/model/raw/*.npz 2>/dev/null | wc -l) files"
echo "split:     $(ls data/local/model/split/*.npz 2>/dev/null | wc -l) files"
echo "enriched:  $(ls data/local/model/enriched/*.json 2>/dev/null | wc -l) files"
echo "full_pool: $(ls data/local/model/full_pool/*.npz 2>/dev/null | wc -l) files"
echo "bulk_eval: $(ls data/local/eval/bulk_eval/*.npz 2>/dev/null | wc -l) files"
echo "user ctx:  $(ls -d user/*/ 2>/dev/null | wc -l) folders"
```

### 2. Extract RAA source data (if raw/ is empty)

```bash
python3 scripts/extract_raa.py \
  --output-dir data/local/model/raw \
  --max-requests {max_requests}
```

If `extract_raa.py` doesn't exist or raw/ already has data, skip this step and note it.

### 3. Create train/test split

```bash
python3 scripts/create_split_data.py \
  --data-dir data/local/model/raw \
  --output-dir data/local/model/split \
  --max-requests {max_requests}
```

### 4. Extract production predictions (optional but recommended)

```bash
python3 scripts/extract_prod_predictions.py \
  --data-dir data/local/model/raw \
  --output-dir data/local/model/enriched
```

Note: This requires Hive/Presto access. If it fails, proceed without — tools will use cosine fallback.

### 5. Generate user context folders

```bash
python3 scripts/prepare_contexts.py \
  --data-dir data/local/model/split \
  --output-dir user/
```

### 6. Generate ads pool understanding

If `ads_pool/refresh.py` exists:
```bash
python3 ads_pool/refresh.py --data-dir data/local/model/split
```

Otherwise, note that ads_pool context files need to be created manually or the refresh script needs to be built.

### 7. Report

Print summary of what was created:
- Raw data: N requests
- Split data: N requests (with history/test labels)
- Enriched: N requests with prod predictions
- User contexts: N folders
- Ads pool: status

Suggest: "Data is ready. Run `/recommend` to start recommending, or `/analyze` to analyze system behavior."
