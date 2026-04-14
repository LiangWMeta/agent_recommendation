---
name: data-setup
description: Pull and prepare all data needed to run the agent recommendation system
---

# Data Setup

Prepare all data needed to run the agent. Uses `scripts/setup_data.py` which orchestrates the full pipeline: check existing data, build missing stages, report readiness.

## Usage

### Check status only (no changes)

```bash
python3 scripts/setup_data.py --check-only
```

### Build missing data (default)

```bash
python3 scripts/setup_data.py --max-requests 100
```

### Force rebuild everything

```bash
python3 scripts/setup_data.py --force --max-requests 100
```

## What it does

The setup script checks each stage and only builds what's missing or stale:

1. **Raw data** (`data/local/model/raw/`) — extracted from Hive via `extract_raa.py`
2. **Train/test split** (`data/local/model/split/`) — 50/50 label split for leakage prevention
3. **Prod predictions** (`data/local/model/enriched/`) — optional, SlimDSNN calibrated CTR
4. **User contexts** (`user/`) — per-request profile, engagement, interests, context
5. **Ads pool understanding** (`ads_pool/`) — pool overview, catalog, semantic clusters

Staleness detection: if source data is newer than derived data, the stage is rebuilt.

## After setup

Report readiness and suggest next steps:
- "Data is ready. Run `/recommend` to start recommending, or `/analyze` to analyze system behavior."
