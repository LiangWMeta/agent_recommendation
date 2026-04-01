---
name: recommend
description: Run agent recommendation on a batch of ad requests via MCP retrieval tools
---

# Agent Recommendation

Run the ads recommendation agent on a batch of requests. For each request, load context, start the MCP tool server, call retrieval tools following the adaptive strategy, and output ranked ad lists.

## Arguments

- First arg or `--requests N`: Number of requests to process (default: 20)
- `--run-id ID`: Run identifier for output directory (default: `run_YYYYMMDD_HHMMSS`)
- `--data-dir PATH`: Data directory (default: `data/local/model/split`)
- `--request ID`: Process a single specific request by ID

## Workflow

### 1. Setup

```bash
# Find available requests
ls {data_dir}/*.npz | head -{N}
```

Set `run_id` to the provided value or generate one: `run_$(date +%Y%m%d_%H%M%S)`.
Create output directory: `outputs/{run_id}/`.

### 2. For each request

#### a. Load context
Read these files to understand the landscape and user:
- `ads_pool/pool_overview.md` — pool size, categories, recent changes
- `ads_pool/semantic_clusters.md` — HSNN cluster labels (if exists)
- `user/{request_id}/profile.md` — user demographics, embedding summary
- `user/{request_id}/interests.md` — stable interest clusters
- `user/{request_id}/engagement.md` — engagement history
- `user/{request_id}/context.md` — request context

If user context files don't exist, note it and proceed with tool-based analysis only.

#### b. Start MCP server and call tools
The MCP server is at `tools/mcp_server.py`. For each request:

```bash
python3 tools/mcp_server.py --request-npz {data_dir}/request_{request_id}.npz
```

Alternatively, for batch efficiency, use the benchmark script directly:
```bash
python3 scripts/run_baseline_weighted.py --run-id {run_id} --data-dir {data_dir} --max-requests {N}
```

#### c. Follow skill.md reasoning
1. **Step 0**: Load context (already done above)
2. **Step 1**: Call `engagement_pattern_analyzer` — assess signal quality (similarity_gap)
3. **Step 2**: Call `fr_centroid_search` + `lookup_similar_requests`
4. **Step 3**: Multi-route retrieval based on signal quality:
   - Strong (gap > 0.05): embedding primary + fr_centroid + hsnn + prod_model
   - Moderate (0.01-0.05): balanced blend
   - Weak (< 0.01): fr_centroid primary + hsnn expanded + prod_model
5. **Step 4**: Blend with `parallel_routes_blender`, reduce with `ml_reducer`, simulate with `pipeline_simulator`
6. **Step 5**: Output ranked_ads JSON

#### d. Save output
Write to `outputs/{run_id}/{request_id}.json`:
```json
{
  "request_id": 1005207739,
  "ranked_ads": [ad_id_1, ad_id_2, ...],
  "strategy": "Description of strategy used"
}
```

Also write `outputs/{run_id}/meta.json` with run metadata:
```json
{
  "run_id": "run_20260331_214500",
  "data_dir": "data/local/model/split",
  "n_requests": 20,
  "request_ids": [1005207739, 1017453312, ...],
  "timestamp": "2026-03-31T21:45:00"
}
```

### 3. Summary

After all requests, print:
- Requests processed: N
- Run ID: `{run_id}`
- Output directory: `outputs/{run_id}/`
- Suggest: "Run `/analyze --run-id {run_id}` to evaluate results"
