---
name: recommend
description: Run agent recommendation on a batch of ad requests via MCP retrieval tools
---

# Agent Recommendation

Run the ads recommendation agent on a batch of requests. Pre-computes all tool results in Python (fast), then spawns parallel Agent subagents to reason over results and produce ranked ad lists.

## Arguments

- First arg or `--requests N`: Number of requests to process (default: 20)
- `--run-id ID`: Run identifier for output directory (default: `run_YYYYMMDD_HHMMSS`)
- `--data-dir PATH`: Data directory (default: `data/local/model/split`)
- `--request ID`: Process a single specific request by ID

## Workflow

### 1. Pre-compute tool results

Run the precompute script to call all retrieval tools via Python (no LLM needed, ~3s/request):

```bash
python3 scripts/precompute_tool_results.py \
  --data-dir {data_dir} \
  --output-dir /tmp/tool_results \
  --max-requests {N}
```

This produces `/tmp/tool_results/{request_id}.md` for each request and a `manifest.json`.

### 2. Setup output directory

Set `run_id` to the provided value or generate one: `run_$(date +%Y%m%d_%H%M%S)`.

```bash
mkdir -p outputs/{run_id}
```

### 3. Read manifest

Read `/tmp/tool_results/manifest.json` to get the list of request_ids to process.

### 4. Spawn agents in waves

For each wave of up to 5 requests, spawn Agent subagents **in parallel** using `run_in_background: true`. Send all agents in a single message to maximize parallelism.

**Agent prompt for each request:**

```
You are an ads recommendation agent. Your job is to reason about retrieval tool results and produce a ranked list of ad IDs.

Read the pre-computed tool results at /tmp/tool_results/{request_id}.md.
Also read user context if available at user/{request_id}/ (profile.md, engagement.md, interest_clusters.md, context.md).

Follow this reasoning framework:
1. Check similarity_gap from engagement_pattern_analyzer:
   - > 0.05: Strong signal — trust pselect_main_route as primary
   - 0.01-0.05: Moderate — blend pselect and forced_retrieval equally
   - < 0.01: Weak — forced_retrieval is primary, don't trust pselect

2. Assess route reliability:
   - forced_retrieval centroid_gap vs user_emb_gap — which query vector is stronger?
   - centroid_vs_user_correlation — low means independent signals (good for diversity)

3. Look for consensus ads appearing in multiple routes (high confidence).

4. Check cluster patterns:
   - High engagement_rate clusters deserve more candidates
   - If a cluster has high engagement but low route coverage, those ads are being missed

5. Check prod_model_ranker if available — strongest per-ad quality signal.

6. Merge all route results. Prioritize:
   - Consensus ads (4+ routes) first
   - Forced retrieval unique finds (historically most valuable)
   - High-engagement cluster ads
   - Prod model top ads
   - PSelect/anti-negative ads to fill

Write your output as a JSON file at outputs/{run_id}/{request_id}.json:
```json
{
  "request_id": {request_id},
  "ranked_ads": [ad_id_1, ad_id_2, ...],
  "strategy": "Brief description of your reasoning"
}
```

Include 150+ ranked ad IDs. The ranking should reflect your best judgment of engagement likelihood.
```

**Important agent settings:**
- `run_in_background: true` — allows parallel execution
- `mode: "auto"` — agents can read files and write output without prompts

### 5. Wait for agents and collect results

After each wave completes (you'll be notified), check that output files were written:
```bash
ls outputs/{run_id}/*.json | wc -l
```

Then spawn the next wave. Continue until all requests are processed.

### 6. Write metadata

After all requests complete, write `outputs/{run_id}/meta.json`:
```json
{
  "run_id": "{run_id}",
  "data_dir": "{data_dir}",
  "n_requests": N,
  "request_ids": [...],
  "timestamp": "..."
}
```

### 7. Summary

Print:
- Requests processed: N
- Run ID: `{run_id}`
- Output directory: `outputs/{run_id}/`
- Suggest: "Run `/analyze --run-id {run_id}` to evaluate results"

## Single Request Mode

For `--request ID`, skip pre-compute and directly:
1. Read user context from `user/{request_id}/`
2. Start MCP server: `python3 tools/mcp_server.py --request-npz {data_dir}/request_{id}.npz`
3. Call tools interactively following skill.md reasoning
4. Output ranked_ads JSON
