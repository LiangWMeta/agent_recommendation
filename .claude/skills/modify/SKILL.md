---
name: modify
description: Modify the recommendation system — add routes, add data, change algorithms via prompt
---

# Modify

Modify the agent recommendation system based on a free-form prompt. Handles adding new retrieval routes, bringing in new data sources, changing algorithms, and updating strategies.

## Arguments

The entire argument is a natural language prompt describing what to modify.

Examples:
- `/modify add a route that scores ads by creative recency`
- `/modify bring in advertiser budget data from Hive`
- `/modify make ML reducer use prod_prediction as primary signal`
- `/modify prioritize FR centroid more for weak-signal users`

## Workflow

### 1. Classify the modification

Read the user's prompt and determine the type:

| Type | Trigger phrases | What to do |
|------|----------------|------------|
| **Add route** | "add a route", "new retrieval", "new tool" | Create tool, register, update docs |
| **Add data** | "bring in", "add data", "new features", "new signal" | Write extraction, update registry |
| **Change algorithm** | "change", "modify", "make X use Y", "improve" | Modify existing tool code |
| **Change strategy** | "prioritize", "weight", "when to use" | Update skill.md reasoning |

### 2. Execute the modification

#### For "Add Route":

1. **Read existing tool** for pattern reference: `tools/embedding_search.py`
   - Function takes `user_emb, ad_embs, ad_ids, labels, ...` → returns `Dict`
   - All ad_ids as Python ints, scores as Python floats
   - Handle edge cases (empty arrays, 0 positives)

2. **Create new tool** at `tools/{tool_name}.py`
   - Follow the exact same function signature pattern
   - Include the new scoring/retrieval logic

3. **Register in `tools/tool_registry.py`**:
   - Add entry to `TOOLS` list with `name`, `description`, `input_schema`
   - Add dispatch case in `execute_tool()` function
   - Import at top of file

4. **Register in `tools/mcp_server.py`**:
   - Import at top
   - Add tool schema to `handle_tools_list()` tools list
   - Add dispatch case to `handle_tool_call()`

5. **Update documentation**:
   - `architecture.md`: Add to Production or Exploration track table
   - `skill.md`: Add when/how to use the new tool
   - `CLAUDE.md`: Add to Available Tools table

6. **Suggest**: "New route added. Run `/analyze` to measure its impact."

#### For "Add Data":

1. **Understand the data source** — ask for Hive table path or describe the schema
2. **Write extraction script** in `scripts/` (follow `extract_prod_predictions.py` pattern)
3. **Update `data/datasets.md`**:
   - Add to Remote Datasets table
   - Add to Local ↔ Remote Lineage
   - Add loading code example
4. **Update context files** if applicable:
   - `ads_pool/catalog.md` schema if it's per-ad data
   - `user/` context files if it's per-user data
5. **Update tools** that should use the new data

#### For "Change Algorithm":

1. **Read the current tool** implementation
2. **Modify the algorithm** as requested
3. **Suggest**: "Algorithm changed. Run `/analyze` to compare against previous results."

#### For "Change Strategy":

1. **Read `skill.md`** current strategy
2. **Modify the relevant section** (signal thresholds, route weights, tool ordering)
3. **Update `CLAUDE.md`** if the Recommended Strategy section needs updating

### 3. Verify

After any modification:
- Run a quick smoke test if code was changed:
  ```bash
  python3 -c "import sys; sys.path.insert(0, '.'); from tools.{tool_name} import {func_name}; print('Import OK')"
  ```
- Suggest running `/analyze` to measure impact
