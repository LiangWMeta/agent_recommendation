#!/usr/bin/env python3
"""
Run production-aligned recall pipeline for agent recommendation.

Generates the same blending recall SQL as production PSelect/GR workflows,
with agent_score as the extra_route_total_value. Executes via Presto.

Usage:
    python3 scripts/run_recall_pipeline.py \\
      --run-id cc_9sample --ds 2026-03-19 \\
      --recall-type pm_to_ai_recall --page-type 19 \\
      --proportions 0.0 0.1 0.2 0.5 1.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import evaluation.prod_recall
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.prod_recall import (
    PAGE_TYPE_CONFIG_MAP,
    RecallResult,
    format_results_table,
    parse_recall_result,
)


def run_presto_query(query, namespace="ad_delivery", limit=10000, retries=2):
    """Execute a Presto query via jf graphql."""
    # Escape double quotes in the query for embedding in GraphQL
    escaped_query = query.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    graphql_query = (
        '{ xfb_presto_tools { execute_query(input: {'
        f'query: "{escaped_query}", '
        f'namespace: "{namespace}", '
        f'limit: {limit}'
        '}) { success data_table_json row_count columns_json error } } }'
    )
    cmd = ["jf", "graphql", "--query", graphql_query]

    for attempt in range(retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"  Query timed out (attempt {attempt+1})", file=sys.stderr)
            if attempt < retries:
                time.sleep(5)
                continue
            return None

        if result.returncode != 0:
            print(f"  Query failed (rc={result.returncode}): {result.stderr[:300]}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5)
                continue
            return None

        try:
            response = json.loads(result.stdout)
            data = response["xfb_presto_tools"]["execute_query"]
            if not data["success"]:
                print(f"  Query error: {data.get('error', 'unknown')[:300]}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(5)
                    continue
                return None
            if data["data_table_json"]:
                return json.loads(data["data_table_json"])
            return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5)
                continue
            return None
    return None


def load_agent_scores(run_id, outputs_dir="outputs"):
    """Load agent scores from benchmark output files."""
    scores = {}
    run_dir = Path(outputs_dir) / run_id
    if not run_dir.exists():
        return scores
    for f in run_dir.glob("*.json"):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
        ranked_ads = data.get("ranked_ads", [])
        request_id = data.get("request_id")
        if not ranked_ads or request_id is None:
            continue
        N = len(ranked_ads)
        for rank, ad_id in enumerate(ranked_ads):
            scores[(int(request_id), int(ad_id))] = (N - rank) / N
    return scores


def build_agent_scores_cte(agent_scores):
    """Build an inline CTE with agent scores using VALUES clause."""
    if not agent_scores:
        return "agent_scores AS (SELECT CAST(0 AS BIGINT) AS request_id, CAST(0 AS BIGINT) AS ad_id, CAST(0.0 AS DOUBLE) AS agent_score WHERE 1=0)"
    values = []
    for (req_id, ad_id), score in agent_scores.items():
        values.append(f"({req_id}, {ad_id}, {score:.6f})")
    values_str = ", ".join(values)
    return f"agent_scores AS (SELECT request_id, ad_id, agent_score FROM (VALUES {values_str}) AS t(request_id, ad_id, agent_score))"


def generate_blending_recall_query(
    ds,
    page_type,
    proportion,
    agent_scores_cte,
    raa_table="prematch_ctr_consideration_data_for_recall",
    raa_pipeline="ar_dwnn_airaa_mobile_feed_noprod",
    num_ads_fm_matched=11119,
    num_ads_ai_ranked=4189,
    num_ads_ai_returned=709,
    **kwargs,
):
    """Generate the full production-aligned PM-to-AI blending recall SQL.

    Uses inline CTE for agent scores (no external table needed).
    The blending logic, caps, dedup, and recall computation are identical
    to production PSelect/GR workflows.
    """
    query = f"""
WITH {agent_scores_cte},
all_candidates_read AS (
    SELECT
        ac.separable_id,
        ac.timestamp,
        ac.request_id AS request_id,
        ac.page_type,
        ac.ad_id,
        CAST(ac.float_features[584] AS INTEGER) AS conv_type,
        ac.id_list_features[1][1] AS account_id,
        ac.id_list_features[2][1] AS campaign_id,
        COALESCE(ac.id_list_features[9], ARRAY[0]) AS fbobj_ids,
        COALESCE(CAST(ac.float_features[5534] AS DOUBLE), 0) AS price_floor,
        CAST(ac.float_features[4029] AS BIGINT) AS ee_key,
        (COALESCE(ac.float_features[4012881], 0) = 1) AS is_dco_ad,
        ac.float_features[3281] AS ai_total_value,
        ac.float_features[3281] AS ai_ads_value,
        ac.float_features[5166] AS pm_total_value,
        COALESCE(agent.agent_score, 0) AS extra_route_total_value,
        ac.float_features[5166] AS pm_ads_value,
        0.0 AS pm_quality_value,
        0.0 AS pm_ectr,
        (ac.float_features[6278] = 1) AS is_piggyback,
        (ac.float_features[6317] = 1) AS is_related_ads,
        (ac.float_features[6318] = 1) AS is_forced_retrieval,
        (CASE WHEN ac.page_type = 19 THEN 9 WHEN ac.page_type IN (35, 49) THEN 18 ELSE 20 END) AS pm_acct_conv_cap_threshold,
        ac.ds
    FROM {raa_table} ac
    LEFT JOIN agent_scores agent
        ON ac.request_id = agent.request_id
        AND ac.ad_id = agent.ad_id
    WHERE
        ac.ds = '{ds}'
        AND ac.pipeline = '{raa_pipeline}'
        AND ac.page_type IN ({page_type})
        AND ac.float_features[7444] = 1
),
adjusted_scores AS (
    SELECT
        *,
        IF(main_route_rank <= {num_ads_fm_matched} * (1.0 - {proportion}), -1, extra_route_total_value)
            AS adjusted_extra_route_total_value
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp ORDER BY pm_total_value DESC) AS main_route_rank,
            MAX(pm_total_value) OVER (PARTITION BY request_id, separable_id, timestamp) AS max_pm_total_value
        FROM all_candidates_read
    )
),
pm AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp ORDER BY modified_pm_total_value DESC) AS blend_rank
    FROM (
        SELECT *,
            (CASE
                WHEN extra_route_rank <= {num_ads_fm_matched} * {proportion} THEN max_pm_total_value + adjusted_extra_route_total_value
                ELSE pm_total_value
            END) AS modified_pm_total_value
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp ORDER BY adjusted_extra_route_total_value DESC) AS extra_route_rank
            FROM adjusted_scores
        )
    )
),
fm AS (SELECT * FROM pm WHERE blend_rank <= {num_ads_fm_matched}),
post_fm_ee_dedup AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, ee_key ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_ee_dedup_rank
        FROM fm
    ) WHERE post_fm_ee_dedup_rank = 1
),
post_fm_dco_campaign_cap AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, campaign_id, is_dco_ad ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_dco_campaign_cap_rank
        FROM post_fm_ee_dedup
    ) WHERE NOT is_dco_ad OR post_fm_dco_campaign_cap_rank <= 6
),
post_fm_campaign_cap AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, campaign_id ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_campaign_cap_rank
        FROM post_fm_dco_campaign_cap
    ) WHERE post_fm_campaign_cap_rank <= 9
),
post_fm_account_conv_cap AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, account_id, conv_type ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_account_conv_cap_rank
        FROM post_fm_campaign_cap
    ) WHERE post_fm_account_conv_cap_rank <= pm_acct_conv_cap_threshold
),
post_fm_cap_trunc_dedup_dummy_pm AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp ORDER BY is_piggyback DESC, pm_total_value DESC) AS pm_rank_final
    FROM post_fm_account_conv_cap
),
ai AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp ORDER BY ai_total_value DESC) AS ai_rank
        FROM (SELECT * FROM post_fm_cap_trunc_dedup_dummy_pm WHERE pm_rank_final <= {num_ads_ai_ranked})
    ) WHERE ai_rank <= {num_ads_ai_returned}
),
raa_post_fm_ee_dedup AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, ee_key ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_ee_dedup_rank
        FROM pm
    ) WHERE post_fm_ee_dedup_rank = 1
),
raa_post_fm_dco_campaign_cap AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, campaign_id, is_dco_ad ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_dco_campaign_cap_rank
        FROM raa_post_fm_ee_dedup
    ) WHERE NOT is_dco_ad OR post_fm_dco_campaign_cap_rank <= 6
),
raa_post_fm_campaign_cap AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, campaign_id ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_campaign_cap_rank
        FROM raa_post_fm_dco_campaign_cap
    ) WHERE post_fm_campaign_cap_rank <= 9
),
raa_post_fm_account_conv_cap AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp, account_id, conv_type ORDER BY is_piggyback DESC, pm_total_value DESC) AS post_fm_account_conv_cap_rank
        FROM raa_post_fm_campaign_cap
    ) WHERE post_fm_account_conv_cap_rank <= pm_acct_conv_cap_threshold
),
raa_ai AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, separable_id, timestamp ORDER BY ai_total_value DESC) AS ai_rank
        FROM raa_post_fm_account_conv_cap
    ) WHERE ai_rank <= {num_ads_ai_returned}
),
recall_per_request_unordered AS (
    SELECT
        (CASE WHEN raa.ad_id IS NOT NULL THEN raa.ds ELSE prod.ds END) AS ds,
        (CASE WHEN raa.ad_id IS NOT NULL THEN raa.request_id ELSE prod.request_id END) AS request_id,
        (CASE WHEN raa.ad_id IS NOT NULL THEN raa.separable_id ELSE prod.separable_id END) AS separable_id,
        (CASE WHEN raa.ad_id IS NOT NULL THEN raa.timestamp ELSE prod.timestamp END) AS timestamp,
        CAST(COUNT(CASE WHEN raa.ad_id IS NOT NULL THEN raa.ad_id END) AS INTEGER) AS scorer_topk_ad_cnt,
        CAST(COUNT(CASE WHEN raa.ad_id IS NOT NULL AND prod.ad_id IS NOT NULL THEN prod.ad_id END) AS INTEGER) AS replayed_scorer_topk_recalled_ad_cnt,
        SUM(raa.ai_total_value) AS scorer_topk_total_value,
        SUM(prod.ai_total_value) AS replayed_scorer_topk_total_value
    FROM raa_ai AS raa
    FULL OUTER JOIN ai AS prod
        ON raa.request_id = prod.request_id AND raa.separable_id = prod.separable_id AND raa.timestamp = prod.timestamp AND raa.ad_id = prod.ad_id
    GROUP BY 1, 2, 3, 4
),
winsorization AS (
    SELECT
        ds,
        APPROX_PERCENTILE(replayed_scorer_topk_total_value, 0.99) AS p99_replayed,
        APPROX_PERCENTILE(scorer_topk_total_value, 0.99) AS p99_scorer,
        APPROX_PERCENTILE(replayed_scorer_topk_total_value, 0.01) AS p1_replayed,
        APPROX_PERCENTILE(scorer_topk_total_value, 0.01) AS p1_scorer
    FROM recall_per_request_unordered GROUP BY 1
)
SELECT
    'agent_recommendation' AS raa,
    SET_AGG(recall_per_request_unordered.ds) AS ds,
    SUM(replayed_scorer_topk_total_value) / SUM(scorer_topk_total_value) AS soft_recall,
    SUM(CASE WHEN replayed_scorer_topk_total_value > p99_replayed THEN p99_replayed WHEN replayed_scorer_topk_total_value < p1_replayed THEN p1_replayed ELSE replayed_scorer_topk_total_value END) /
    SUM(CASE WHEN scorer_topk_total_value > p99_scorer THEN p99_scorer WHEN scorer_topk_total_value < p1_scorer THEN p1_scorer ELSE scorer_topk_total_value END) AS winsorized_soft_recall,
    SUM(replayed_scorer_topk_recalled_ad_cnt) * 1.0 / SUM(scorer_topk_ad_cnt) AS hard_recall
FROM recall_per_request_unordered
LEFT JOIN winsorization ON recall_per_request_unordered.ds = winsorization.ds
"""
    return query.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run production-aligned recall pipeline for agent recommendation"
    )
    parser.add_argument("--run-id", required=True, help="Agent run ID (e.g. cc_9sample)")
    parser.add_argument("--ds", required=True, help="RAA data date (e.g. 2026-03-19)")
    parser.add_argument("--recall-type", default="pm_to_ai_recall",
                        help="Recall type (default: pm_to_ai_recall)")
    parser.add_argument("--page-type", type=int, default=19, help="Page type (default: 19)")
    parser.add_argument("--proportions", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.5, 1.0],
                        help="Traffic proportions to evaluate")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory with agent benchmark outputs")
    parser.add_argument("--namespace", default="ad_delivery",
                        help="Presto namespace for RAA table")
    parser.add_argument("--raa-table", default="prematch_ctr_consideration_data_for_recall",
                        help="RAA table name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated SQL without executing")
    args = parser.parse_args()

    # Get page type config
    config = PAGE_TYPE_CONFIG_MAP.get(args.page_type)
    if config is None:
        print(f"Error: unsupported page_type {args.page_type}. "
              f"Supported: {list(PAGE_TYPE_CONFIG_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    # Load agent scores and build inline CTE
    print(f"Loading agent scores from outputs/{args.run_id}/...")
    agent_scores = load_agent_scores(args.run_id, args.output_dir)
    print(f"  Loaded {len(agent_scores)} (request_id, ad_id) scores")
    agent_cte = build_agent_scores_cte(agent_scores)
    print(f"  CTE size: {len(agent_cte)} chars")

    raa_pipeline = {
        19: "ar_dwnn_airaa_mobile_feed_noprod",
        103: "ar_dwnn_airaa_mobile_reels_noprod",
        35: "ar_dwnn_airaa_ig_stream_noprod",
        49: "ar_dwnn_airaa_ig_story_noprod",
    }.get(args.page_type, "ar_dwnn_airaa_mobile_feed_noprod")

    print(f"\nRunning {args.recall_type} pipeline")
    print(f"  run_id:      {args.run_id}")
    print(f"  ds:          {args.ds}")
    print(f"  page_type:   {args.page_type}")
    print(f"  proportions: {args.proportions}")
    print(f"  RAA table:   {args.raa_table}")
    print(f"  RAA pipeline: {raa_pipeline}")
    print(f"  Agent scores: {len(agent_scores)} inline")
    print()

    results = {}
    results_dir = Path(__file__).resolve().parent.parent / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for proportion in args.proportions:
        print(f"--- Proportion {proportion:.2f} ---")

        sql = generate_blending_recall_query(
            ds=args.ds,
            page_type=args.page_type,
            proportion=proportion,
            agent_scores_cte=agent_cte,
            raa_table=args.raa_table,
            raa_pipeline=raa_pipeline,
            num_ads_fm_matched=config["num_ads_fm_matched"],
            num_ads_ai_ranked=config["num_ads_ai_ranked"],
            num_ads_ai_returned=config["num_ads_ai_returned"],
        )

        if args.dry_run:
            print(sql)
            print()
            continue

        print("  Executing query...")
        rows = run_presto_query(sql, namespace=args.namespace, limit=100)

        if rows is None:
            print("  ERROR: query returned None", file=sys.stderr)
            continue

        if not rows:
            print("  WARNING: query returned empty result", file=sys.stderr)
            continue

        row = rows[0]
        recall = parse_recall_result(row, proportion=proportion)
        results[proportion] = recall

        print(f"  soft_recall:       {recall.soft_recall:.6f}")
        print(f"  hard_recall:       {recall.hard_recall:.6f}")
        print(f"  winsorized_soft:   {recall.winsorized_soft_recall:.6f}")
        print()

        time.sleep(1)  # Rate limiting between queries

    if args.dry_run:
        return

    if not results:
        print("No results obtained. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    print("\n=== Recall Results ===")
    print(format_results_table(results))

    # Save results
    output_path = results_dir / f"{args.run_id}_prod_recall.json"
    output_data = {
        "run_id": args.run_id,
        "ds": args.ds,
        "page_type": args.page_type,
        "recall_type": args.recall_type,
        "agent_namespace": args.agent_namespace,
        "agent_table": args.agent_table,
        "results": {
            str(proportion): results[proportion].to_dict()
            for proportion in sorted(results.keys())
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
