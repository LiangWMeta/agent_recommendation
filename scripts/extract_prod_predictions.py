#!/usr/bin/env python3
"""
Extract production signals from Hive for our candidate ads.

Three data sources (in priority order):
1. fct_raa_consideration_data — PM/AI/AF total_value, FR flags (~60-65% coverage)
2. user_ads_ranked_ecpm_daily — aggregated eCPM across users (~66% coverage)
3. gr_p_select_bulk_eval_input_table — prod_prediction (pCTR), ~60% coverage

Usage:
    python3 scripts/extract_prod_predictions.py --data-dir data/local/model/split --max-requests 10
    python3 scripts/extract_prod_predictions.py --data-dir data/local/model/split  # all requests
"""

import argparse
import json
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

NAMESPACE = "ad_delivery"
RAA_TABLE = "fct_raa_consideration_data"
RAA_TIER = "ONLINE_AF_RAA_PM"
ECPM_TABLE = "user_ads_ranked_ecpm_daily"
BULK_EVAL_TABLE = "gr_p_select_bulk_eval_input_table"
DS = "2026-03-22"
BATCH_SIZE = 500


def run_presto_query(query, limit=10000, retries=2):
    """Execute a Presto query via jf graphql."""
    graphql_query = (
        '{ xfb_presto_tools { execute_query(input: {'
        f'query: "{query}", '
        f'namespace: "{NAMESPACE}", '
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


def extract_raa(ad_ids):
    """Extract PM/AI total_value and FR flags from fct_raa_consideration_data.

    Returns aggregated median PM total_value per ad across all user contexts.
    This is the best eCPM signal available (~60-65% coverage).
    """
    results = {}
    ad_id_list = [int(x) for x in ad_ids]

    for batch_start in range(0, len(ad_id_list), BATCH_SIZE):
        batch = ad_id_list[batch_start:batch_start + BATCH_SIZE]
        id_str = ", ".join(str(x) for x in batch)

        query = (
            f"SELECT ad_id, "
            f"AVG(pm_stage_struct.total_value) AS avg_pm_tv, "
            f"APPROX_PERCENTILE(pm_stage_struct.total_value, 0.5) AS median_pm_tv, "
            f"AVG(ai_stage_struct.total_value) AS avg_ai_tv, "
            f"APPROX_PERCENTILE(ai_stage_struct.total_value, 0.5) AS median_ai_tv, "
            f"BOOL_OR(is_forced_retrieval_ads) AS is_forced_retrieval, "
            f"BOOL_OR(is_piggyback_ad) AS is_piggyback, "
            f"COUNT(*) AS n_obs "
            f"FROM {RAA_TABLE} "
            f"WHERE ds = '{DS}' "
            f"AND consideration_data_tier = '{RAA_TIER}' "
            f"AND ad_id IN ({id_str}) "
            f"GROUP BY ad_id"
        )

        rows = run_presto_query(query, limit=len(batch) + 100)
        if rows:
            for row in rows:
                aid = int(row["ad_id"])
                entry = {}
                if row.get("median_pm_tv") is not None:
                    entry["median_pm_tv"] = float(row["median_pm_tv"])
                if row.get("avg_pm_tv") is not None:
                    entry["avg_pm_tv"] = float(row["avg_pm_tv"])
                if row.get("median_ai_tv") is not None:
                    entry["median_ai_tv"] = float(row["median_ai_tv"])
                if row.get("avg_ai_tv") is not None:
                    entry["avg_ai_tv"] = float(row["avg_ai_tv"])
                entry["is_forced_retrieval"] = bool(row.get("is_forced_retrieval", False))
                entry["is_piggyback"] = bool(row.get("is_piggyback", False))
                entry["raa_n_obs"] = int(row.get("n_obs", 0))
                results[aid] = entry

        time.sleep(0.5)

    return results


def extract_ecpm(ad_ids):
    """Extract aggregated eCPM from user_ads_ranked_ecpm_daily (~66% coverage)."""
    results = {}
    ad_id_list = [int(x) for x in ad_ids]

    for batch_start in range(0, len(ad_id_list), BATCH_SIZE):
        batch = ad_id_list[batch_start:batch_start + BATCH_SIZE]
        id_str = ", ".join(str(x) for x in batch)

        query = (
            f"SELECT ad_id, "
            f"AVG(ar_ranking_ecpm) AS avg_ecpm, "
            f"APPROX_PERCENTILE(ar_ranking_ecpm, 0.5) AS median_ecpm, "
            f"COUNT(*) AS n_obs "
            f"FROM {ECPM_TABLE} "
            f"WHERE ds = '{DS}' "
            f"AND ad_id IN ({id_str}) "
            f"GROUP BY ad_id"
        )

        rows = run_presto_query(query, limit=len(batch) + 100)
        if rows:
            for row in rows:
                aid = int(row["ad_id"])
                results[aid] = {
                    "median_ecpm": float(row["median_ecpm"]),
                    "avg_ecpm": float(row["avg_ecpm"]),
                    "ecpm_n_obs": int(row["n_obs"]),
                }

        time.sleep(0.5)

    return results


def extract_bulk_eval(ad_ids):
    """Extract prod_prediction from bulk eval input table (~60% coverage)."""
    results = {}
    ad_id_list = [int(x) for x in ad_ids]

    for batch_start in range(0, len(ad_id_list), BATCH_SIZE):
        batch = ad_id_list[batch_start:batch_start + BATCH_SIZE]
        id_str = ", ".join(str(x) for x in batch)

        query = (
            f"SELECT ad_id, prod_prediction "
            f"FROM {BULK_EVAL_TABLE} "
            f"WHERE ds = '{DS}' "
            f"AND ad_id IN ({id_str})"
        )

        rows = run_presto_query(query, limit=len(batch) + 100)
        if rows:
            for row in rows:
                aid = int(row["ad_id"])
                if row.get("prod_prediction") is not None:
                    results[aid] = {"prod_prediction": float(row["prod_prediction"])}

        time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract production signals from Hive")
    parser.add_argument("--data-dir", default="data/local/model/split", help="Directory with npz files")
    parser.add_argument("--output-dir", default="data/local/model/enriched", help="Output directory")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip requests with existing output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    npz_files = sorted(data_dir.glob("request_*.npz"))
    if args.max_requests:
        npz_files = npz_files[:args.max_requests]

    print(f"Extracting production signals for {len(npz_files)} requests...")
    print(f"  RAA table: {RAA_TABLE} (tier={RAA_TIER}), ds={DS}")
    print(f"  eCPM table: {ECPM_TABLE}, ds={DS}")
    print(f"  Bulk eval: {BULK_EVAL_TABLE}, ds={DS}")

    total_ads = 0
    raa_hits = 0
    ecpm_hits = 0
    bulk_hits = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        request_id = int(data["request_id"])
        ad_ids = data["ad_ids"]

        out_path = output_dir / f"{request_id}_prod.json"
        if args.skip_existing and out_path.exists():
            print(f"  [{i+1}/{len(npz_files)}] {request_id}: skipped (exists)")
            continue

        print(f"  [{i+1}/{len(npz_files)}] {request_id}: {len(ad_ids)} ads...", end=" ", flush=True)

        # Source 1: RAA consideration data (PM/AI total_value, FR flags)
        raa_data = extract_raa(ad_ids)

        # Source 2: aggregated eCPM
        ecpm_data = extract_ecpm(ad_ids)

        # Source 3: bulk eval (prod_prediction / pCTR)
        bulk_data = extract_bulk_eval(ad_ids)

        # Merge all sources per ad
        result = []
        for aid in ad_ids:
            entry = {"ad_id": int(aid)}
            # Layer sources: bulk_eval first, then ecpm, then raa (highest priority last)
            bulk = bulk_data.get(int(aid))
            if bulk:
                entry.update(bulk)
            ecpm = ecpm_data.get(int(aid))
            if ecpm:
                entry.update(ecpm)
            raa = raa_data.get(int(aid))
            if raa:
                entry.update(raa)
            result.append(entry)

        with open(out_path, "w") as f:
            json.dump(result, f)

        n_raa = len(raa_data)
        n_ecpm = len(ecpm_data)
        n_bulk = len(bulk_data)
        total_ads += len(ad_ids)
        raa_hits += n_raa
        ecpm_hits += n_ecpm
        bulk_hits += n_bulk
        print(f"raa={n_raa}/{len(ad_ids)} ({n_raa/len(ad_ids):.0%}), "
              f"ecpm={n_ecpm}/{len(ad_ids)} ({n_ecpm/len(ad_ids):.0%}), "
              f"bulk={n_bulk}/{len(ad_ids)} ({n_bulk/len(ad_ids):.0%})")

    print(f"\nDone!")
    print(f"  RAA coverage: {raa_hits}/{total_ads} ({raa_hits/total_ads:.0%})")
    print(f"  eCPM coverage: {ecpm_hits}/{total_ads} ({ecpm_hits/total_ads:.0%})")
    print(f"  Bulk eval coverage: {bulk_hits}/{total_ads} ({bulk_hits/total_ads:.0%})")
    print(f"Output saved to {output_dir}/")


if __name__ == "__main__":
    main()
