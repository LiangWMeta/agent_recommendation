#!/usr/bin/env python3
"""
Extract production signals from Hive for our candidate ads.

Queries gr_p_select_bulk_eval_input_table to get:
- prod_prediction: calibrated CTR from SlimDSNN
- pm_total_value: full PM eCPM score (pCTR × pCVR × bid)
- ai_total_value: full AI eCPM score
- is_forced_retrieval: FR flag
- is_piggyback: piggyback flag

Usage:
    python3 scripts/extract_prod_predictions.py --max-requests 10
    python3 scripts/extract_prod_predictions.py  # all 306 requests
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

NAMESPACE = "ad_delivery"
TABLE = "gr_p_select_bulk_eval_input_table"
DS = "2026-03-22"
OUTPUT_DIR = Path("data/local/model/enriched")
BATCH_SIZE = 500  # ad_ids per query


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


def extract_for_request(request_id, ad_ids):
    """Extract prod_prediction for a set of ad_ids."""
    all_results = {}

    # Batch ad_ids to avoid query size limits
    ad_id_list = [int(x) for x in ad_ids]
    for batch_start in range(0, len(ad_id_list), BATCH_SIZE):
        batch = ad_id_list[batch_start:batch_start + BATCH_SIZE]
        id_str = ", ".join(str(x) for x in batch)

        query = (
            f"SELECT ad_id, prod_prediction, "
            f"float_features[5166] AS pm_total_value, "
            f"float_features[3281] AS ai_total_value, "
            f"(float_features[6318] = 1) AS is_forced_retrieval, "
            f"(float_features[6278] = 1) AS is_piggyback "
            f"FROM {TABLE} "
            f"WHERE ds = '{DS}' "
            f"AND ad_id IN ({id_str})"
        )

        rows = run_presto_query(query, limit=len(batch) + 100)
        if rows:
            for row in rows:
                aid = int(row["ad_id"])
                entry = {}
                if row.get("prod_prediction") is not None:
                    entry["prod_prediction"] = float(row["prod_prediction"])
                if row.get("pm_total_value") is not None:
                    entry["pm_total_value"] = float(row["pm_total_value"])
                if row.get("ai_total_value") is not None:
                    entry["ai_total_value"] = float(row["ai_total_value"])
                entry["is_forced_retrieval"] = bool(row.get("is_forced_retrieval", False))
                entry["is_piggyback"] = bool(row.get("is_piggyback", False))
                if entry:
                    all_results[aid] = entry

        time.sleep(0.5)  # Rate limiting

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Extract prod_prediction from Hive")
    parser.add_argument("--data-dir", default="data/local/model/raw", help="Directory with npz files")
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

    print(f"Extracting prod_prediction for {len(npz_files)} requests...")
    print(f"Table: {TABLE}, ds={DS}")

    total_ads = 0
    total_hits = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        request_id = int(data["request_id"])
        ad_ids = data["ad_ids"]

        out_path = output_dir / f"{request_id}_prod.json"
        if args.skip_existing and out_path.exists():
            print(f"  [{i+1}/{len(npz_files)}] {request_id}: skipped (exists)")
            continue

        print(f"  [{i+1}/{len(npz_files)}] {request_id}: {len(ad_ids)} ads...", end=" ")

        predictions = extract_for_request(request_id, ad_ids)

        # Save as JSON — each entry has ad_id + all available signals
        result = []
        for aid in ad_ids:
            entry = {"ad_id": int(aid)}
            signals = predictions.get(int(aid), {})
            entry.update(signals)
            result.append(entry)

        with open(out_path, "w") as f:
            json.dump(result, f)

        hits = len(predictions)
        total_ads += len(ad_ids)
        total_hits += hits
        print(f"got {hits}/{len(ad_ids)} ({hits/len(ad_ids):.0%})")

    print(f"\nDone! Total coverage: {total_hits}/{total_ads} ({total_hits/total_ads:.0%})")
    print(f"Output saved to {output_dir}/")


if __name__ == "__main__":
    main()
