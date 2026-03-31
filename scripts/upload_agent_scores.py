#!/usr/bin/env python3
"""
Upload agent recommendation scores to Hive for recall pipeline evaluation.

Reads agent outputs from outputs/<run_id>/<request_id>.json,
creates a Hive table with (request_id, ad_id, agent_rank, agent_score).

Usage:
    python3 scripts/upload_agent_scores.py --run-id cc_9sample --ds 2026-03-19
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_presto_query(query, namespace="tmp_ads", limit=10000, retries=2):
    """Execute a Presto query via jf graphql."""
    # Escape double quotes in the query for embedding in GraphQL
    escaped_query = query.replace("\\", "\\\\").replace('"', '\\"')
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


def load_agent_outputs(output_dir, run_id):
    """Load all agent output JSONs for a given run_id.

    Returns list of (request_id, ad_id, agent_rank, agent_score) tuples.
    """
    run_dir = Path(output_dir) / run_id
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    json_files = sorted(run_dir.glob("*.json"))
    if not json_files:
        print(f"Error: no JSON files found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: skipping {json_path.name}: {e}", file=sys.stderr)
            continue

        request_id = data.get("request_id")
        ranked_ads = data.get("ranked_ads", [])
        if not request_id or not ranked_ads:
            print(f"  Warning: skipping {json_path.name}: missing request_id or ranked_ads", file=sys.stderr)
            continue

        n = len(ranked_ads)
        for rank, ad_id in enumerate(ranked_ads):
            # agent_score: normalized 0-1, higher is better
            # rank 0 (top) gets score (N-0)/N = 1.0, last gets 1/N
            agent_score = (n - rank) / n
            rows.append((int(request_id), int(ad_id), rank + 1, agent_score))

    return rows


def create_table_if_not_exists(namespace, table_name):
    """Create the Hive table if it doesn't exist."""
    query = (
        f"CREATE TABLE IF NOT EXISTS {table_name} ("
        f"ds VARCHAR, "
        f"run_id VARCHAR, "
        f"request_id BIGINT, "
        f"ad_id BIGINT, "
        f"agent_rank INTEGER, "
        f"agent_score DOUBLE"
        f")"
    )
    print(f"Creating table {namespace}.{table_name} if not exists...")
    result = run_presto_query(query, namespace=namespace, limit=1)
    if result is None:
        print("  Warning: CREATE TABLE returned None (may already exist)", file=sys.stderr)
    else:
        print("  Table ready.")


def upload_rows(rows, ds, run_id, namespace, table_name, batch_size=1000):
    """Upload rows to Hive via batched INSERT INTO ... VALUES statements."""
    total = len(rows)
    print(f"Uploading {total} rows in batches of {batch_size}...")

    uploaded = 0
    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        values_parts = []
        for request_id, ad_id, agent_rank, agent_score in batch:
            values_parts.append(
                f"('{ds}', '{run_id}', {request_id}, {ad_id}, {agent_rank}, {agent_score:.6f})"
            )
        values_str = ", ".join(values_parts)
        query = f"INSERT INTO {table_name} VALUES {values_str}"

        result = run_presto_query(query, namespace=namespace, limit=1)
        if result is None:
            print(f"  Warning: batch {batch_start}-{batch_start+len(batch)} may have failed", file=sys.stderr)
        else:
            uploaded += len(batch)
            print(f"  Uploaded {uploaded}/{total} rows")

        time.sleep(0.5)  # Rate limiting

    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Upload agent recommendation scores to Hive"
    )
    parser.add_argument("--run-id", required=True, help="Run ID (e.g. cc_9sample)")
    parser.add_argument("--ds", required=True, help="Date string (e.g. 2026-03-19)")
    parser.add_argument("--output-dir", default="outputs", help="Directory with run outputs")
    parser.add_argument("--namespace", default="tmp_ads", help="Hive namespace")
    parser.add_argument("--table-name", default="agent_recommendation_scores", help="Hive table name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per INSERT batch")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without uploading")
    args = parser.parse_args()

    # Load agent outputs
    rows = load_agent_outputs(args.output_dir, args.run_id)
    print(f"Loaded {len(rows)} (request_id, ad_id) pairs from {args.output_dir}/{args.run_id}/")

    if not rows:
        print("No rows to upload. Exiting.")
        sys.exit(1)

    # Print summary
    request_ids = set(r[0] for r in rows)
    print(f"  Unique requests: {len(request_ids)}")
    print(f"  Avg ads per request: {len(rows) / len(request_ids):.1f}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload to:")
        print(f"  Table: {args.namespace}.{args.table_name}")
        print(f"  ds={args.ds}, run_id={args.run_id}")
        print(f"  Sample rows:")
        for row in rows[:5]:
            print(f"    request_id={row[0]}, ad_id={row[1]}, rank={row[2]}, score={row[3]:.4f}")
        return

    # Create table and upload
    create_table_if_not_exists(args.namespace, args.table_name)
    uploaded = upload_rows(
        rows, args.ds, args.run_id,
        args.namespace, args.table_name,
        batch_size=args.batch_size,
    )
    print(f"\nDone! Uploaded {uploaded}/{len(rows)} rows to {args.namespace}.{args.table_name}")
    print(f"  ds={args.ds}, run_id={args.run_id}")


if __name__ == "__main__":
    main()
