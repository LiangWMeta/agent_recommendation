#!/usr/bin/env python3
"""Compare metrics across multiple evaluation runs side by side."""

import argparse
import json
import os
import sys


K_VALUES = [10, 20, 50, 100]
METRICS = ["recall", "precision", "ndcg"]


def load_run(run_id: str, results_dir: str = "evaluation/results") -> dict:
    """Load a run's results JSON. Tries <run_id>.json and baseline.json."""
    path = os.path.join(results_dir, f"{run_id}.json")
    if not os.path.exists(path):
        print(f"WARNING: results file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple evaluation runs")
    parser.add_argument("--run-ids", nargs="+", required=True,
                        help="List of run IDs to compare (use 'baseline' for baseline)")
    parser.add_argument("--results-dir", default="evaluation/results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    runs = {}
    for rid in args.run_ids:
        data = load_run(rid, args.results_dir)
        if data:
            runs[rid] = data

    if not runs:
        print("ERROR: No valid runs to compare.")
        sys.exit(1)

    run_ids = list(runs.keys())

    # Determine which metrics are available across runs
    available_keys = set()
    for data in runs.values():
        available_keys.update(data.get("aggregate", {}).keys())

    # Build ordered list of metric keys
    metric_keys = []
    for metric in METRICS:
        for k in K_VALUES:
            key = f"{metric}@{k}"
            if key in available_keys:
                metric_keys.append(key)

    # Print header
    col_width = max(12, max(len(rid) for rid in run_ids) + 2)
    header = f"{'Metric':<16}"
    for rid in run_ids:
        header += f" {rid:>{col_width}}"
    if len(run_ids) >= 2:
        header += f" {'Delta':>{col_width}}"

    print(f"\nComparison of {len(run_ids)} runs")
    print(f"{'='*len(header)}")

    # Print request counts
    counts_line = f"{'n_requests':<16}"
    for rid in run_ids:
        n = runs[rid].get("n_requests", "?")
        counts_line += f" {str(n):>{col_width}}"
    print(counts_line)
    print(f"{'-'*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for key in metric_keys:
        line = f"{key:<16}"
        values = []
        for rid in run_ids:
            agg = runs[rid].get("aggregate", {})
            if key in agg:
                val = agg[key]["mean"]
                values.append(val)
                line += f" {val:>{col_width}.4f}"
            else:
                values.append(None)
                line += f" {'N/A':>{col_width}}"

        # Delta: last run vs first run
        if len(run_ids) >= 2 and values[0] is not None and values[-1] is not None:
            delta = values[-1] - values[0]
            line += f" {delta:>+{col_width}.4f}"

        print(line)

    print(f"{'-'*len(header)}")

    # Print 95% CI for each run
    print(f"\n95% Confidence Intervals:")
    for rid in run_ids:
        print(f"\n  {rid}:")
        agg = runs[rid].get("aggregate", {})
        for key in metric_keys:
            if key in agg and "ci_95" in agg[key]:
                ci = agg[key]["ci_95"]
                print(f"    {key:<16} [{ci[0]:.4f}, {ci[1]:.4f}]")


if __name__ == "__main__":
    main()
