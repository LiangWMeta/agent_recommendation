#!/usr/bin/env python3
"""
Unified data setup: check existence, build missing data, report readiness.

Orchestrates the full data pipeline:
  1. Raw data (extract_raa.py) — requires Hive access
  2. Train/test split (create_split_data.py)
  3. Prod predictions (extract_prod_predictions.py) — optional
  4. User contexts (prepare_contexts.py)
  5. Ads pool understanding (ads_pool/refresh.py)

Usage:
    python3 scripts/setup_data.py                  # check + build missing
    python3 scripts/setup_data.py --check-only      # just report status
    python3 scripts/setup_data.py --force            # rebuild everything
    python3 scripts/setup_data.py --max-requests 50  # limit request count
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def count_files(directory, pattern):
    """Count files matching a glob pattern in a directory."""
    d = Path(directory)
    if not d.exists():
        return 0
    return len(list(d.glob(pattern)))


def newest_mtime(directory, pattern):
    """Return the newest modification time of files matching pattern, or 0."""
    d = Path(directory)
    if not d.exists():
        return 0
    files = list(d.glob(pattern))
    if not files:
        return 0
    return max(f.stat().st_mtime for f in files)


def check_status():
    """Check and report the status of each data stage."""
    stages = {}

    # Raw data
    raw_dir = BASE_DIR / "data" / "local" / "model" / "raw"
    raw_count = count_files(raw_dir, "request_*.npz")
    stages["raw"] = {"count": raw_count, "dir": str(raw_dir), "ok": raw_count > 0}

    # Split data
    split_dir = BASE_DIR / "data" / "local" / "model" / "split"
    split_count = count_files(split_dir, "request_*.npz")
    split_mtime = newest_mtime(split_dir, "request_*.npz")
    raw_mtime = newest_mtime(raw_dir, "request_*.npz")
    stages["split"] = {
        "count": split_count,
        "dir": str(split_dir),
        "ok": split_count > 0,
        "stale": split_count > 0 and raw_mtime > split_mtime,
    }

    # Enriched (prod predictions)
    enriched_dir = BASE_DIR / "data" / "local" / "model" / "enriched"
    enriched_count = count_files(enriched_dir, "*_prod.json")
    stages["enriched"] = {
        "count": enriched_count,
        "dir": str(enriched_dir),
        "ok": enriched_count > 0,
        "optional": True,
    }

    # User contexts
    user_dir = BASE_DIR / "user"
    user_count = len([d for d in user_dir.iterdir() if d.is_dir()]) if user_dir.exists() else 0
    user_mtime = newest_mtime(user_dir, "*/profile.md")
    stages["user_context"] = {
        "count": user_count,
        "dir": str(user_dir),
        "ok": user_count > 0,
        "stale": user_count > 0 and split_mtime > user_mtime,
    }

    # Ads pool
    pool_dir = BASE_DIR / "ads_pool"
    pool_files = ["pool_overview.md", "catalog.md", "semantic_clusters.md"]
    pool_count = sum(1 for f in pool_files if (pool_dir / f).exists())
    pool_mtime = newest_mtime(pool_dir, "*.md")
    stages["ads_pool"] = {
        "count": pool_count,
        "expected": len(pool_files),
        "dir": str(pool_dir),
        "ok": pool_count == len(pool_files),
        "stale": pool_count > 0 and split_mtime > pool_mtime,
    }

    return stages


def print_status(stages):
    """Print a status table."""
    print("\n=== Data Status ===\n")
    print(f"{'Stage':<16} {'Status':<10} {'Count':<10} {'Notes'}")
    print("-" * 60)

    for name, info in stages.items():
        if info["ok"]:
            status = "STALE" if info.get("stale") else "OK"
        elif info.get("optional"):
            status = "MISSING*"
        else:
            status = "MISSING"

        count_str = str(info["count"])
        if "expected" in info:
            count_str = f"{info['count']}/{info['expected']}"

        notes = ""
        if info.get("stale"):
            notes = "source data is newer — rebuild recommended"
        if info.get("optional") and not info["ok"]:
            notes = "optional — tools use cosine fallback"

        print(f"{name:<16} {status:<10} {count_str:<10} {notes}")

    print()


def run_step(description, cmd, cwd=None):
    """Run a subprocess step with status reporting."""
    print(f"\n--- {description} ---")
    print(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd or str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        return False
    print(f"  OK ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup data for agent recommendation")
    parser.add_argument("--check-only", action="store_true",
                        help="Only report status, don't build anything")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild all stages even if data exists")
    parser.add_argument("--max-requests", type=int, default=100,
                        help="Max requests to process")
    args = parser.parse_args()

    stages = check_status()
    print_status(stages)

    if args.check_only:
        all_ok = all(
            s["ok"] and not s.get("stale")
            for s in stages.values()
            if not s.get("optional")
        )
        sys.exit(0 if all_ok else 1)

    # Step 1: Raw data
    if not stages["raw"]["ok"]:
        extract_script = BASE_DIR / "scripts" / "extract_raa.py"
        if extract_script.exists():
            run_step(
                "Extract RAA source data",
                [sys.executable, str(extract_script),
                 "--output-dir", "data/local/model/raw",
                 "--max-requests", str(args.max_requests)],
            )
        else:
            print("\n--- Extract RAA source data ---")
            print("  SKIP: scripts/extract_raa.py not found.")
            print("  You need raw data in data/local/model/raw/ before proceeding.")
            print("  See data/datasets.md for manual extraction instructions.")
            sys.exit(1)
        # Refresh status
        stages = check_status()

    # Step 2: Train/test split
    if args.force or not stages["split"]["ok"] or stages["split"].get("stale"):
        run_step(
            "Create train/test split",
            [sys.executable, "scripts/create_split_data.py",
             "--data-dir", "data/local/model/raw",
             "--output-dir", "data/local/model/split",
             "--max-requests", str(args.max_requests)],
        )
        stages = check_status()

    # Step 3: Prod predictions (optional)
    if args.force or not stages["enriched"]["ok"]:
        extract_prod = BASE_DIR / "scripts" / "extract_prod_predictions.py"
        if extract_prod.exists():
            ok = run_step(
                "Extract production predictions (optional)",
                [sys.executable, str(extract_prod),
                 "--data-dir", "data/local/model/raw",
                 "--output-dir", "data/local/model/enriched"],
            )
            if not ok:
                print("  Continuing without prod predictions — tools will use cosine fallback.")
        else:
            print("\n--- Extract production predictions ---")
            print("  SKIP: scripts/extract_prod_predictions.py not found.")
        stages = check_status()

    # Step 4: User contexts
    if args.force or not stages["user_context"]["ok"] or stages["user_context"].get("stale"):
        run_step(
            "Generate user context folders",
            [sys.executable, "scripts/prepare_contexts.py",
             "--data-dir", "data/local/model/split",
             "--output-dir", "user/"],
        )
        stages = check_status()

    # Step 5: Ads pool understanding
    if args.force or not stages["ads_pool"]["ok"] or stages["ads_pool"].get("stale"):
        run_step(
            "Generate ads pool understanding",
            [sys.executable, "ads_pool/refresh.py",
             "--data-dir", "data/local/model/split"],
        )
        stages = check_status()

    # Final report
    print_status(stages)

    all_required_ok = all(
        s["ok"] for s in stages.values() if not s.get("optional")
    )
    if all_required_ok:
        print("Data is ready. Run the agent benchmark or pilot diagnosis.")
    else:
        print("Some required data is still missing. Check the status above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
