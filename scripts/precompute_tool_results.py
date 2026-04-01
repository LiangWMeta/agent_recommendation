#!/usr/bin/env python3
"""Pre-compute all tool results for batch agent recommendation.

Runs all retrieval tools on each request and writes formatted markdown
files that agents can read directly. No Claude/LLM needed — pure Python.

Usage:
    python3 scripts/precompute_tool_results.py \
      --data-dir data/local/model/split \
      --output-dir /tmp/tool_results \
      --max-requests 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_common import load_request, precompute_tool_results, format_tool_results


def main():
    parser = argparse.ArgumentParser(description="Pre-compute tool results for agent recommendation")
    parser.add_argument("--data-dir", default="data/local/model/split")
    parser.add_argument("--output-dir", default="/tmp/tool_results")
    parser.add_argument("--max-requests", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("request_*.npz"))[:args.max_requests]
    if not npz_files:
        print(f"ERROR: No NPZ files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Pre-computing tool results for {len(npz_files)} requests...", file=sys.stderr)
    t0 = time.time()
    request_ids = []

    for i, npz_path in enumerate(npz_files):
        rd = load_request(npz_path)
        rid = rd["request_id"]
        request_ids.append(rid)

        results = precompute_tool_results(rd)
        tool_text = format_tool_results(rd, results)

        out_path = output_dir / f"{rid}.md"
        out_path.write_text(tool_text)

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(npz_files)}] done", file=sys.stderr)

    # Write manifest
    manifest = {
        "data_dir": str(data_dir),
        "n_requests": len(request_ids),
        "request_ids": request_ids,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"Done: {len(request_ids)} requests in {elapsed:.1f}s ({elapsed / len(request_ids):.2f}s/req)", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
