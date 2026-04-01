#!/usr/bin/env python3
"""
Create train/test split data to eliminate label leakage.

For each request, splits positive ads 50/50:
  - history_labels: tools see these (simulates past clicks)
  - test_labels: evaluation only (simulates future clicks)

Output: data_split/<request_id>.npz with:
  - Same fields as original (user_emb, ad_embs, ad_ids)
  - history_labels: 1 for history positives, 0 otherwise (tools use this)
  - test_labels: 1 for test positives, 0 otherwise (evaluation uses this)
  - labels: original labels (for reference only, NOT used by tools or eval)

Usage:
    python3 scripts/create_split_data.py --data-dir data --max-requests 100
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/local/model/raw")
    parser.add_argument("--output-dir", default="data/local/model/split")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(Path(args.data_dir).glob("request_*.npz"))[:args.max_requests]
    rng = np.random.RandomState(args.seed)

    print(f"Creating train/test split for {len(npz_files)} requests...")

    stats = {"total_pos": 0, "history": 0, "test": 0}

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        labels = data["labels"]
        pos_idx = np.where(labels == 1)[0]

        rng.shuffle(pos_idx)
        half = len(pos_idx) // 2
        history_idx = set(pos_idx[:half].tolist())
        test_idx = set(pos_idx[half:].tolist())

        history_labels = np.zeros_like(labels)
        test_labels = np.zeros_like(labels)
        for idx in history_idx:
            history_labels[idx] = 1
        for idx in test_idx:
            test_labels[idx] = 1

        stats["total_pos"] += len(pos_idx)
        stats["history"] += len(history_idx)
        stats["test"] += len(test_idx)

        np.savez(
            out_dir / npz_path.name,
            request_id=data["request_id"],
            user_emb=data["user_emb"],
            ad_embs=data["ad_embs"],
            ad_ids=data["ad_ids"],
            labels=labels,  # original, for reference
            history_labels=history_labels,  # tools use this
            test_labels=test_labels,  # evaluation uses this
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(npz_files)}] done")

    print(f"\nDone: {len(npz_files)} requests")
    print(f"  Total positives: {stats['total_pos']}")
    print(f"  History (tools): {stats['history']} ({stats['history']/stats['total_pos']*100:.0f}%)")
    print(f"  Test (eval):     {stats['test']} ({stats['test']/stats['total_pos']*100:.0f}%)")
    print(f"  Output: {out_dir}/")


if __name__ == "__main__":
    main()
