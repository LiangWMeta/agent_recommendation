#!/usr/bin/env python3
"""
Extract top-N candidate ad_ids per request from the PSelect bulk eval table.

These are the ads that the agent should score — the top of the PSelect
prediction list, which is where recall truncation bites.

The agent then scores these ads (using its 11 retrieval tools), and the
scored ads get boosted in the prediction ranking for recall evaluation.

Usage:
    # Extract top-2000 ad_ids per request for 20 requests
    python3 scripts/extract_bulk_eval_candidates.py --max-requests 20 --top-n 2000

    # This creates data_bulk_eval/<separable_id>.npz with:
    #   - ad_ids: top-N ad_ids from PSelect prediction
    #   - ad_embs: 32d PSelect embeddings for those ads
    #   - user_emb: user embedding
    #   - labels: engagement labels (from RAA)
    #   - pselect_scores: PSelect prediction scores
"""

import argparse
import gc
import json
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This script must be run in an environment with Bamboo (e.g., Bento or FBLearner)
# For local dev, use the FBLearner flow approach instead.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", default="2026-03-18", help="Bulk eval partition date")
    parser.add_argument("--model-id", type=int, default=698799520, help="PSelect model ID")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=2000, help="Top-N ads to extract per request")
    parser.add_argument("--output-dir", default="data/local/eval/bulk_eval")
    args = parser.parse_args()

    logger.info(
        f"This script requires Bamboo (analytics.bamboo). "
        f"Run via Bento notebook or FBLearner flow.\n"
        f"Alternatively, create an FBLearner workflow that:\n"
        f"  1. Calls read_aggregate_bulk_eval_table(ds={args.ds}, model_id={args.model_id}, max_requests={args.max_requests})\n"
        f"  2. For each request, extracts top-{args.top_n} ad_ids from ad_id_list_by_prediction\n"
        f"  3. Loads their embeddings from pselect_bulk_eval_ad_embeddings\n"
        f"  4. Saves as npz files to {args.output_dir}/\n"
    )
    logger.info(
        f"Example FBLearner workflow code:\n\n"
        f"from fblearner.flow.projects.ads.ads_prm.recall_util import read_aggregate_bulk_eval_table\n"
        f"df = read_aggregate_bulk_eval_table(eval_ds='{args.ds}', pselect_model_id={args.model_id}, max_requests={args.max_requests})\n"
        f"for idx, row in df.iterrows():\n"
        f"    pred_ads = list(row['ad_id_list_by_prediction'])\n"
        f"    top_ads = pred_ads[-{args.top_n}:]  # top-N (ascending convention, best at end)\n"
        f"    # Load embeddings, labels, user_emb for these ads\n"
        f"    # Save as npz\n"
    )


if __name__ == "__main__":
    main()
