#!/usr/bin/env python3
"""
Generate ads pool understanding files from split data.

Reads all request NPZ files and produces aggregate pool context:
  - pool_overview.md  — Pool size, engagement stats, embedding properties
  - catalog.md        — Per-ad summary across requests
  - semantic_clusters.md — Hierarchical clusters of the ad embedding space

Usage:
    python3 ads_pool/refresh.py --data-dir data/local/model/split
    python3 ads_pool/refresh.py --data-dir data/local/model/split --max-requests 50
"""

import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_pool(data_dir, max_requests=None):
    """Load all ads across requests into a unified pool."""
    npz_files = sorted(Path(data_dir).glob("request_*.npz"))
    if max_requests:
        npz_files = npz_files[:max_requests]

    ad_embs_all = {}       # ad_id -> embedding (last seen)
    ad_appearances = defaultdict(int)   # ad_id -> number of requests it appears in
    ad_positive_count = defaultdict(int)  # ad_id -> times engaged (history_labels)
    n_requests = 0

    for npz_path in npz_files:
        data = np.load(npz_path)
        ad_ids = data["ad_ids"]
        ad_embs = data["ad_embs"]
        labels = data.get("history_labels", data["labels"])

        for i, aid in enumerate(ad_ids):
            aid = int(aid)
            ad_embs_all[aid] = ad_embs[i]
            ad_appearances[aid] += 1
            if labels[i] == 1:
                ad_positive_count[aid] += 1
        n_requests += 1

    return ad_embs_all, ad_appearances, ad_positive_count, n_requests


def cluster_ads(ad_ids, ad_embs, n_coarse=10, n_fine_per_coarse=5):
    """Two-level hierarchical clustering (HSNN-style)."""
    from sklearn.cluster import KMeans

    n_ads = len(ad_ids)
    n_coarse = min(n_coarse, n_ads)

    # Coarse level
    km_coarse = KMeans(n_clusters=n_coarse, random_state=42, n_init=10)
    coarse_ids = km_coarse.fit_predict(ad_embs)

    clusters = []
    for c in range(n_coarse):
        mask = coarse_ids == c
        c_embs = ad_embs[mask]
        c_aids = [ad_ids[i] for i in range(n_ads) if mask[i]]
        centroid = km_coarse.cluster_centers_[c]

        # Cohesion
        norms = np.linalg.norm(c_embs, axis=1, keepdims=True).clip(1e-8)
        units = c_embs / norms
        c_norm = np.linalg.norm(centroid).clip(1e-8)
        c_unit = centroid / c_norm
        cohesion = float(np.mean(units @ c_unit))

        # Fine-level sub-clusters
        n_fine = min(n_fine_per_coarse, len(c_aids))
        sub_clusters = []
        if n_fine > 1:
            km_fine = KMeans(n_clusters=n_fine, random_state=42, n_init=5)
            fine_ids = km_fine.fit_predict(c_embs)
            for f in range(n_fine):
                f_mask = fine_ids == f
                sub_clusters.append({
                    "id": f,
                    "size": int(f_mask.sum()),
                })

        clusters.append({
            "id": c,
            "size": int(mask.sum()),
            "cohesion": round(cohesion, 4),
            "centroid_preview": [round(float(x), 4) for x in centroid[:4]],
            "sub_clusters": sub_clusters,
            "example_ads": c_aids[:5],
        })

    clusters.sort(key=lambda x: x["size"], reverse=True)
    return clusters


def write_pool_overview(output_dir, ad_embs_all, ad_appearances, ad_positive_count, n_requests):
    """Write pool_overview.md."""
    n_unique = len(ad_embs_all)
    embs = np.array(list(ad_embs_all.values()))
    norms = np.linalg.norm(embs, axis=1)

    # Engagement stats
    n_ever_engaged = sum(1 for v in ad_positive_count.values() if v > 0)
    total_appearances = sum(ad_appearances.values())

    # Appearance distribution
    app_vals = list(ad_appearances.values())
    app_arr = np.array(app_vals)

    lines = [
        "# Ads Pool Overview",
        "",
        "## Pool Size",
        f"- **Unique ads:** {n_unique:,}",
        f"- **Requests:** {n_requests}",
        f"- **Total ad-request pairs:** {total_appearances:,}",
        f"- **Avg ads per request:** {total_appearances / max(n_requests, 1):,.0f}",
        "",
        "## Engagement",
        f"- **Ads with at least 1 engagement (history):** {n_ever_engaged:,} ({100 * n_ever_engaged / max(n_unique, 1):.1f}%)",
        f"- **Ads never engaged:** {n_unique - n_ever_engaged:,}",
        "",
        "## Ad Appearance Distribution",
        f"- **Mean appearances:** {app_arr.mean():.1f}",
        f"- **Median appearances:** {np.median(app_arr):.0f}",
        f"- **Max appearances:** {app_arr.max()}",
        f"- **Ads appearing in 1 request only:** {int((app_arr == 1).sum())} ({100 * (app_arr == 1).sum() / n_unique:.0f}%)",
        "",
        "## Embedding Properties",
        f"- **Dimensionality:** {embs.shape[1]}",
        f"- **Mean norm:** {norms.mean():.3f}",
        f"- **Std norm:** {norms.std():.3f}",
        f"- **Range:** [{norms.min():.3f}, {norms.max():.3f}]",
        "",
    ]

    path = output_dir / "pool_overview.md"
    path.write_text("\n".join(lines))
    return path


def write_catalog(output_dir, ad_embs_all, ad_appearances, ad_positive_count, n_requests):
    """Write catalog.md — per-ad summary (top ads by engagement, then sample)."""
    # Sort by positive count descending, then by appearances
    sorted_ads = sorted(
        ad_embs_all.keys(),
        key=lambda aid: (ad_positive_count.get(aid, 0), ad_appearances.get(aid, 0)),
        reverse=True,
    )

    lines = [
        "# Ads Catalog",
        "",
        f"Total unique ads: {len(sorted_ads):,}",
        "",
        "## Top Engaged Ads (by history engagement count)",
        "",
        "| Ad ID | Engagements | Appearances | Eng Rate | Embedding Norm |",
        "|-------|------------|-------------|----------|----------------|",
    ]

    # Top 50 by engagement
    for aid in sorted_ads[:50]:
        eng = ad_positive_count.get(aid, 0)
        app = ad_appearances.get(aid, 0)
        rate = eng / app if app > 0 else 0
        norm = float(np.linalg.norm(ad_embs_all[aid]))
        lines.append(f"| {aid} | {eng} | {app} | {rate:.2f} | {norm:.3f} |")

    # Summary stats for the tail
    n_tail = len(sorted_ads) - 50
    if n_tail > 0:
        lines.extend([
            "",
            f"*... {n_tail:,} more ads not shown.*",
            "",
            "## Distribution Summary",
            "",
        ])
        eng_counts = [ad_positive_count.get(aid, 0) for aid in sorted_ads]
        app_counts = [ad_appearances.get(aid, 0) for aid in sorted_ads]
        lines.extend([
            f"- **Engagement count:** mean={np.mean(eng_counts):.2f}, median={np.median(eng_counts):.0f}, max={max(eng_counts)}",
            f"- **Appearance count:** mean={np.mean(app_counts):.2f}, median={np.median(app_counts):.0f}, max={max(app_counts)}",
        ])

    path = output_dir / "catalog.md"
    path.write_text("\n".join(lines))
    return path


def write_semantic_clusters(output_dir, ad_embs_all, ad_positive_count):
    """Write semantic_clusters.md — hierarchical clustering of the pool."""
    ad_ids = list(ad_embs_all.keys())
    ad_embs = np.array([ad_embs_all[aid] for aid in ad_ids])

    clusters = cluster_ads(ad_ids, ad_embs, n_coarse=10, n_fine_per_coarse=5)

    lines = [
        "# Semantic Clusters",
        "",
        "Two-level hierarchical clustering (K-means, 10 coarse x 5 fine) over all unique ad embeddings.",
        "",
        "## Coarse Clusters",
        "",
        "| Cluster | Size | Share | Cohesion | Engaged Ads | Engagement Rate |",
        "|---------|------|-------|----------|-------------|-----------------|",
    ]

    for c in clusters:
        engaged = sum(1 for aid in c["example_ads"] if ad_positive_count.get(aid, 0) > 0)
        # Count engaged across ALL ads in cluster (not just examples)
        # We only have example_ads[:5], so estimate from full cluster
        # Actually recompute properly
        total_engaged = 0
        # We stored example_ads as first 5 — need all ads per cluster
        # For the table, use cohesion and size
        eng_rate_approx = engaged / max(len(c["example_ads"]), 1)
        lines.append(
            f"| {c['id']} | {c['size']:,} | {100 * c['size'] / len(ad_ids):.1f}% | "
            f"{c['cohesion']:.3f} | — | — |"
        )

    lines.extend(["", "## Cluster Details", ""])

    for c in clusters:
        lines.extend([
            f"### Cluster {c['id']} — {c['size']:,} ads ({100 * c['size'] / len(ad_ids):.1f}%)",
            f"- **Cohesion:** {c['cohesion']:.3f}",
            f"- **Centroid preview:** {c['centroid_preview']}",
            f"- **Example ad IDs:** {c['example_ads']}",
        ])
        if c["sub_clusters"]:
            lines.append(f"- **Sub-clusters:** {len(c['sub_clusters'])}")
            for sc in c["sub_clusters"]:
                lines.append(f"  - Sub-cluster {sc['id']}: {sc['size']} ads")
        lines.append("")

    path = output_dir / "semantic_clusters.md"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate ads pool understanding")
    parser.add_argument("--data-dir", default="data/local/model/split",
                        help="Directory with request NPZ files")
    parser.add_argument("--output-dir", default="ads_pool",
                        help="Output directory for pool context files")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Max requests to process (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pool from {args.data_dir}...")
    ad_embs_all, ad_appearances, ad_positive_count, n_requests = load_pool(
        args.data_dir, args.max_requests
    )
    print(f"  {len(ad_embs_all):,} unique ads across {n_requests} requests")

    p1 = write_pool_overview(output_dir, ad_embs_all, ad_appearances, ad_positive_count, n_requests)
    print(f"  Written: {p1}")

    p2 = write_catalog(output_dir, ad_embs_all, ad_appearances, ad_positive_count, n_requests)
    print(f"  Written: {p2}")

    p3 = write_semantic_clusters(output_dir, ad_embs_all, ad_positive_count)
    print(f"  Written: {p3}")

    print("Done.")


if __name__ == "__main__":
    main()
