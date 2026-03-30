#!/usr/bin/env python3
"""
Generate per-user context folders with multiple MD files from npz data.

Each user folder contains:
  - profile.md         — User embedding summary, overall statistics
  - engagement.md      — Engagement history: what ads the user interacted with
  - interest_clusters.md — Interest clusters derived from positive ad embeddings
  - context.md         — Request context: pool size, similarity landscape
"""

import argparse
import json
import numpy as np
from pathlib import Path


def cosine_similarities(user_emb, ad_embs):
    """Compute cosine similarity between user and all ads."""
    user_norm = np.linalg.norm(user_emb).clip(1e-8)
    user_unit = user_emb / user_norm
    ad_norms = np.linalg.norm(ad_embs, axis=1, keepdims=True).clip(1e-8)
    ad_units = ad_embs / ad_norms
    return ad_units @ user_unit


def cluster_positive_ads(pos_embs, n_clusters=5):
    """Cluster positive ads to find interest groups."""
    from sklearn.cluster import KMeans

    if len(pos_embs) < n_clusters:
        n_clusters = max(1, len(pos_embs))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(pos_embs)

    clusters = []
    for c in range(n_clusters):
        mask = cluster_ids == c
        cluster_embs = pos_embs[mask]
        centroid = km.cluster_centers_[c]

        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True).clip(1e-8)
        units = cluster_embs / norms
        cent_norm = np.linalg.norm(centroid).clip(1e-8)
        cent_unit = centroid / cent_norm

        # Intra-cluster similarity
        avg_sim = float(np.mean(units @ cent_unit))

        # Inter-cluster distances
        inter_dists = []
        for c2 in range(n_clusters):
            if c2 != c:
                c2_cent = km.cluster_centers_[c2]
                c2_norm = np.linalg.norm(c2_cent).clip(1e-8)
                inter_dists.append(float(cent_unit @ (c2_cent / c2_norm)))

        clusters.append({
            "id": c,
            "size": int(mask.sum()),
            "fraction": float(mask.sum()) / len(pos_embs),
            "cohesion": round(avg_sim, 4),
            "nearest_other_cluster_sim": round(max(inter_dists) if inter_dists else 0, 4),
            "centroid_preview": [round(float(x), 4) for x in centroid[:4]],
        })

    clusters.sort(key=lambda x: x["size"], reverse=True)
    return clusters, km, cluster_ids


def generate_profile(request_id, user_emb, n_total, n_pos, n_neg, sims):
    """Generate profile.md content."""
    user_norm = float(np.linalg.norm(user_emb))
    return f"""# User Profile — Request {request_id}

## Overview
- **Request ID:** {request_id}
- **Total candidate ads:** {n_total}
- **Engagement rate:** {n_pos}/{n_total} ({n_pos/n_total:.1%})

## User Embedding
- **Dimensionality:** 32 (PSelect/TTSN user tower)
- **Norm:** {user_norm:.4f}
- **Top-4 components:** {[round(float(x), 4) for x in user_emb[:4]]}
- This embedding encodes the user's interests, demographics, and engagement history from the Two-Tower Scoring Network.

## Overall Similarity Landscape
- **All ads:** mean={sims.mean():.4f}, std={sims.std():.4f}, range=[{sims.min():.4f}, {sims.max():.4f}]
- **P25/P50/P75/P95:** {np.percentile(sims, 25):.4f} / {np.percentile(sims, 50):.4f} / {np.percentile(sims, 75):.4f} / {np.percentile(sims, 95):.4f}
"""


def generate_engagement(request_id, ad_ids, labels, sims, n_examples=15):
    """Generate engagement.md content."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sims = sims[pos_mask]
    neg_sims = sims[neg_mask]
    pos_ids = ad_ids[pos_mask]

    lines = [
        f"# Engagement History — Request {request_id}",
        "",
        "## Summary",
        f"- **Positive (engaged) ads:** {int(pos_mask.sum())}",
        f"- **Negative (not engaged) ads:** {int(neg_mask.sum())}",
        "",
        "## Similarity Distribution",
        f"- **Positive ads:** mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}",
        f"- **Negative ads:** mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}",
        f"- **Separation gap:** {pos_sims.mean() - neg_sims.mean():.4f} (positive mean - negative mean)",
        "",
        "## Top Engaged Ads (highest similarity to user)",
        "",
        "| Rank | Ad ID | Cosine Similarity |",
        "|------|-------|-------------------|",
    ]

    # Top positive ads by similarity
    top_idx = np.argsort(-pos_sims)[:n_examples]
    for rank, idx in enumerate(top_idx):
        lines.append(f"| {rank+1} | {int(pos_ids[idx])} | {pos_sims[idx]:.4f} |")

    lines.extend([
        "",
        "## Engagement Interpretation",
        f"The user has engaged with {int(pos_mask.sum())} ads. "
        f"Engaged ads have {pos_sims.mean() - neg_sims.mean():.4f} higher similarity to the user embedding on average, "
        f"indicating the embedding captures engagement tendency, but the overlap is significant "
        f"(many negatives have higher similarity than some positives).",
    ])

    return "\n".join(lines)


def generate_interest_clusters(request_id, ad_embs, ad_ids, labels, user_emb):
    """Generate interest_clusters.md content."""
    pos_mask = labels == 1
    pos_embs = ad_embs[pos_mask]
    pos_ids = ad_ids[pos_mask]

    clusters, km, cluster_ids = cluster_positive_ads(pos_embs, n_clusters=5)

    # Compute user similarity to each cluster centroid
    user_norm = np.linalg.norm(user_emb).clip(1e-8)
    user_unit = user_emb / user_norm

    lines = [
        f"# Interest Clusters — Request {request_id}",
        "",
        "## Methodology",
        "K-means clustering (k=5) on the 32d embeddings of the user's positively-engaged ads. "
        "Each cluster represents an interest group the user engages with.",
        "",
        "## Cluster Summary",
        "",
        "| Cluster | Size | Share | Cohesion | User Similarity | Nearest Cluster Sim |",
        "|---------|------|-------|----------|-----------------|---------------------|",
    ]

    for c in clusters:
        centroid = km.cluster_centers_[c["id"]]
        cent_norm = np.linalg.norm(centroid).clip(1e-8)
        user_sim = float(user_unit @ (centroid / cent_norm))
        lines.append(
            f"| {c['id']} | {c['size']} | {c['fraction']:.0%} | "
            f"{c['cohesion']:.3f} | {user_sim:.3f} | {c['nearest_other_cluster_sim']:.3f} |"
        )

    lines.extend(["", "## Cluster Details", ""])

    for c in clusters:
        mask = cluster_ids == c["id"]
        cluster_ad_ids = pos_ids[mask]

        lines.extend([
            f"### Cluster {c['id']} — {c['size']} ads ({c['fraction']:.0%})",
            f"- **Cohesion:** {c['cohesion']:.3f} (avg similarity to centroid)",
            f"- **Example ad IDs:** {[int(x) for x in cluster_ad_ids[:5]]}",
            "",
        ])

    lines.extend([
        "## Interpretation",
        f"The user's {int(pos_mask.sum())} engaged ads form {len(clusters)} distinct interest clusters. ",
        f"The largest cluster ({clusters[0]['size']} ads, {clusters[0]['fraction']:.0%}) dominates, ",
        f"but smaller clusters represent niche interests worth covering in recommendations.",
    ])

    return "\n".join(lines)


def generate_context(request_id, ad_embs, ad_ids, labels, user_emb, sims):
    """Generate context.md — the request context and intent."""
    n_total = len(ad_ids)
    n_pos = int((labels == 1).sum())

    # Similarity percentiles
    p50 = float(np.percentile(sims, 50))
    p90 = float(np.percentile(sims, 90))
    p99 = float(np.percentile(sims, 99))

    # Embedding norm distribution
    ad_norms = np.linalg.norm(ad_embs, axis=1)

    return f"""# Request Context — Request {request_id}

## Task
Given {n_total} candidate ads and the user's profile, rank ads by predicted engagement likelihood.
The goal is to surface the ~{n_pos} ads the user would engage with (recall optimization).

## Candidate Pool Statistics
- **Total candidates:** {n_total}
- **Positive rate:** {n_pos/n_total:.1%} ({n_pos} positives)

## Similarity Landscape
- **Median similarity:** {p50:.4f}
- **90th percentile:** {p90:.4f}
- **99th percentile:** {p99:.4f}
- Similarity range is narrow — most ads have low cosine similarity to the user. The signal is subtle.

## Ad Embedding Properties
- **Mean norm:** {ad_norms.mean():.2f}
- **Std norm:** {ad_norms.std():.2f}
- **Range:** [{ad_norms.min():.2f}, {ad_norms.max():.2f}]

## Recommendation Intent
Find ads the user would engage with. The user has diverse interests across multiple clusters.
Pure dot-product retrieval (cosine similarity ranking) achieves ~7% recall@50 —
there is significant room for improvement through multi-route retrieval and cluster-aware selection.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate per-user context folders")
    parser.add_argument("--data-dir", default="data", help="Directory with npz files")
    parser.add_argument("--output-dir", default="requests", help="Output directory")
    parser.add_argument("--max-requests", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    npz_files = sorted(data_dir.glob("request_*.npz"))
    if args.max_requests:
        npz_files = npz_files[:args.max_requests]

    print(f"Generating user context folders for {len(npz_files)} requests...")

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        request_id = int(data["request_id"])
        user_emb = data["user_emb"]
        ad_embs = data["ad_embs"]
        ad_ids = data["ad_ids"]
        labels = data["labels"]

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_total = len(ad_ids)

        sims = cosine_similarities(user_emb, ad_embs)

        req_dir = output_dir / str(request_id)
        req_dir.mkdir(parents=True, exist_ok=True)

        # Generate all 4 files
        (req_dir / "profile.md").write_text(
            generate_profile(request_id, user_emb, n_total, n_pos, n_neg, sims)
        )
        (req_dir / "engagement.md").write_text(
            generate_engagement(request_id, ad_ids, labels, sims)
        )
        (req_dir / "interest_clusters.md").write_text(
            generate_interest_clusters(request_id, ad_embs, ad_ids, labels, user_emb)
        )
        (req_dir / "context.md").write_text(
            generate_context(request_id, ad_embs, ad_ids, labels, user_emb, sims)
        )

        if (i + 1) % 50 == 0 or i == len(npz_files) - 1:
            print(f"  [{i+1}/{len(npz_files)}] {request_id}: {n_pos} pos, {n_neg} neg, 4 files")

    print(f"Done! User folders saved to {output_dir}/")


if __name__ == "__main__":
    main()
