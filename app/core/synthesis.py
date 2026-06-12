"""Viewpoint clustering and conflict/consensus synthesis (final stage).

Local mode is fully deterministic: k-means on document embeddings, clusters
labeled by their most distinguishing terms, conflict = the two most distant
cluster centroids with representative excerpts. When an LLM backend is
configured, a narrative summary is layered on top of the same structure.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from app.core.embeddings import tokenize


def kmeans(E: np.ndarray, k: int, iters: int = 25, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(E)
    k = max(1, min(k, n))
    # k-means++ style init
    centroids = [E[rng.integers(n)]]
    while len(centroids) < k:
        d2 = np.min([np.sum((E - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = d2 / (d2.sum() or 1.0)
        centroids.append(E[rng.choice(n, p=probs)])
    C = np.vstack(centroids)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((E[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            members = E[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
    return labels


def pca_2d(E: np.ndarray) -> np.ndarray:
    X = E - E.mean(axis=0, keepdims=True)
    if len(X) < 2:
        return np.zeros((len(X), 2))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    span = np.abs(coords).max() or 1.0
    return coords / span


def _distinguishing_terms(cluster_texts: list[str], other_texts: list[str], top: int = 4) -> list[str]:
    inside = Counter(t for txt in cluster_texts for t in tokenize(txt) if "_" not in t)
    outside = Counter(t for txt in other_texts for t in tokenize(txt) if "_" not in t)
    scored = {t: c / (1.0 + outside.get(t, 0)) for t, c in inside.items() if len(t) > 3}
    return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:top]]


def synthesize_viewpoints(docs: list[dict], embeddings: np.ndarray) -> dict:
    """docs: [{url, domain, title, content}], embeddings: row-aligned matrix."""
    n = len(docs)
    if n == 0:
        return {"clusters": [], "consensus_terms": [], "conflict": None}
    k = 3 if n >= 6 else (2 if n >= 3 else 1)
    labels = kmeans(embeddings, k)

    clusters = []
    for j in sorted(set(labels)):
        idx = [i for i in range(n) if labels[i] == j]
        centroid = embeddings[idx].mean(axis=0)
        ranked = sorted(idx, key=lambda i: float(np.sum((embeddings[i] - centroid) ** 2)))
        texts_in = [docs[i]["content"] for i in idx]
        texts_out = [docs[i]["content"] for i in range(n) if labels[i] != j]
        clusters.append({
            "cluster_id": int(j),
            "size": len(idx),
            "label_terms": _distinguishing_terms(texts_in, texts_out),
            "domains": sorted({docs[i]["domain"] for i in idx}),
            "excerpts": [
                {
                    "title": docs[i]["title"],
                    "url": docs[i]["url"],
                    "domain": docs[i]["domain"],
                    "text": docs[i]["content"][:320],
                }
                for i in ranked[:2]
            ],
            "_centroid": centroid,
        })

    # Consensus: terms appearing in a majority of documents corpus-wide
    doc_term_sets = [set(t for t in tokenize(d["content"]) if "_" not in t and len(t) > 3) for d in docs]
    counts = Counter(t for s in doc_term_sets for t in s)
    consensus = [t for t, c in counts.most_common(40) if c >= max(2, int(0.6 * n))][:6]

    # Conflict: the two most distant cluster centroids
    conflict = None
    if len(clusters) >= 2:
        best = (-1.0, 0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = float(np.sum((clusters[a]["_centroid"] - clusters[b]["_centroid"]) ** 2))
                if d > best[0]:
                    best = (d, a, b)
        _, a, b = best
        conflict = {
            "distance": round(best[0], 4),
            "viewpoint_a": {"label_terms": clusters[a]["label_terms"], "excerpt": clusters[a]["excerpts"][0]},
            "viewpoint_b": {"label_terms": clusters[b]["label_terms"], "excerpt": clusters[b]["excerpts"][0]},
        }

    for c in clusters:
        c.pop("_centroid", None)
    return {"clusters": clusters, "consensus_terms": consensus, "conflict": conflict,
            "labels": [int(x) for x in labels]}
