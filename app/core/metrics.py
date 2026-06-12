"""NumPy reference implementations of the corpus metrics (Pillars B and C).

These are the ground truth used to cross-check the ClickHouse SQL results in
the test suite, and they power the in-memory database fallback.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def semantic_spread(embeddings: list[np.ndarray]) -> float:
    """Mean pairwise cosine distance: 2 / (K(K-1)) * sum_{i<j} (1 - cos(e_i, e_j)).

    For unit vectors this equals 1 - (||s||^2 - K) / (K(K-1)) where s = sum e_i.
    """
    K = len(embeddings)
    if K < 2:
        return 0.0
    E = np.vstack([e / (np.linalg.norm(e) or 1.0) for e in embeddings])
    s = E.sum(axis=0)
    sum_cos = (float(s @ s) - K) / 2.0          # sum_{i<j} cos(e_i, e_j)
    n_pairs = K * (K - 1) / 2
    return float(1.0 - sum_cos / n_pairs)


def shannon_entropy(labels: list[str]) -> float:
    """H(S) = -sum p(x) log2 p(x), in bits."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def normalized_entropy(labels: list[str]) -> float:
    """H(S) / log2(|S|), in [0, 1]. 1.0 = perfectly uniform across sources."""
    unique = len(set(labels))
    if unique <= 1:
        return 0.0
    return shannon_entropy(labels) / math.log2(unique)


def doc_level_embeddings(rows: list[dict]) -> dict[str, np.ndarray]:
    """Mean chunk embedding per URL (matches SQL avgForEach + GROUP BY url)."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for r in rows:
        e = np.asarray(r["embedding"], dtype=np.float64)
        u = r["url"]
        if u in sums:
            sums[u] += e
            counts[u] += 1
        else:
            sums[u] = e.copy()
            counts[u] = 1
    return {u: (sums[u] / counts[u]) for u in sums}
