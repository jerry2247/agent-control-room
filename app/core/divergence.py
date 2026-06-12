"""Constrained Query Divergence (Pillar A).

Objective:
    max  sum_{i<j} ||x_i - x_j||_2^2     s.t.  ||x_i - c||_2 < epsilon

Key identity (all x_i unit-normalized, n selected):
    sum_{i<j} ||x_i - x_j||^2 = n^2 - ||sum_i x_i||^2

so maximizing pairwise dispersion is EXACTLY minimizing the length of the
resultant vector s = sum_i x_i ("balance the forces"). This makes the
objective O(d) to evaluate per candidate subset and lets us brute-force the
optimum for hackathon-scale candidate pools, with a greedy + 2-swap fallback.

epsilon is user-customizable (fixed value) and optimizable (auto mode picks
the knee of the diversity-vs-epsilon curve: the smallest radius achieving
>= 95% of the maximum attainable diversity, i.e. minimal topical drift for
near-maximal viewpoint spread).
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np

BRUTE_FORCE_LIMIT = 30000


def pairwise_sq_sum(E: np.ndarray) -> float:
    """sum_{i<j} ||e_i - e_j||^2 for unit rows of E, via the resultant identity."""
    n = len(E)
    if n < 2:
        return 0.0
    s = E.sum(axis=0)
    return float(n * n - float(s @ s))


def mean_pairwise_sq(E: np.ndarray) -> float:
    n = len(E)
    if n < 2:
        return 0.0
    return pairwise_sq_sum(E) / comb(n, 2)


def _greedy_with_swaps(E: np.ndarray, feasible: np.ndarray, n: int) -> list[int]:
    """Farthest-point greedy seeded by the most distant pair, then 2-swap polish.

    Greedy step uses the identity: adding x to current sum s changes the
    objective by maximizing (k+1)^2 - ||s + x||^2, i.e. pick x minimizing
    ||s + x||^2.
    """
    feas = list(feasible)
    # Seed: most distant feasible pair
    best_pair, best_d = (feas[0], feas[1]), -1.0
    for i, j in itertools.combinations(feas, 2):
        d = float(np.sum((E[i] - E[j]) ** 2))
        if d > best_d:
            best_d, best_pair = d, (i, j)
    selected = list(best_pair)
    s = E[selected[0]] + E[selected[1]]

    while len(selected) < n:
        remaining = [i for i in feas if i not in selected]
        pick = min(remaining, key=lambda i: float(np.sum((s + E[i]) ** 2)))
        selected.append(pick)
        s = s + E[pick]

    # 2-swap local improvement
    improved = True
    iters = 0
    while improved and iters < 60:
        improved = False
        iters += 1
        for out_idx in list(selected):
            for in_idx in [i for i in feas if i not in selected]:
                s_new = s - E[out_idx] + E[in_idx]
                if float(s_new @ s_new) < float(s @ s) - 1e-12:
                    selected.remove(out_idx)
                    selected.append(in_idx)
                    s = s_new
                    improved = True
                    break
            if improved:
                break
    return selected


def select_dispersed(
    E: np.ndarray, center: np.ndarray, epsilon: float, n: int
) -> tuple[list[int], np.ndarray]:
    """Select <= n candidate indices inside the epsilon-ball maximizing dispersion.

    Returns (selected_indices, distances_to_center_for_all_candidates).
    """
    dists = np.linalg.norm(E - center, axis=1)
    feasible = np.where(dists <= epsilon)[0]
    if len(feasible) == 0:
        return [], dists
    if len(feasible) <= n:
        return list(feasible), dists

    if comb(len(feasible), n) <= BRUTE_FORCE_LIMIT:
        best, best_val = None, -1.0
        for sub in itertools.combinations(feasible, n):
            val = pairwise_sq_sum(E[list(sub)])
            if val > best_val:
                best_val, best = val, sub
        return list(best), dists

    return _greedy_with_swaps(E, feasible, n), dists


def optimize_epsilon(
    E: np.ndarray,
    center: np.ndarray,
    n: int,
    grid_min: float = 0.55,
    grid_max: float = 1.45,
    steps: int = 11,
    capture: float = 0.95,
) -> tuple[float, list[dict]]:
    """Auto-tune epsilon: sweep the radius, record achieved diversity, return
    the smallest epsilon capturing >= `capture` of max diversity with a full
    selection. The curve is returned for the dashboard."""
    grid = np.linspace(grid_min, grid_max, steps)
    curve = []
    for eps in grid:
        idx, _ = select_dispersed(E, center, float(eps), n)
        div = mean_pairwise_sq(E[idx]) if len(idx) >= 2 else 0.0
        curve.append({"epsilon": round(float(eps), 4), "diversity": round(div, 6),
                      "selected": len(idx)})
    max_div = max(c["diversity"] for c in curve)
    if max_div <= 0:
        return float(grid[-1]), curve
    full = [c for c in curve if c["selected"] >= min(n, 2) and c["diversity"] >= capture * max_div]
    chosen = full[0] if full else max(curve, key=lambda c: c["diversity"])
    return chosen["epsilon"], curve
