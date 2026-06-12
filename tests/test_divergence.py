"""Unit tests for the constrained dispersion math (Pillar A)."""
import itertools

import numpy as np
import pytest

from app.core.divergence import (
    mean_pairwise_sq, optimize_epsilon, pairwise_sq_sum, select_dispersed,
)


def _unit_rows(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_resultant_identity_matches_naive_sum():
    """sum_{i<j} ||xi-xj||^2 == n^2 - ||sum xi||^2 for unit vectors."""
    E = _unit_rows(7, 16)
    naive = sum(
        float(np.sum((E[i] - E[j]) ** 2)) for i, j in itertools.combinations(range(7), 2)
    )
    assert pairwise_sq_sum(E) == pytest.approx(naive, rel=1e-10)


def test_selection_respects_epsilon_constraint():
    E = _unit_rows(20, 32, seed=3)
    center = E[0]
    eps = 1.1
    idx, dists = select_dispersed(E, center, eps, 5)
    assert len(idx) >= 1
    for i in idx:
        assert dists[i] <= eps + 1e-9


def test_brute_force_is_optimal_on_small_instance():
    """Exact path must beat or match every other feasible subset."""
    E = _unit_rows(10, 8, seed=5)
    center = np.zeros(8)
    center[0] = 1.0
    eps = 2.0  # everything feasible
    idx, _ = select_dispersed(E, center, eps, 4)
    best = pairwise_sq_sum(E[idx])
    for sub in itertools.combinations(range(10), 4):
        assert best >= pairwise_sq_sum(E[list(sub)]) - 1e-9


def test_greedy_path_reasonable_on_large_instance():
    """Force the greedy+swap path (C(40,8) >> limit) and sanity-check quality."""
    E = _unit_rows(40, 16, seed=9)
    center = E.mean(axis=0)
    center /= np.linalg.norm(center)
    idx, _ = select_dispersed(E, center, 2.0, 8)
    assert len(idx) == 8
    greedy_val = pairwise_sq_sum(E[idx])
    rng = np.random.default_rng(0)
    for _ in range(200):  # greedy must beat random subsets
        rand = rng.choice(40, size=8, replace=False)
        assert greedy_val >= pairwise_sq_sum(E[rand]) - 1e-9


def test_diversity_monotone_in_epsilon():
    """Larger radius can never reduce achievable diversity."""
    E = _unit_rows(15, 16, seed=11)
    center = E[0]
    prev = -1.0
    for eps in [0.6, 0.9, 1.2, 1.5, 2.0]:
        idx, _ = select_dispersed(E, center, eps, 5)
        d = mean_pairwise_sq(E[idx]) if len(idx) >= 2 else 0.0
        assert d >= prev - 1e-9
        prev = d


def test_optimize_epsilon_returns_knee_and_curve():
    E = _unit_rows(15, 16, seed=13)
    center = E[0]
    eps, curve = optimize_epsilon(E, center, 5)
    assert len(curve) == 11
    max_div = max(c["diversity"] for c in curve)
    chosen = next(c for c in curve if c["epsilon"] == eps)
    assert chosen["diversity"] >= 0.95 * max_div
    # knee: no smaller epsilon achieves the capture threshold with full selection
    for c in curve:
        if c["epsilon"] < eps:
            assert c["diversity"] < 0.95 * max_div or c["selected"] < 2
