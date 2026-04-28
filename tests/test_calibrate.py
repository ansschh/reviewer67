"""Unit tests for the calibration metric helpers — no API required."""
from __future__ import annotations

import numpy as np

from paper_reviewer.calibrate import _bootstrap_ci, _jaccard, _spearman, _auc


def test_bootstrap_returns_consistent_ci():
    a = list(np.linspace(0, 10, 50))
    b = [x + np.random.default_rng(0).normal(0, 0.2) for x in a]
    point, lo, hi = _bootstrap_ci(a, b, _spearman, n_iter=200)
    assert 0.9 < point <= 1.0
    assert lo <= point <= hi
    assert hi - lo < 0.2  # tight for low-noise pairs


def test_bootstrap_handles_constant_input():
    # All-zeros AUC label: stat returns nan, bootstrap should not crash.
    point, lo, hi = _bootstrap_ci([0] * 30, [0.5] * 30, _auc, n_iter=100)
    assert np.isnan(point) or np.isnan(lo)


def test_jaccard_basic():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0
    assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5
    assert np.isnan(_jaccard(set(), set()))


def test_spearman_perfect():
    assert abs(_spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9


def test_auc_separable():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert _auc(labels, scores) == 1.0
