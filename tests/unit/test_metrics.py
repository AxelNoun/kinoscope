"""Evaluation metrics, checked against analytically known vectors.

Every assertion targets an exactly known value rather than a tolerance chosen
after the fact: perfect ranking, reversed ranking, no signal at all.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from kinoscope.eval.metrics import (
    RegressionMetrics,
    bootstrap_ci,
    compute_metrics,
    concordance_index,
    enrichment_factor,
    r2m,
    rmse,
)

Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0])


def _noisy_pair(
    n: int = 400, seed: int = 0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    y = rng.normal(7.0, 1.5, n)
    return y, y + rng.normal(0.0, 0.5, n)


def test_rmse_is_zero_on_a_perfect_fit() -> None:
    assert rmse(Y, Y) == 0.0


def test_rmse_equals_the_constant_offset() -> None:
    assert rmse(Y, Y + 2.0) == pytest.approx(2.0)


def test_concordance_index_is_one_for_a_perfect_ranking() -> None:
    assert concordance_index(Y, Y) == 1.0


def test_concordance_index_is_zero_for_a_reversed_ranking() -> None:
    assert concordance_index(Y, -Y) == 0.0


def test_concordance_index_is_one_half_without_any_signal() -> None:
    """Constant prediction: every pair is tied, so CI = 0.5."""
    assert concordance_index(Y, np.zeros_like(Y)) == 0.5


def test_concordance_index_is_invariant_under_chunking() -> None:
    """Chunking is a memory optimisation only: it must change nothing."""
    y, yhat = _noisy_pair()
    assert concordance_index(y, yhat, chunk=17) == pytest.approx(
        concordance_index(y, yhat, chunk=4096)
    )


def test_r2m_is_one_on_a_perfect_fit() -> None:
    assert r2m(Y, Y) == pytest.approx(1.0)


def test_enrichment_factor_rewards_a_perfect_ranking() -> None:
    """One active in five, recovered in the top fifth of the ranking: the active
    fraction there is 1.0 against 0.2 on average, so EF = 5."""
    y = np.array([5.0, 5.0, 5.0, 5.0, 9.0])
    assert enrichment_factor(y, y, threshold=8.0, fraction=0.2) == pytest.approx(5.0)


def test_enrichment_factor_is_nan_without_any_active() -> None:
    assert np.isnan(enrichment_factor(Y, Y, threshold=99.0))


def test_compute_metrics_returns_every_declared_field() -> None:
    y, yhat = _noisy_pair()
    metrics = compute_metrics(y, yhat, activity_threshold=7.0)
    assert isinstance(metrics, RegressionMetrics)
    values = metrics.to_dict()
    assert set(values) == {
        "rmse",
        "mae",
        "pearson_r",
        "spearman_rho",
        "concordance_index",
        "r2m",
        "aupr",
        "auroc",
        "ef1",
    }
    assert values["pearson_r"] > 0.8


def test_classification_metrics_are_nan_when_a_class_is_missing() -> None:
    """Without both classes, AUPR and AUROC are undefined: report NaN rather than
    a misleading number."""
    y, yhat = _noisy_pair()
    metrics = compute_metrics(y, yhat, activity_threshold=99.0).to_dict()
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["aupr"])


def test_bootstrap_ci_brackets_point_estimate() -> None:
    """Exit test 6.2: the point estimate falls inside its own interval."""
    y, yhat = _noisy_pair()
    intervals = bootstrap_ci(y, yhat, n_resamples=200, seed=0, activity_threshold=7.0)
    for name, (point, low, high) in intervals.items():
        if np.isnan(point) or np.isnan(low) or np.isnan(high):
            continue
        assert low <= point <= high, f"{name}: {point} outside [{low}, {high}]"


def test_bootstrap_ci_is_reproducible_for_a_given_seed() -> None:
    y, yhat = _noisy_pair()
    first = bootstrap_ci(y, yhat, n_resamples=100, seed=3)
    second = bootstrap_ci(y, yhat, n_resamples=100, seed=3)
    assert first == second
