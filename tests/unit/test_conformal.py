"""Conformal prediction: marginal coverage, finite-sample quantile, Mondrian variant.

The conformal guarantee only holds under exchangeability. This module checks it on
synthetic data constructed to be exchangeable — the only regime in which the
coverage assertion means anything.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from kinoscope.eval.conformal import ConformalCalibrator, coverage_metrics

ALPHA = 0.1

# Size of the synthetic dataset. PICP is a binomial mean: its Monte-Carlo standard
# deviation is sqrt(0.9 * 0.1 / n_test). At n_test = 20 000 it falls to ~0.0021, so
# the [0.88, 0.92] window required by the plan sits about seven standard deviations
# from the bounds. At n_test = 2 000 the same test fails on one seed in twenty from
# sampling noise alone — the window would be narrower than the measurement error,
# and the test would be measuring chance.
N_TOTAL = 40_000


def _exchangeable_sample(
    seed: int, n: int = N_TOTAL
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (y, mu, sigma) with y = mu + sigma * noise, noise ~ N(0, 1).

    Calibration and test are drawn from the same law, so exchangeability holds by
    construction — exactly the assumption split conformal makes.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(7.0, 1.0, n)
    sigma = np.full(n, 0.5)
    y = mu + rng.normal(0.0, 1.0, n) * sigma
    return y, mu, sigma


def test_marginal_coverage_on_exchangeable_data() -> None:
    """Exit test 6.3: PICP within [0.88, 0.92] over 20 seeds at alpha = 0.1."""
    for seed in range(20):
        y, mu, sigma = _exchangeable_sample(seed)
        half = len(y) // 2
        calibrator = ConformalCalibrator(alpha=ALPHA).fit(y[:half], mu[:half], sigma[:half])
        low, high = calibrator.predict_interval(mu[half:], sigma[half:])
        picp = coverage_metrics(y[half:], low, high)["picp"]
        assert 0.88 <= picp <= 0.92, f"seed {seed}: PICP = {picp:.4f}"


def test_tighter_alpha_widens_the_intervals() -> None:
    y, mu, sigma = _exchangeable_sample(0, n=4_000)
    half = len(y) // 2
    widths = {}
    for alpha in (0.2, 0.1, 0.05):
        calibrator = ConformalCalibrator(alpha=alpha).fit(y[:half], mu[:half], sigma[:half])
        low, high = calibrator.predict_interval(mu[half:], sigma[half:])
        widths[alpha] = coverage_metrics(y[half:], low, high)["mpiw"]
    assert widths[0.2] < widths[0.1] < widths[0.05]


def test_nonconformity_is_the_normalised_absolute_residual() -> None:
    y = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([1.0, 2.0, 0.5])
    scores = ConformalCalibrator._nonconformity(y, mu, sigma)
    assert scores == pytest.approx(np.array([0.0, 0.5, 4.0]))


def test_empty_calibration_set_yields_nan_quantile() -> None:
    """Edge case: with no calibration point, the quantile is undefined."""
    empty = np.array([], dtype=np.float64)
    calibrator = ConformalCalibrator(alpha=ALPHA).fit(empty, empty, empty)
    assert np.isnan(calibrator.quantile_)


def test_mondrian_variant_produces_one_quantile_per_group() -> None:
    y, mu, sigma = _exchangeable_sample(0, n=4_000)
    groups = np.array(["TK", "CMGC", "AGC", "CAMK"] * (len(y) // 4))
    calibrator = ConformalCalibrator(alpha=ALPHA).fit(y, mu, sigma, groups=groups)
    assert calibrator.group_quantiles_ is not None
    assert set(calibrator.group_quantiles_) == {"TK", "CMGC", "AGC", "CAMK"}

    low, high = calibrator.predict_interval(mu, sigma, groups=groups)
    assert np.all(high > low)


def test_undersized_group_falls_back_to_the_marginal_quantile() -> None:
    """A group too small to support the requested level must not produce a
    degenerate quantile: fall back to the marginal one."""
    y, mu, sigma = _exchangeable_sample(0, n=4_000)
    groups = np.array(["majority"] * (len(y) - 3) + ["rare"] * 3)
    calibrator = ConformalCalibrator(alpha=ALPHA).fit(y, mu, sigma, groups=groups)
    assert calibrator.group_quantiles_ is not None
    assert calibrator.group_quantiles_["rare"] == calibrator.quantile_


def test_coverage_metrics_reports_width_alongside_coverage() -> None:
    """Correct PICP with enormous intervals is a useless model."""
    y = np.array([0.0, 1.0, 2.0])
    low = np.array([-1.0, 0.0, 1.0])
    high = np.array([1.0, 2.0, 3.0])
    report = coverage_metrics(y, low, high)
    assert report["picp"] == 1.0
    assert report["mpiw"] == pytest.approx(2.0)
    assert report["nmpiw"] == pytest.approx(1.0)
