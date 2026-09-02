"""Prédiction conforme : couverture marginale, quantile fini, variante Mondrian.

La garantie conforme ne vaut que sous échangeabilité. Ce module la vérifie sur
des données synthétiques construites pour être échangeables — c'est le seul
régime où l'assertion de couverture a un sens.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from kinoscope.eval.conformal import ConformalCalibrator, coverage_metrics

ALPHA = 0.1

# Taille du jeu synthétique. Le PICP est une moyenne binomiale : son écart-type
# de Monte-Carlo vaut sqrt(0.9 * 0.1 / n_test). Avec n_test = 20 000, il tombe à
# ~0.0021, si bien que la fenêtre [0.88, 0.92] exigée par le plan se situe à
# environ sept écarts-types des bornes. À n_test = 2 000, le même test échoue une
# fois sur vingt graines par simple bruit d'échantillonnage — la fenêtre serait
# plus étroite que l'erreur de mesure, et le test mesurerait le hasard.
N_TOTAL = 40_000


def _exchangeable_sample(
    seed: int, n: int = N_TOTAL
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Renvoie (y, mu, sigma) avec y = mu + sigma * bruit, bruit ~ N(0, 1).

    Calibration et test sont tirés de la même loi : l'échangeabilité est vraie
    par construction, ce qui est l'hypothèse exacte du conformal split.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(7.0, 1.0, n)
    sigma = np.full(n, 0.5)
    y = mu + rng.normal(0.0, 1.0, n) * sigma
    return y, mu, sigma


def test_marginal_coverage_on_exchangeable_data() -> None:
    """Test de sortie 6.3 : PICP dans [0.88, 0.92] sur 20 graines, alpha = 0.1."""
    for seed in range(20):
        y, mu, sigma = _exchangeable_sample(seed)
        half = len(y) // 2
        calibrator = ConformalCalibrator(alpha=ALPHA).fit(y[:half], mu[:half], sigma[:half])
        low, high = calibrator.predict_interval(mu[half:], sigma[half:])
        picp = coverage_metrics(y[half:], low, high)["picp"]
        assert 0.88 <= picp <= 0.92, f"graine {seed}: PICP = {picp:.4f}"


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
    """Cas limite : sans point de calibration, le quantile n'est pas défini."""
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
    """Un groupe trop petit pour supporter le niveau demandé ne doit pas produire
    un quantile dégénéré : on retombe sur le quantile marginal."""
    y, mu, sigma = _exchangeable_sample(0, n=4_000)
    groups = np.array(["majoritaire"] * (len(y) - 3) + ["rare"] * 3)
    calibrator = ConformalCalibrator(alpha=ALPHA).fit(y, mu, sigma, groups=groups)
    assert calibrator.group_quantiles_ is not None
    assert calibrator.group_quantiles_["rare"] == calibrator.quantile_


def test_coverage_metrics_reports_width_alongside_coverage() -> None:
    """Un PICP correct avec des intervalles énormes est un modèle inutile."""
    y = np.array([0.0, 1.0, 2.0])
    low = np.array([-1.0, 0.0, 1.0])
    high = np.array([1.0, 2.0, 3.0])
    report = coverage_metrics(y, low, high)
    assert report["picp"] == 1.0
    assert report["mpiw"] == pytest.approx(2.0)
    assert report["nmpiw"] == pytest.approx(1.0)
