"""Métriques d'évaluation DTI, avec intervalles de confiance bootstrap.

Toute métrique rapportée dans le README doit être accompagnée de son IC 95 %
bootstrap. Un point isolé sur un test set de 5 000 paires n'est pas un résultat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    pearson_r: float
    spearman_rho: float
    concordance_index: float
    r2m: float
    aupr: float
    auroc: float
    ef1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def rmse(y: npt.NDArray[np.float64], yhat: npt.NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def concordance_index(
    y: npt.NDArray[np.float64], yhat: npt.NDArray[np.float64], *, chunk: int = 2048
) -> float:
    """CI = (paires concordantes + 0.5 * ex aequo) / paires comparables.

    Implémentation O(n^2) chunkée : exacte, et suffisante jusqu'à ~5.10^4 points.
    Au-delà, sous-échantillonner et rapporter la variance.
    """
    n = len(y)
    concordant = ties = comparable = 0.0
    for start in range(0, n, chunk):
        yi = y[start : start + chunk, None]
        pi = yhat[start : start + chunk, None]
        mask = yi > y[None, :]
        comparable += mask.sum()
        concordant += ((pi > yhat[None, :]) & mask).sum()
        ties += ((pi == yhat[None, :]) & mask).sum()
    if comparable == 0:
        return float("nan")
    return float((concordant + 0.5 * ties) / comparable)


def r2m(y: npt.NDArray[np.float64], yhat: npt.NDArray[np.float64]) -> float:
    """r_m^2 de Roy : pénalise les modèles dont la régression forcée à l'origine
    diverge du modèle libre. Standard QSAR pour détecter un biais systématique."""
    r2 = stats.pearsonr(y, yhat)[0] ** 2
    k = float(np.sum(y * yhat) / np.sum(yhat**2))
    r0_2 = 1.0 - np.sum((y - k * yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2 * (1.0 - np.sqrt(abs(r2 - r0_2))))


def enrichment_factor(
    y: npt.NDArray[np.float64],
    yhat: npt.NDArray[np.float64],
    *,
    threshold: float,
    fraction: float = 0.01,
) -> float:
    """EF à x % : facteur d'enrichissement en actifs dans le top x % du ranking.
    Métrique de triage réel de librairie, plus parlante qu'une RMSE."""
    active = y >= threshold
    if active.sum() == 0:
        return float("nan")
    k = max(1, int(round(fraction * len(y))))
    top = np.argsort(-yhat)[:k]
    return float((active[top].mean()) / active.mean())


def compute_metrics(
    y: npt.NDArray[np.float64], yhat: npt.NDArray[np.float64], *, activity_threshold: float = 6.5
) -> RegressionMetrics:
    active = (y >= activity_threshold).astype(int)
    has_both_classes = 0 < active.sum() < len(active)
    return RegressionMetrics(
        rmse=rmse(y, yhat),
        mae=float(np.mean(np.abs(y - yhat))),
        pearson_r=float(stats.pearsonr(y, yhat)[0]),
        spearman_rho=float(stats.spearmanr(y, yhat)[0]),
        concordance_index=concordance_index(y, yhat),
        r2m=r2m(y, yhat),
        aupr=float(average_precision_score(active, yhat)) if has_both_classes else float("nan"),
        auroc=float(roc_auc_score(active, yhat)) if has_both_classes else float("nan"),
        ef1=enrichment_factor(y, yhat, threshold=activity_threshold),
    )


def bootstrap_ci(
    y: npt.NDArray[np.float64],
    yhat: npt.NDArray[np.float64],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    activity_threshold: float = 6.5,
) -> dict[str, tuple[float, float, float]]:
    """Retourne {métrique: (valeur, borne_basse, borne_haute)} par bootstrap percentile."""
    rng = np.random.default_rng(seed)
    point = compute_metrics(y, yhat, activity_threshold=activity_threshold).to_dict()
    draws: dict[str, list[float]] = {k: [] for k in point}
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        m = compute_metrics(y[idx], yhat[idx], activity_threshold=activity_threshold).to_dict()
        for k, v in m.items():
            draws[k].append(v)
    out: dict[str, tuple[float, float, float]] = {}
    for k, v in point.items():
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if arr.size else (np.nan, np.nan)
        out[k] = (v, float(lo), float(hi))
    return out
