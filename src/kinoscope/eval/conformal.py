"""Prédiction conforme normalisée (split conformal + variante Mondrian).

Garantie : sous hypothèse d'échangeabilité entre calibration et test, la
couverture marginale est >= 1 - alpha, indépendamment du modèle.

Avertissement méthodologique à conserver dans le model card : sur un split
cold-target, l'échangeabilité est VIOLÉE par construction (shift de covariables).
La couverture empirique observée est donc un résultat à mesurer et à rapporter,
pas une garantie. La conformalisation Mondrian par groupe de Manning restaure
partiellement la validité conditionnelle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    quantile_: float = float("nan")
    group_quantiles_: dict[str, float] | None = None

    @staticmethod
    def _nonconformity(
        y: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        sigma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Score normalisé |y - mu| / sigma : produit des intervalles adaptatifs,
        plus larges là où le modèle se déclare lui-même incertain."""
        scores: npt.NDArray[np.float64] = np.abs(y - mu) / np.clip(sigma, 1e-6, None)
        return scores

    @staticmethod
    def _conformal_quantile(scores: npt.NDArray[np.float64], alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return float("nan")
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # correction d'échantillon fini
        return float(np.quantile(scores, level, method="higher"))

    def fit(
        self,
        y: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        sigma: npt.NDArray[np.float64],
        groups: npt.NDArray[Any] | None = None,
    ) -> ConformalCalibrator:
        scores = self._nonconformity(y, mu, sigma)
        self.quantile_ = self._conformal_quantile(scores, self.alpha)
        if groups is not None:
            self.group_quantiles_ = {}
            for g in np.unique(groups):
                m = groups == g
                # Repli sur le quantile marginal si le groupe est trop petit pour
                # supporter le niveau demandé (n < 1/alpha - 1).
                self.group_quantiles_[str(g)] = (
                    self._conformal_quantile(scores[m], self.alpha)
                    if m.sum() >= int(np.ceil(1 / self.alpha)) - 1
                    else self.quantile_
                )
        return self

    def predict_interval(
        self,
        mu: npt.NDArray[np.float64],
        sigma: npt.NDArray[np.float64],
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if groups is not None and self.group_quantiles_:
            q = np.array([self.group_quantiles_.get(str(g), self.quantile_) for g in groups])
        else:
            q = np.full_like(mu, self.quantile_, dtype=float)
        half = q * np.clip(sigma, 1e-6, None)
        return mu - half, mu + half


def coverage_metrics(
    y: npt.NDArray[np.float64],
    lo: npt.NDArray[np.float64],
    hi: npt.NDArray[np.float64],
) -> dict[str, float]:
    """PICP = couverture empirique ; MPIW = largeur moyenne d'intervalle.
    Un PICP correct avec un MPIW énorme est un modèle inutile : rapporter les deux."""
    inside = (y >= lo) & (y <= hi)
    return {
        "picp": float(inside.mean()),
        "mpiw": float(np.mean(hi - lo)),
        "nmpiw": float(np.mean(hi - lo) / (y.max() - y.min())),
    }
