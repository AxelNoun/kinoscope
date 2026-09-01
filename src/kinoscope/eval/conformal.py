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

import numpy as np


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    quantile_: float = float("nan")
    group_quantiles_: dict[str, float] | None = None

    @staticmethod
    def _nonconformity(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Score normalisé |y - mu| / sigma : produit des intervalles adaptatifs,
        plus larges là où le modèle se déclare lui-même incertain."""
        return np.abs(y - mu) / np.clip(sigma, 1e-6, None)

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return float("nan")
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # correction d'échantillon fini
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, groups: np.ndarray | None = None):
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

    def predict_interval(self, mu: np.ndarray, sigma: np.ndarray, groups: np.ndarray | None = None):
        if groups is not None and self.group_quantiles_:
            q = np.array([self.group_quantiles_.get(str(g), self.quantile_) for g in groups])
        else:
            q = np.full_like(mu, self.quantile_, dtype=float)
        half = q * np.clip(sigma, 1e-6, None)
        return mu - half, mu + half


def coverage_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    """PICP = couverture empirique ; MPIW = largeur moyenne d'intervalle.
    Un PICP correct avec un MPIW énorme est un modèle inutile : rapporter les deux."""
    inside = (y >= lo) & (y <= hi)
    return {
        "picp": float(inside.mean()),
        "mpiw": float(np.mean(hi - lo)),
        "nmpiw": float(np.mean(hi - lo) / (y.max() - y.min())),
    }
