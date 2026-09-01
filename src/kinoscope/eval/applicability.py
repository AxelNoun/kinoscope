"""Domaine d'applicabilité : refuser de prédire est une décision valide.

Un modèle QSAR interrogé sur un chémotype absent du corpus d'entraînement
extrapole. Le signaler est une exigence de l'OCDE pour les modèles QSAR
réglementaires, et la moindre des choses pour un modèle de portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import DataStructs

from kinoscope.data.splits import morgan_fingerprints


@dataclass
class ApplicabilityDomain:
    threshold: float = 0.35
    _train_fps: list = None  # type: ignore[assignment]

    def fit(self, train_smiles: list[str]) -> "ApplicabilityDomain":
        self._train_fps = morgan_fingerprints(train_smiles)
        return self

    def score(self, smiles: list[str]) -> np.ndarray:
        """Similarité Tanimoto au plus proche voisin du train set."""
        fps = morgan_fingerprints(smiles)
        return np.array([max(DataStructs.BulkTanimotoSimilarity(f, self._train_fps)) for f in fps])

    def is_inside(self, smiles: list[str]) -> np.ndarray:
        return self.score(smiles) >= self.threshold
