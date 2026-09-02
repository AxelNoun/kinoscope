"""Applicability domain: refusing to predict is a valid decision.

A QSAR model queried about a chemotype absent from its training corpus is
extrapolating. Flagging that is an OECD requirement for regulatory QSAR models,
and the least one can do for a portfolio model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from rdkit import DataStructs

from kinoscope.data.splits import morgan_fingerprints


@dataclass
class ApplicabilityDomain:
    threshold: float = 0.35
    # RDKit fingerprints are opaque to the type checker: the stubs shipped by
    # rdkit==2024.3.5 do not parse and are skipped (see pyproject).
    _train_fps: list[Any] = None  # type: ignore[assignment]

    def fit(self, train_smiles: list[str]) -> ApplicabilityDomain:
        self._train_fps = morgan_fingerprints(train_smiles)
        return self

    def score(self, smiles: list[str]) -> npt.NDArray[np.float64]:
        """Tanimoto similarity to the nearest neighbour in the training set."""
        fps = morgan_fingerprints(smiles)
        return np.array([max(DataStructs.BulkTanimotoSimilarity(f, self._train_fps)) for f in fps])

    def is_inside(self, smiles: list[str]) -> npt.NDArray[np.bool_]:
        return self.score(smiles) >= self.threshold
