"""Domaine d'applicabilité : similarité au plus proche voisin du train.

Ce module vérifie la mécanique du score — pas la relation entre similarité et
erreur du modèle, qui relève de l'étape 6.4 et exige de vraies prédictions.
"""

from __future__ import annotations

import numpy as np

from kinoscope.eval.applicability import ApplicabilityDomain

TRAIN = [
    "c1ccc2ccccc2c1CCC",
    "c1ccc2nccnc2c1CC",
    "CC(=O)Oc1ccccc1C(=O)O",
]


def test_a_training_molecule_scores_one_against_itself() -> None:
    domain = ApplicabilityDomain().fit(TRAIN)
    assert domain.score([TRAIN[0]])[0] == 1.0


def test_fit_returns_self_for_chaining() -> None:
    domain = ApplicabilityDomain()
    assert domain.fit(TRAIN) is domain


def test_dissimilar_molecule_scores_below_a_training_molecule() -> None:
    domain = ApplicabilityDomain().fit(TRAIN)
    scores = domain.score([TRAIN[0], "C1CCNCC1CCC"])
    assert scores[0] > scores[1]


def test_is_inside_applies_the_configured_threshold() -> None:
    strict = ApplicabilityDomain(threshold=0.99).fit(TRAIN)
    permissive = ApplicabilityDomain(threshold=0.0).fit(TRAIN)
    probe = ["C1CCNCC1CCC"]
    assert not strict.is_inside(probe)[0]
    assert permissive.is_inside(probe)[0]


def test_scores_are_bounded_to_the_unit_interval() -> None:
    domain = ApplicabilityDomain().fit(TRAIN)
    scores = domain.score([*TRAIN, "C1CCNCC1CCC", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"])
    assert np.all((scores >= 0.0) & (scores <= 1.0))
