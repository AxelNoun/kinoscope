"""Standardisation chimique : sel neutralisé, rejets, idempotence.

L'idempotence n'est pas une coquetterie : la standardisation est appliquée à
l'entraînement puis de nouveau à l'inférence. Si `standardize` n'est pas un
point fixe, les deux chemins divergent et le modèle voit en production des
molécules qu'il n'a jamais vues à l'entraînement.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kinoscope.data.standardize import (
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    StandardizedMolecule,
    standardize,
    standardize_batch,
)

# Imatinib base. Le mésylate est le même principe actif accompagné d'un
# contre-ion méthanesulfonate : après sélection du plus gros fragment, les deux
# doivent produire un InChIKey identique.
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
IMATINIB_MESYLATE = f"{IMATINIB}.CS(=O)(=O)O"

# Jeu de SMILES valides et standardisables, utilisé comme population de tirage
# pour le test d'idempotence.
VALID_SMILES: list[str] = [
    IMATINIB,
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirine
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caféine
    "c1ccc2ccccc2c1CCC",
    "c1ccc2nccnc2c1CC",
    "C1CCNCC1CCC",
    "c1ccc(-c2ccccc2)cc1C",
    "c1ccc2[nH]ccc2c1CC",
]


def test_mesylate_salt_reduces_to_the_free_base() -> None:
    base, salt = standardize(IMATINIB), standardize(IMATINIB_MESYLATE)
    assert base is not None
    assert salt is not None
    assert base.inchikey == salt.inchikey
    assert base.smiles == salt.smiles


def test_returns_a_fully_populated_record() -> None:
    result = standardize(IMATINIB)
    assert isinstance(result, StandardizedMolecule)
    assert result.inchikey.count("-") == 2, "InChIKey attendu sous la forme AAA-BBB-C"
    # Le squelette est la première couche de l'InChIKey, sans stéréochimie.
    assert result.inchikey.startswith(result.inchikey_skeleton)
    assert result.n_heavy_atoms > MIN_HEAVY_ATOMS
    assert result.mol_weight > 0.0


@pytest.mark.parametrize(
    ("smiles", "motif"),
    [
        ("[Na+].[Cl-]", "sel inorganique : aucun fragment organique retenu"),
        ("CC", "trop peu d'atomes lourds"),
        ("C" * (MAX_HEAVY_ATOMS + 20), "trop d'atomes lourds"),
        ("[Pt](Cl)(Cl)(N)N", "élément hors liste autorisée"),
        ("ceci_n_est_pas_un_smiles", "non parsable"),
    ],
)
def test_rejects_out_of_domain_inputs(smiles: str, motif: str) -> None:
    assert standardize(smiles) is None, f"aurait dû être rejeté ({motif})"


def test_batch_preserves_order_and_marks_rejects() -> None:
    results = standardize_batch([IMATINIB, "[Na+].[Cl-]", "CC(=O)Oc1ccccc1C(=O)O"])
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


@given(smiles=st.sampled_from(VALID_SMILES))
@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_standardization_is_idempotent(smiles: str) -> None:
    """`standardize(standardize(x)) == standardize(x)`.

    Vérifié sur la sortie canonique, pas sur l'entrée : c'est la propriété de
    point fixe qui garantit la parité entraînement/inférence.
    """
    once = standardize(smiles)
    assert once is not None
    twice = standardize(once.smiles)
    assert twice is not None
    assert twice.smiles == once.smiles
    assert twice.inchikey == once.inchikey
