"""Chemical standardisation: salt neutralised, rejections, idempotence.

Idempotence is not a nicety: standardisation runs at training time and again at
inference time. If `standardize` is not a fixed point, the two paths diverge and
the model sees molecules in production that it never saw in training.
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

# Imatinib free base. The mesylate is the same active ingredient accompanied by a
# methanesulfonate counter-ion: after largest-fragment selection, both must yield
# an identical InChIKey.
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
IMATINIB_MESYLATE = f"{IMATINIB}.CS(=O)(=O)O"

# Valid, standardisable SMILES used as the sampling population for the idempotence
# property.
VALID_SMILES: list[str] = [
    IMATINIB,
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
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
    assert result.inchikey.count("-") == 2, "InChIKey expected in AAA-BBB-C form"
    # The skeleton is the first InChIKey layer, without stereochemistry.
    assert result.inchikey.startswith(result.inchikey_skeleton)
    assert result.n_heavy_atoms > MIN_HEAVY_ATOMS
    assert result.mol_weight > 0.0


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("[Na+].[Cl-]", "inorganic salt: no organic fragment retained"),
        ("CC", "too few heavy atoms"),
        ("C" * (MAX_HEAVY_ATOMS + 20), "too many heavy atoms"),
        ("[Pt](Cl)(Cl)(N)N", "element outside the allowed list"),
        ("this_is_not_a_smiles", "unparsable"),
    ],
)
def test_rejects_out_of_domain_inputs(smiles: str, reason: str) -> None:
    assert standardize(smiles) is None, f"should have been rejected ({reason})"


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

    Checked on the canonical output rather than the input: it is that fixed-point
    property which guarantees training/inference parity.
    """
    once = standardize(smiles)
    assert once is not None
    twice = standardize(once.smiles)
    assert twice is not None
    assert twice.smiles == once.smiles
    assert twice.inchikey == once.inchikey
