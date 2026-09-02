"""Contrats d'API : la validation chimique et syntaxique se fait à l'entrée.

Une entrée invalide doit être refusée par le schéma — c'est-à-dire produire un
HTTP 422 côté FastAPI — jamais atteindre le modèle ni provoquer une prédiction
silencieuse sur une molécule que RDKit ne sait pas lire.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kinoscope.serving.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)

IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
ABL1 = "P00519"


def test_accepts_a_well_formed_request() -> None:
    request = PredictionRequest(smiles=IMATINIB, uniprot_id=ABL1)
    assert request.smiles == IMATINIB
    assert request.assay_type == "Kd"


@pytest.mark.parametrize(
    "smiles",
    [
        "ceci_n_est_pas_un_smiles",
        "c1ccccc",  # cycle non refermé
        "C(((",
    ],
)
def test_rejects_smiles_that_rdkit_cannot_parse(smiles: str) -> None:
    with pytest.raises(ValidationError, match="RDKit"):
        PredictionRequest(smiles=smiles, uniprot_id=ABL1)


@pytest.mark.parametrize("uniprot_id", ["PAS_UNE_ACCESSION", "12345", "P0051", ""])
def test_rejects_malformed_uniprot_accessions(uniprot_id: str) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(smiles=IMATINIB, uniprot_id=uniprot_id)


def test_rejects_an_unknown_assay_type() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(smiles=IMATINIB, uniprot_id=ABL1, assay_type="RMN")


def test_batch_is_bounded_at_both_ends() -> None:
    item = PredictionRequest(smiles=IMATINIB, uniprot_id=ABL1)
    assert len(BatchPredictionRequest(items=[item]).items) == 1

    with pytest.raises(ValidationError):
        BatchPredictionRequest(items=[])
    with pytest.raises(ValidationError):
        BatchPredictionRequest(items=[item] * 1001)


def test_response_defaults_to_the_nominal_conformal_coverage() -> None:
    response = PredictionResponse(
        p_affinity=7.5,
        ci_lower=6.9,
        ci_upper=8.1,
        p_active=0.8,
        in_applicability_domain=True,
        nearest_neighbour_tanimoto=0.42,
        model_version="0.1.0-test",
    )
    assert response.coverage == 0.90
    assert response.top_contacts == []
