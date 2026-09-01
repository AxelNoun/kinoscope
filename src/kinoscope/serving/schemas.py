"""Contrats d'API (Pydantic v2). La validation chimique se fait ICI, pas dans le modèle."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

AssayType = Literal["Kd", "Ki", "IC50", "EC50", "unknown"]


class PredictionRequest(BaseModel):
    smiles: Annotated[str, Field(min_length=2, max_length=600, examples=["Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"])]
    uniprot_id: Annotated[str, Field(pattern=r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$", examples=["P00519"])]
    assay_type: AssayType = "Kd"

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        from rdkit import Chem, RDLogger  # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        if Chem.MolFromSmiles(v) is None:
            raise ValueError("SMILES non parsable par RDKit")
        return v


class BatchPredictionRequest(BaseModel):
    items: Annotated[list[PredictionRequest], Field(min_length=1, max_length=1000)]


class AtomResidueContact(BaseModel):
    atom_index: int
    klifs_position: int
    weight: float


class PredictionResponse(BaseModel):
    p_affinity: float = Field(description="Affinité prédite, échelle -log10(M)")
    ci_lower: float
    ci_upper: float
    coverage: float = Field(default=0.90, description="Couverture nominale de l'intervalle conforme")
    p_active: float
    in_applicability_domain: bool
    nearest_neighbour_tanimoto: float
    top_contacts: list[AtomResidueContact] = Field(default_factory=list)
    model_version: str

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    device: str
    n_targets_cached: int

    model_config = {"protected_namespaces": ()}
