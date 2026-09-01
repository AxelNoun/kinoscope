"""Verrouille le contrat featurizer <-> modèle. Un décalage ici produit un
IndexError d'embedding, ou pire, un modèle qui s'entraîne sur des features
silencieusement tronquées."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from kinoscope.data.featurizer import (
    N_ATOM_FEATURES,
    N_BOND_FEATURES,
    mol_to_graph,
    smiles_to_graph,
)
from kinoscope.models.model import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                                       # aspirine
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", # imatinib
    "C[C@H](N)C(=O)O",                                             # stéréocentre
    "c1ccc2c(c1)ccc1ccccc12",                                      # polycyclique
    "[Na+].[Cl-]",                                                 # sel (à rejeter en amont)
]


def test_feature_dims_match_model() -> None:
    assert len(ATOM_FEATURE_DIMS) == N_ATOM_FEATURES
    assert len(BOND_FEATURE_DIMS) == N_BOND_FEATURES


@pytest.mark.parametrize("smi", SMILES[:4])
def test_graph_topology(smi: str) -> None:
    mol = Chem.MolFromSmiles(smi)
    g = mol_to_graph(mol)
    assert g.x.shape == (mol.GetNumAtoms(), N_ATOM_FEATURES)
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] == 2 * mol.GetNumBonds()  # graphe bidirectionnel
    assert g.edge_attr.shape == (2 * mol.GetNumBonds(), N_BOND_FEATURES)


@pytest.mark.parametrize("smi", SMILES[:4])
def test_indices_within_embedding_cardinality(smi: str) -> None:
    g = smiles_to_graph(smi)
    for col, dim in enumerate(ATOM_FEATURE_DIMS):
        assert int(g.x[:, col].max()) < dim, f"feature atomique {col} déborde de {dim}"
        assert int(g.x[:, col].min()) >= 0
    if g.edge_attr.numel():
        for col, dim in enumerate(BOND_FEATURE_DIMS):
            assert int(g.edge_attr[:, col].max()) < dim


def test_determinism() -> None:
    a, b = smiles_to_graph(SMILES[1]), smiles_to_graph(SMILES[1])
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)


def test_stereochemistry_is_encoded() -> None:
    r = smiles_to_graph("C[C@H](N)C(=O)O")
    s = smiles_to_graph("C[C@@H](N)C(=O)O")
    assert not torch.equal(r.x, s.x), "les énantiomères doivent différer"


def test_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError):
        smiles_to_graph("this-is-not-a-smiles")
