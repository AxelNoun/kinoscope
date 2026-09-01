"""Featurisation ligand : RDKit Mol -> torch_geometric.data.Data.

Les cardinalités produites ici DOIVENT rester synchronisées avec
``kinoscope.models.model.ATOM_FEATURE_DIMS`` / ``BOND_FEATURE_DIMS``.
Le test ``tests/unit/test_featurizer.py::test_feature_dims_match_model``
verrouille cet invariant.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType, HybridizationType
from torch_geometric.data import Data

# --- tables de correspondance (index 0 = <unk>/défaut) ---------------------- #
HYBRIDIZATION: dict[HybridizationType, int] = {
    HybridizationType.SP: 1,
    HybridizationType.SP2: 2,
    HybridizationType.SP3: 3,
    HybridizationType.SP3D: 4,
    HybridizationType.SP3D2: 5,
}
BOND_TYPE: dict[BondType, int] = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
CHIRAL_TAG: dict[ChiralType, int] = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}

N_ATOM_FEATURES = 9
N_BOND_FEATURES = 4


def _ring_size_index(mol: Chem.Mol, atom_idx: int) -> int:
    """0 = acyclique, 1..4 = cycles de taille 3..6, 5 = cycle >= 7."""
    ri = mol.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings() if atom_idx in r]
    if not sizes:
        return 0
    smallest = min(sizes)
    return min(smallest - 2, 5)


def atom_features(atom: Chem.Atom, mol: Chem.Mol) -> list[int]:
    return [
        min(atom.GetAtomicNum(), 119),  # dim 120
        min(atom.GetDegree(), 7),  # dim 8
        min(max(atom.GetFormalCharge() + 2, 0), 5),  # dim 6, plage [-2, +3]
        HYBRIDIZATION.get(atom.GetHybridization(), 0),  # dim 6
        int(atom.GetIsAromatic()),  # dim 2
        min(atom.GetTotalNumHs(), 5),  # dim 6
        CHIRAL_TAG.get(atom.GetChiralTag(), 3),  # dim 4
        int(atom.IsInRing()),  # dim 2
        _ring_size_index(mol, atom.GetIdx()),  # dim 6
    ]


def bond_features(bond: Chem.Bond) -> list[int]:
    return [
        BOND_TYPE.get(bond.GetBondType(), 0),  # dim 5
        int(bond.GetIsConjugated()),  # dim 2
        int(bond.IsInRing()),  # dim 2
        min(int(bond.GetStereo()), 5),  # dim 6
    ]


def mol_to_graph(mol: Chem.Mol, *, inchikey: str | None = None) -> Data:
    """Graphe non orienté représenté par ses deux arcs (convention PyG)."""
    if mol is None:
        raise ValueError("mol is None")
    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("molecule has no heavy atom")

    x = torch.tensor([atom_features(a, mol) for a in mol.GetAtoms()], dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    edge_feats: list[list[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [feats, feats]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.long)
    else:  # molécule mono-atomique après filtrage : graphe sans arête
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, N_BOND_FEATURES), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = mol.GetNumAtoms()
    if inchikey is not None:
        data.inchikey = inchikey
    return data


def smiles_to_graph(smiles: str, **kwargs: object) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    return mol_to_graph(mol, **kwargs)  # type: ignore[arg-type]
