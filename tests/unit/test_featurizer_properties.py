"""Tests par propriétés du featurizer et de la standardisation.

Les tests par exemples de ``test_featurizer.py`` couvrent les cas auxquels on a
pensé. Ceux-ci couvrent la classe : chaque invariant est vérifié sur des
centaines de molécules tirées et transformées, ce qui est le seul moyen
d'attraper les bugs de featurisation qui ne se voient pas à l'œil — un index
qui déborde de la cardinalité d'embedding sur un motif rare, une featurisation
qui dépend de l'ordre d'écriture du SMILES, une arête qui traverse deux
molécules après collation.
"""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st
from rdkit import Chem
from torch_geometric.data import Batch

from kinoscope.data.featurizer import N_ATOM_FEATURES, N_BOND_FEATURES, mol_to_graph
from kinoscope.data.standardize import standardize
from kinoscope.models.model import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from tests.strategies.chem import REFERENCE_SMILES, equivalent_writings, molecules, renumbered_pairs


def test_reference_pool_is_parsable() -> None:
    """Garde-fou du pool : une entrée invalide invaliderait tous les tests
    ci-dessous sans que l'échec en désigne la cause."""
    invalid = [s for s in REFERENCE_SMILES if Chem.MolFromSmiles(s) is None]
    assert not invalid, f"SMILES de référence non parsables : {invalid}"


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
@given(smiles=st.sampled_from(REFERENCE_SMILES))
def test_standardize_is_idempotent(smiles: str) -> None:
    """``std(std(m)) == std(m)``.

    Une standardisation non idempotente produit des doublons invisibles : deux
    entrées identiques du jeu de données se retrouvent de part et d'autre de la
    partition parce qu'elles ont convergé vers deux formes canoniques
    différentes selon le nombre de passages.
    """
    once = standardize(smiles)
    assume(once is not None)
    assert once is not None  # affine le type pour la suite
    twice = standardize(once.smiles)
    assert twice is not None, "une molécule déjà standardisée a été rejetée au second passage"
    assert twice.smiles == once.smiles
    assert twice.inchikey == once.inchikey
    assert twice.inchikey_skeleton == once.inchikey_skeleton


@given(pair=equivalent_writings())
def test_standardize_is_invariant_to_smiles_writing(pair: tuple[str, str]) -> None:
    """Deux écritures de la même molécule donnent la même forme standardisée.

    C'est la propriété qui garantit que la déduplication par InChIKey fonctionne
    sur des sources hétérogènes : ChEMBL, BindingDB et Davis n'écrivent pas les
    mêmes SMILES pour les mêmes composés.
    """
    canonical, rewritten = pair
    a, b = standardize(canonical), standardize(rewritten)
    assume(a is not None)
    assert a is not None
    assert b is not None, f"écriture équivalente rejetée : {rewritten!r}"
    assert a.smiles == b.smiles
    assert a.inchikey == b.inchikey
    assert a.n_heavy_atoms == b.n_heavy_atoms


# --------------------------------------------------------------------------- #
# Featurisation
# --------------------------------------------------------------------------- #
@given(mol=molecules())
def test_graph_preserves_atom_and_bond_counts(mol: Chem.Mol) -> None:
    """Le graphe conserve la taille de la molécule, hydrogènes explicites compris.

    La convention PyG est bidirectionnelle : une liaison chimique donne deux
    arcs. Confondre les deux comptes fausse silencieusement toute normalisation
    par le degré dans les couches de message passing.
    """
    graph = mol_to_graph(mol)
    assert graph.num_nodes == mol.GetNumAtoms()
    assert tuple(graph.x.shape) == (mol.GetNumAtoms(), N_ATOM_FEATURES)
    assert tuple(graph.edge_index.shape) == (2, 2 * mol.GetNumBonds())
    assert tuple(graph.edge_attr.shape) == (2 * mol.GetNumBonds(), N_BOND_FEATURES)


@given(mol=molecules())
def test_feature_indices_stay_within_embedding_cardinality(mol: Chem.Mol) -> None:
    """Aucun index ne déborde de la table d'embedding qui le consommera.

    Un débordement produit un ``IndexError`` à l'exécution dans le meilleur cas,
    et dans le pire un index négatif silencieusement replié sur la fin de la
    table — donc deux motifs chimiques distincts partageant un embedding.
    """
    graph = mol_to_graph(mol)
    for col, dim in enumerate(ATOM_FEATURE_DIMS):
        column = graph.x[:, col]
        assert int(column.min()) >= 0, f"feature atomique {col} négative"
        assert int(column.max()) < dim, f"feature atomique {col} déborde de {dim}"
    if graph.edge_attr.numel():
        for col, dim in enumerate(BOND_FEATURE_DIMS):
            column = graph.edge_attr[:, col]
            assert int(column.min()) >= 0, f"feature de liaison {col} négative"
            assert int(column.max()) < dim, f"feature de liaison {col} déborde de {dim}"


@given(triple=renumbered_pairs())
def test_renumbering_atoms_permutes_the_graph_without_changing_it(
    triple: tuple[Chem.Mol, Chem.Mol, list[int]],
) -> None:
    """Renuméroter les atomes permute le graphe, ne le modifie pas.

    L'ordre des atomes dans un SMILES est un artefact d'écriture. Si la
    featurisation en dépend, deux enregistrements du même composé produisent
    deux représentations distinctes, et le modèle apprend l'artefact.
    """
    original, renumbered, order = triple
    g_old, g_new = mol_to_graph(original), mol_to_graph(renumbered)

    # `RenumberAtoms` garantit : atome_new[i] == atome_old[order[i]].
    for new_idx, old_idx in enumerate(order):
        assert (
            g_new.x[new_idx].tolist() == g_old.x[old_idx].tolist()
        ), f"features de l'atome {old_idx} modifiées par la renumérotation"

    inverse = [0] * len(order)
    for new_idx, old_idx in enumerate(order):
        inverse[old_idx] = new_idx

    def edge_map(graph: object, relabel: list[int] | None = None) -> dict[tuple[int, int], tuple]:
        src, dst = graph.edge_index.tolist()  # type: ignore[attr-defined]
        feats = graph.edge_attr.tolist()  # type: ignore[attr-defined]
        if relabel is None:
            return {(a, b): tuple(f) for a, b, f in zip(src, dst, feats, strict=True)}
        return {(relabel[a], relabel[b]): tuple(f) for a, b, f in zip(src, dst, feats, strict=True)}

    assert edge_map(g_old, inverse) == edge_map(g_new)


@given(mols=st.lists(molecules(explicit_hs=False), min_size=2, max_size=5))
def test_collation_keeps_every_edge_inside_a_single_molecule(mols: list[Chem.Mol]) -> None:
    """Après collation, aucune arête ne relie deux molécules du lot.

    Une seule arête inter-molécules suffit à faire circuler de l'information
    entre deux composés du batch : le modèle prédit alors en partie à partir de
    ses voisins de lot, ce qui ne se voit ni dans la loss ni dans les métriques
    de validation.
    """
    batch = Batch.from_data_list([mol_to_graph(m) for m in mols])
    src, dst = batch.edge_index
    assert bool((batch.batch[src] == batch.batch[dst]).all()), "arête traversant deux molécules"
    assert int(batch.num_nodes) == sum(m.GetNumAtoms() for m in mols)
