"""Property-based tests for the featuriser and the standardisation pipeline.

The example-based tests in ``test_featurizer.py`` cover the cases one thinks of.
These cover the class: each invariant is checked over hundreds of drawn and
transformed molecules, which is the only way to catch featurisation bugs that are
invisible by inspection — an index overflowing an embedding cardinality on a rare
motif, a featurisation that depends on SMILES writing order, an edge crossing two
molecules after collation.
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
    """Pool guard: an invalid entry would invalidate every test below without the
    failure pointing at the cause."""
    invalid = [s for s in REFERENCE_SMILES if Chem.MolFromSmiles(s) is None]
    assert not invalid, f"unparsable reference SMILES: {invalid}"


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
@given(smiles=st.sampled_from(REFERENCE_SMILES))
def test_standardize_is_idempotent(smiles: str) -> None:
    """``std(std(m)) == std(m)``.

    A non-idempotent standardisation produces invisible duplicates: two identical
    dataset entries end up on opposite sides of the partition because they
    converged to two different canonical forms depending on how many times they
    were processed.
    """
    once = standardize(smiles)
    assume(once is not None)
    assert once is not None  # narrows the type for what follows
    twice = standardize(once.smiles)
    assert twice is not None, "an already-standardised molecule was rejected on the second pass"
    assert twice.smiles == once.smiles
    assert twice.inchikey == once.inchikey
    assert twice.inchikey_skeleton == once.inchikey_skeleton


@given(pair=equivalent_writings())
def test_standardize_is_invariant_to_smiles_writing(pair: tuple[str, str]) -> None:
    """Two writings of the same molecule give the same standardised form.

    This is the property that makes InChIKey deduplication work across
    heterogeneous sources: ChEMBL, BindingDB and Davis do not write the same
    SMILES for the same compounds.
    """
    canonical, rewritten = pair
    a, b = standardize(canonical), standardize(rewritten)
    assume(a is not None)
    assert a is not None
    assert b is not None, f"equivalent writing rejected: {rewritten!r}"
    assert a.smiles == b.smiles
    assert a.inchikey == b.inchikey
    assert a.n_heavy_atoms == b.n_heavy_atoms


# --------------------------------------------------------------------------- #
# Featurisation
# --------------------------------------------------------------------------- #
@given(mol=molecules())
def test_graph_preserves_atom_and_bond_counts(mol: Chem.Mol) -> None:
    """The graph preserves the molecule's size, explicit hydrogens included.

    The PyG convention is bidirectional: one chemical bond yields two arcs.
    Confusing the two counts silently corrupts any degree normalisation in the
    message-passing layers.
    """
    graph = mol_to_graph(mol)
    assert graph.num_nodes == mol.GetNumAtoms()
    assert tuple(graph.x.shape) == (mol.GetNumAtoms(), N_ATOM_FEATURES)
    assert tuple(graph.edge_index.shape) == (2, 2 * mol.GetNumBonds())
    assert tuple(graph.edge_attr.shape) == (2 * mol.GetNumBonds(), N_BOND_FEATURES)


@given(mol=molecules())
def test_feature_indices_stay_within_embedding_cardinality(mol: Chem.Mol) -> None:
    """No index overflows the embedding table that will consume it.

    An overflow yields a runtime ``IndexError`` at best, and at worst a negative
    index silently wrapping to the end of the table — so two distinct chemical
    motifs sharing one embedding.
    """
    graph = mol_to_graph(mol)
    for col, dim in enumerate(ATOM_FEATURE_DIMS):
        column = graph.x[:, col]
        assert int(column.min()) >= 0, f"atom feature {col} is negative"
        assert int(column.max()) < dim, f"atom feature {col} overflows {dim}"
    if graph.edge_attr.numel():
        for col, dim in enumerate(BOND_FEATURE_DIMS):
            column = graph.edge_attr[:, col]
            assert int(column.min()) >= 0, f"bond feature {col} is negative"
            assert int(column.max()) < dim, f"bond feature {col} overflows {dim}"


@given(triple=renumbered_pairs())
def test_renumbering_atoms_permutes_the_graph_without_changing_it(
    triple: tuple[Chem.Mol, Chem.Mol, list[int]],
) -> None:
    """Renumbering atoms permutes the graph; it does not modify it.

    Atom order in a SMILES is a writing artefact. If featurisation depends on it,
    two records of the same compound yield two distinct representations and the
    model learns the artefact.
    """
    original, renumbered, order = triple
    g_old, g_new = mol_to_graph(original), mol_to_graph(renumbered)

    # `RenumberAtoms` guarantees: atom_new[i] == atom_old[order[i]].
    for new_idx, old_idx in enumerate(order):
        assert (
            g_new.x[new_idx].tolist() == g_old.x[old_idx].tolist()
        ), f"features of atom {old_idx} changed by renumbering"

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
    """After collation, no edge links two molecules of the batch.

    A single cross-molecule edge is enough to circulate information between two
    compounds in a batch: the model then predicts partly from its batch
    neighbours, which shows up neither in the loss nor in the validation metrics.
    """
    batch = Batch.from_data_list([mol_to_graph(m) for m in mols])
    src, dst = batch.edge_index
    assert bool((batch.batch[src] == batch.batch[dst]).all()), "edge crossing two molecules"
    assert int(batch.num_nodes) == sum(m.GetNumAtoms() for m in mols)
