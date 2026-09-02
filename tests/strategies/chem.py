"""Hypothesis strategies for generating molecules and affinity datasets.

Generating SMILES character by character is the wrong reflex: almost every string
drawn is invalid, and the strategy spends its time in the filter rather than in
chemical space. We work the other way round — a pool of real molecules, from
which we draw *transformations that preserve chemical identity*:

  - non-canonical SMILES rewriting (traversal rooted at an arbitrary atom);
  - random atom renumbering (``Chem.RenumberAtoms``);
  - adding or removing explicit hydrogens.

Those three transformations are exactly the ones a correct featuriser must absorb
without changing its output up to isomorphism, and that a standardisation pipeline
must reduce to a single form.

The pool covers the cases that break this kind of code: salts and counter-ions, a
tautomer pair, enantiomers, a macrocycle, a metal complex, formal charges,
heavy halogens, and nitrogen- and sulfur-containing heterocycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Real molecules. The validity of each is locked down by
# ``test_reference_pool_is_parsable``: an invalid entry would fail the property
# tests below for a reason unrelated to the property under test.
REFERENCE_SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # imatinib
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine: enantiomer of the above
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose: four stereocentres
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",  # indole: aromatic NH
    "c1ccc2nccnc2c1",
    "c1ccsc1",
    "c1ccoc1",
    "C1CCNCC1",
    "C1COCCN1",
    "C1CCCCC1",
    "CC(=O)[O-].[Na+]",  # salt: two fragments
    "CN(C)CCc1c[nH]c2ccccc12.Cl",  # hydrochloride
    "O=c1cccc[nH]1",  # 2-pyridone     \ tautomer pair:
    "Oc1ccccn1",  # 2-hydroxypyridine  / same formula, distinct graphs
    "C1CCCCCCCCCCCCCCC1",  # 16-membered macrocycle
    "O=[N+]([O-])c1ccccc1",  # nitro: explicit formal charges
    "NS(=O)(=O)c1ccccc1",
    "CC(=O)[O-].[Zn+2].CC(=O)[O-]",  # metal: rejected downstream, not upstream
    "OP(=O)(O)OCc1ccccc1",
    "Fc1ccc(Cl)c(Br)c1",  # heavy halogens
    "C#Cc1ccccc1",  # triple bond
    "C[N+](C)(C)C.[Cl-]",  # quaternary ammonium
    "C/C=C/c1ccccc1",  # double-bond stereochemistry (E)
    "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",  # EGFR-like kinase inhibitor
)

# InChIKeys are computed once: recomputing them for every drawn example would
# dominate the runtime of the split tests.
_INCHIKEYS: dict[str, str] = {
    smi: Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in REFERENCE_SMILES
}

MANNING_GROUPS: tuple[str, ...] = ("TK", "CMGC", "AGC", "CAMK", "STE", "CK1", "TKL", "atypical")


def parse(smiles: str) -> Chem.Mol:
    """``MolFromSmiles`` that raises instead of silently returning ``None``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable reference-pool SMILES: {smiles!r}")
    return mol


reference_smiles = st.sampled_from(REFERENCE_SMILES)


@st.composite
def molecules(draw: st.DrawFn, *, explicit_hs: bool = True) -> Chem.Mol:
    """A molecule from the pool, with atoms randomly renumbered.

    ``explicit_hs`` allows drawing an ``AddHs`` variant: the graph then changes
    size without changing molecule, which is exactly the edge case where a
    featuriser confuses "number of atoms" with "number of heavy atoms".
    """
    mol = parse(draw(reference_smiles))
    if explicit_hs and draw(st.booleans()):
        mol = Chem.AddHs(mol)
    order = list(draw(st.permutations(range(mol.GetNumAtoms()))))
    return Chem.RenumberAtoms(mol, order)


@st.composite
def renumbered_pairs(draw: st.DrawFn) -> tuple[Chem.Mol, Chem.Mol, list[int]]:
    """``(mol, renumbered_mol, order)`` with ``atom_new[i] == atom_old[order[i]]``."""
    mol = draw(molecules())
    order = list(draw(st.permutations(range(mol.GetNumAtoms()))))
    return mol, Chem.RenumberAtoms(mol, order), order


@st.composite
def equivalent_writings(draw: st.DrawFn) -> tuple[str, str]:
    """Two SMILES writings of the same molecule, one of them non-canonical.

    ``rootedAtAtom`` is bounded not by the molecule's atom count but by that of
    its *first fragment*: on a salt, RDKit raises ``RuntimeError: Range Error`` as
    soon as the index exceeds that size. Verified on RDKit 2024.03.5 —
    ``CC(=O)[O-].[Na+]`` (fragments of 4 and 1 atoms) only accepts roots 0 to 3.
    Drawing the root over the whole molecule made this strategy flaky: it hit the
    faulty case only once in several dozen examples.
    """
    base = draw(reference_smiles)
    mol = parse(base)
    first_fragment = Chem.GetMolFrags(mol, asMols=True)[0]
    root = draw(st.integers(min_value=0, max_value=first_fragment.GetNumAtoms() - 1))
    return base, Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=root)


@st.composite
def affinity_frames(
    draw: st.DrawFn, *, min_rows: int = 40, max_rows: int = 200, max_targets: int = 12
) -> pd.DataFrame:
    """A (ligand, target, affinity) dataset of the shape ``data.splits`` expects.

    The columns reproduce the module's real contract: ``smiles``, ``inchikey``,
    ``uniprot``, ``manning_group``, ``p_affinity``, ``doc_year``. The InChIKey is
    the real one, computed from the SMILES, so two rows carrying the same ligand
    share the same key — without which the leakage audit would be testing an empty
    property.
    """
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_targets = draw(st.integers(min_value=2, max_value=max_targets))
    n_groups = draw(st.integers(min_value=2, max_value=len(MANNING_GROUPS)))

    smiles = draw(st.lists(reference_smiles, min_size=n, max_size=n))
    targets = draw(st.lists(st.integers(0, n_targets - 1), min_size=n, max_size=n))
    groups = draw(st.lists(st.integers(0, n_groups - 1), min_size=n, max_size=n))
    years = draw(st.lists(st.integers(2000, 2024), min_size=n, max_size=n))
    affinities = draw(st.lists(st.floats(4.0, 11.0, allow_nan=False), min_size=n, max_size=n))

    return pd.DataFrame(
        {
            "smiles": smiles,
            "inchikey": [_INCHIKEYS[s] for s in smiles],
            "uniprot": [f"P{t:05d}" for t in targets],
            "manning_group": [MANNING_GROUPS[g] for g in groups],
            "p_affinity": np.asarray(affinities, dtype=np.float64),
            "doc_year": np.asarray(years, dtype=np.int64),
        }
    )
