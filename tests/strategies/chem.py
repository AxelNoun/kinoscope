"""Stratégies Hypothesis pour la génération de molécules et de jeux d'affinité.

Générer des SMILES caractère par caractère est un mauvais réflexe : la quasi-
totalité des chaînes tirées est invalide, et la stratégie passe son temps dans
le filtre plutôt que dans l'espace chimique. On procède à l'envers — un pool de
molécules réelles, dont on tire des *transformations qui préservent l'identité
chimique* :

  - réécriture SMILES non canonique (parcours enraciné sur un atome arbitraire) ;
  - renumérotation aléatoire des atomes (``Chem.RenumberAtoms``) ;
  - ajout ou retrait des hydrogènes explicites.

Ces trois transformations sont précisément celles qu'un featurizer correct doit
absorber sans changer sa sortie à isomorphisme près, et qu'un pipeline de
standardisation doit réduire à une forme unique.

Le pool couvre les cas qui font tomber ce genre de code : sels et contre-ions,
paire de tautomères, énantiomères, macrocycle, composé métallique, charges
formelles portées, halogènes lourds, hétérocycles azotés et soufrés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Molécules réelles. La validité de chacune est verrouillée par
# ``test_reference_pool_is_parsable`` : une entrée invalide ferait échouer les
# tests par propriétés pour une raison sans rapport avec la propriété testée.
REFERENCE_SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirine
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caféine
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # imatinib
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofène
    "CC(=O)Nc1ccc(O)cc1",  # paracétamol
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine : énantiomère du précédent
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose : quatre stéréocentres
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",  # indole : NH aromatique
    "c1ccc2nccnc2c1",
    "c1ccsc1",
    "c1ccoc1",
    "C1CCNCC1",
    "C1COCCN1",
    "C1CCCCC1",
    "CC(=O)[O-].[Na+]",  # sel : deux fragments
    "CN(C)CCc1c[nH]c2ccccc12.Cl",  # chlorhydrate
    "O=c1cccc[nH]1",  # 2-pyridone     \ paire de tautomères :
    "Oc1ccccn1",  # 2-hydroxypyridine  / même formule, graphes distincts
    "C1CCCCCCCCCCCCCCC1",  # macrocycle à 16 chaînons
    "O=[N+]([O-])c1ccccc1",  # nitro : charges formelles portées
    "NS(=O)(=O)c1ccccc1",
    "CC(=O)[O-].[Zn+2].CC(=O)[O-]",  # métal : rejeté en aval, pas en amont
    "OP(=O)(O)OCc1ccccc1",
    "Fc1ccc(Cl)c(Br)c1",  # halogènes lourds
    "C#Cc1ccccc1",  # liaison triple
    "C[N+](C)(C)C.[Cl-]",  # ammonium quaternaire
    "C/C=C/c1ccccc1",  # stéréochimie de double liaison (E)
    "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",  # inhibiteur de kinase type EGFR
)

# Les InChIKey sont calculés une fois pour toutes : les recalculer à chaque
# exemple tiré dominerait le temps d'exécution des tests de splits.
_INCHIKEYS: dict[str, str] = {
    smi: Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in REFERENCE_SMILES
}

MANNING_GROUPS: tuple[str, ...] = ("TK", "CMGC", "AGC", "CAMK", "STE", "CK1", "TKL", "atypical")


def parse(smiles: str) -> Chem.Mol:
    """``MolFromSmiles`` qui lève au lieu de renvoyer ``None`` silencieusement."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES du pool de référence non parsable : {smiles!r}")
    return mol


reference_smiles = st.sampled_from(REFERENCE_SMILES)


@st.composite
def molecules(draw: st.DrawFn, *, explicit_hs: bool = True) -> Chem.Mol:
    """Une molécule du pool, atomes renumérotés au hasard.

    ``explicit_hs`` autorise le tirage d'une variante ``AddHs`` : le graphe
    change alors de taille sans changer de molécule, ce qui est exactement le
    cas limite où un featurizer confond « nombre d'atomes » et « nombre
    d'atomes lourds ».
    """
    mol = parse(draw(reference_smiles))
    if explicit_hs and draw(st.booleans()):
        mol = Chem.AddHs(mol)
    order = list(draw(st.permutations(range(mol.GetNumAtoms()))))
    return Chem.RenumberAtoms(mol, order)


@st.composite
def renumbered_pairs(draw: st.DrawFn) -> tuple[Chem.Mol, Chem.Mol, list[int]]:
    """``(mol, mol_renumérotée, ordre)`` avec ``atome_new[i] == atome_old[ordre[i]]``."""
    mol = draw(molecules())
    order = list(draw(st.permutations(range(mol.GetNumAtoms()))))
    return mol, Chem.RenumberAtoms(mol, order), order


@st.composite
def equivalent_writings(draw: st.DrawFn) -> tuple[str, str]:
    """Deux écritures SMILES de la même molécule, dont une non canonique.

    ``rootedAtAtom`` n'est pas borné par le nombre d'atomes de la molécule mais
    par celui de son *premier fragment* : sur un sel, RDKit lève
    ``RuntimeError: Range Error`` dès que l'index dépasse cette taille. Vérifié
    sur RDKit 2024.03.5 — ``CC(=O)[O-].[Na+]`` (fragments 4 et 1) n'accepte que
    les racines 0 à 3. Tirer la racine sur toute la molécule rendait cette
    stratégie instable : elle ne tombait sur le cas fautif qu'une fois sur
    plusieurs dizaines d'exemples.
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
    """Un jeu (ligand, cible, affinité) de la forme attendue par ``data.splits``.

    Les colonnes reproduisent le contrat réel du module : ``smiles``,
    ``inchikey``, ``uniprot``, ``manning_group``, ``p_affinity``, ``doc_year``.
    L'InChIKey est le vrai, calculé depuis le SMILES : deux lignes portant le
    même ligand partagent donc la même clé, sans quoi l'audit de fuite testerait
    une propriété vide.
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
