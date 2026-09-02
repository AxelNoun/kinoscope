"""Tests par propriétés des stratégies de partitionnement.

``test_splits.py`` vérifie ces invariants sur une fixture unique de 300 lignes.
Le risque de cette approche est connu : une fixture qui, par construction,
n'exerce jamais le cas dégénéré (un seul groupe, groupes de tailles très
inégales, années toutes identiques) laisse passer un bug qui n'apparaîtra que
sur les vraies données ChEMBL. Les propriétés ci-dessous sont vérifiées sur des
jeux tirés dont la taille, le nombre de cibles et le nombre de groupes varient.

Les tests qui appellent ``scaffold_split``, ``butina_split`` ou ``audit_leakage``
portent un ``max_examples`` explicite : ces fonctions recalculent des
descripteurs RDKit pour chaque ligne à chaque exemple, et le profil ``ci`` les
rendrait sinon plus longs que tout le reste de la suite réuni.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from rdkit import Chem

from kinoscope.data.splits import (
    Split,
    audit_leakage,
    butina_split,
    cold_pair_split,
    cold_target_split,
    murcko_scaffold,
    random_split,
    scaffold_split,
    temporal_split,
)
from tests.strategies.chem import MANNING_GROUPS, REFERENCE_SMILES, affinity_frames

_VALID_YEAR, _TEST_YEAR = 2012, 2019


def _fixed_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Jeu déterministe construit hors Hypothesis.

    Réservé aux tests dont l'issue ne dépend pas de la variabilité du jeu : y
    faire tourner un moteur de génération et de réduction coûterait du temps
    sans rien apporter.
    """
    rng = np.random.default_rng(seed)
    smiles = [REFERENCE_SMILES[i % len(REFERENCE_SMILES)] for i in range(n)]
    return pd.DataFrame(
        {
            "smiles": smiles,
            "inchikey": [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in smiles],
            "uniprot": [f"P{i % 9:05d}" for i in range(n)],
            "manning_group": [MANNING_GROUPS[i % len(MANNING_GROUPS)] for i in range(n)],
            "p_affinity": rng.normal(7.0, 1.2, size=n),
            "doc_year": rng.integers(2005, 2024, size=n),
        }
    )


def _assert_partitions(split: Split, n_rows: int) -> None:
    """Exhaustivité et disjonction : toute ligne dans exactement une partition."""
    joined = np.concatenate([split.train, split.valid, split.test])
    assert len(joined) == len(np.unique(joined)), "index présent dans deux partitions"
    assert set(joined.tolist()) == set(range(n_rows)), "partition non exhaustive"


# --------------------------------------------------------------------------- #
# Structure de la partition
# --------------------------------------------------------------------------- #
@settings(max_examples=30)
@given(df=affinity_frames(max_rows=120))
def test_partitioning_strategies_are_exhaustive_and_disjoint(df: pd.DataFrame) -> None:
    for split in (
        random_split(df),
        scaffold_split(df),
        cold_target_split(df),
        temporal_split(df, valid_year=_VALID_YEAR, test_year=_TEST_YEAR),
    ):
        _assert_partitions(split, len(df))


@settings(max_examples=15)
@given(df=affinity_frames(min_rows=20, max_rows=60))
def test_butina_split_is_exhaustive_and_disjoint(df: pd.DataFrame) -> None:
    """Séparé des autres : le clustering Butina est quadratique en nombre de
    composés, il ne supporte pas les jeux tirés de la même taille."""
    _assert_partitions(butina_split(df), len(df))


@settings(max_examples=30)
@given(df=affinity_frames(max_rows=120))
def test_cold_pair_is_disjoint_but_deliberately_not_exhaustive(df: pd.DataFrame) -> None:
    """``cold_pair`` est une *intersection* de deux partitions, pas une partition.

    Les paires dont le ligand tombe en train et la cible en test sont écartées :
    c'est voulu — elles ne sont ni entraînables ni évaluables dans ce régime.
    La conséquence, en revanche, doit rester consciente : la somme des trois
    tailles est inférieure au nombre de lignes, et un ratio 80/10/10 annoncé sur
    ce régime ne décrit pas ce que le modèle voit réellement.
    """
    split = cold_pair_split(df)
    joined = np.concatenate([split.train, split.valid, split.test])
    assert len(joined) == len(np.unique(joined)), "index présent dans deux partitions"
    assert set(joined.tolist()) <= set(range(len(df))), "index hors du jeu de données"


# --------------------------------------------------------------------------- #
# Absence de fuite, par régime
# --------------------------------------------------------------------------- #
@settings(max_examples=30)
@given(df=affinity_frames(max_rows=120))
def test_scaffold_split_never_shares_a_scaffold(df: pd.DataFrame) -> None:
    split = scaffold_split(df)
    assume(len(split.train) > 0 and len(split.test) > 0)
    train = {murcko_scaffold(s) for s in df.iloc[split.train]["smiles"]}
    test = {murcko_scaffold(s) for s in df.iloc[split.test]["smiles"]}
    assert train.isdisjoint(test), f"squelettes partagés : {sorted(train & test)}"


@settings(max_examples=30)
@pytest.mark.parametrize("by", ["manning_group", "uniprot"])
@given(df=affinity_frames(max_rows=120))
def test_cold_target_split_never_shares_a_group(df: pd.DataFrame, by: str) -> None:
    """Vérifié sur les deux regroupements que le module expose : la famille de
    Manning (leave-group-out, le régime dur) et l'identifiant UniProt."""
    split = cold_target_split(df, by=by)
    assume(len(split.train) > 0 and len(split.test) > 0)
    train = set(df.iloc[split.train][by])
    test = set(df.iloc[split.test][by])
    assert train.isdisjoint(test), f"groupes partagés sur {by} : {sorted(train & test)}"


@settings(max_examples=30)
@given(df=affinity_frames(max_rows=120))
def test_temporal_split_respects_the_arrow_of_time(df: pd.DataFrame) -> None:
    split = temporal_split(df, valid_year=_VALID_YEAR, test_year=_TEST_YEAR)
    assume(len(split.train) > 0 and len(split.test) > 0)
    assert df.iloc[split.train]["doc_year"].max() < df.iloc[split.test]["doc_year"].min()


@settings(max_examples=15)
@given(df=affinity_frames(min_rows=20, max_rows=60))
def test_leakage_audit_confirms_scaffold_split_is_clean(df: pd.DataFrame) -> None:
    """L'audit de fuite et la stratégie de split doivent être d'accord.

    Ce sont deux implémentations indépendantes de la même notion : si elles
    divergent, l'une des deux est fausse et le chiffre publié l'est aussi.
    """
    split = scaffold_split(df)
    assume(len(split.train) > 0 and len(split.test) > 0)
    report = audit_leakage(df, split, sample=50)
    assert report.shared_scaffolds == 0
    assert report.n_train + report.n_valid + report.n_test == len(df)


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
@settings(max_examples=30)
@given(df=affinity_frames(max_rows=120))
def test_splits_are_reproducible_at_fixed_seed(df: pd.DataFrame) -> None:
    for factory in (random_split, scaffold_split, cold_target_split):
        first, second = factory(df, seed=7), factory(df, seed=7)
        assert np.array_equal(first.train, second.train)
        assert np.array_equal(first.valid, second.valid)
        assert np.array_equal(first.test, second.test)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Défaut identifié, correction non décidée : `_group_split` trie chaque "
        "compartiment avant de le renvoyer, ce qui annule le `rng.shuffle` et rend "
        "`seed` sans effet sur tous les splits par groupe (scaffold, butina, "
        "cold_target, cold_pair). Conséquence : impossible de tirer des splits "
        "répliqués, donc tout IC calculé sur plusieurs graines pour ces régimes "
        "est artificiellement étroit. Retirer ce marqueur une fois tranché."
    ),
)
def test_group_splits_should_depend_on_the_seed() -> None:
    """Délibérément hors Hypothesis : l'assertion échoue par construction, et la
    phase de réduction chercherait un contre-exemple minimal pour un défaut déjà
    localisé — près de trois minutes de calcul pour aucune information de plus."""
    df = _fixed_frame()
    a, b = scaffold_split(df, seed=1), scaffold_split(df, seed=999)
    assert not np.array_equal(a.test, b.test)
