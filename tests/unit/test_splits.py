"""Invariants des splits. Ces assertions sont la garantie scientifique du projet :
si elles tombent, tous les chiffres du README sont invalides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, RDLogger

from kinoscope.data.splits import (
    audit_leakage,
    butina_clusters,
    butina_split,
    cold_pair_split,
    cold_target_split,
    murcko_scaffold,
    random_split,
    scaffold_split,
    temporal_split,
)

# Dix-sept systèmes cycliques distincts, chacun décliné en quatre longueurs de
# chaîne alkyle. Trois propriétés de cette liste sont délibérées.
#
# 1. Le squelette de Bemis-Murcko dépouille la chaîne alkyle : les quatre variantes
#    d'un cœur partagent donc un seul squelette. Le nombre de cœurs fixe le nombre
#    de groupes disponibles pour le remplissage glouton. Avec cinq cœurs seulement,
#    une partition 80/10/10 sur 300 lignes donnait train=240, valid=60, test=0 : le
#    test set était vide et les assertions de disjonction passaient à vide.
# 2. La quinoxaline est écrite depuis le cycle benzénique (`c1ccc2nccnc2c1`) et non
#    depuis l'azote (`c1cnc2ccccc2n1`). Les deux graphies désignent la même molécule
#    — InChIKey XSCHRSMBECNVNS-UHFFFAOYSA-N — mais la seconde place un azote
#    aromatique en dernière position : la concaténation du suffixe alkyle y produit
#    un N substitué non kékulisable, et `MolFromSmiles` renvoie `None`.
# 3. Cyclopentane, cyclohexane et cycloheptane non substitués ont des empreintes
#    ECFP4 **identiques** : tous leurs atomes présentent le même environnement à
#    rayon 2. Leur similarité de Tanimoto vaut 1.0 malgré des squelettes disjoints.
#    On les conserve : c'est précisément le phénomène que l'audit de fuite doit
#    savoir détecter, et l'illustration de la thèse du projet — la disjonction de
#    squelettes ne suffit pas à garantir un test set cold-drug.
_TOY_CORES = [
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "c1ccc2ccccc2c1",
    "c1ccc2nccnc2c1",
    "c1ccc2ncccc2c1",
    "c1ccsc1",
    "c1ccoc1",
    "C1CCNCC1",
    "C1CCOCC1",
    "c1ccc2c(c1)CCCC2",
    "C1CCCC1",
    "c1ccc2[nH]ccc2c1",
    "C1CCCCCC1",
    "c1ccc(-c2ccccc2)cc1",
    "C1CCNC1",
    "c1ccc2occc2c1",
]


@pytest.fixture
def toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    RDLogger.DisableLog("rdApp.*")
    for i in range(300):
        core = _TOY_CORES[i % len(_TOY_CORES)]
        smi = f"{core}" + "C" * (i % 4)
        # Garde-fou : un SMILES invalide dans la fixture ferait échouer les splits
        # pour une raison sans rapport avec la logique testée.
        assert Chem.MolFromSmiles(smi) is not None, f"SMILES de fixture invalide : {smi}"
        rows.append(
            {
                "smiles": smi,
                "inchikey": f"IK{i:05d}",
                "uniprot": f"P{i % 12:05d}",
                "manning_group": ["TK", "CMGC", "AGC", "CAMK", "STE", "CK1", "TKL"][i % 7],
                "p_affinity": float(rng.normal(7, 1.2)),
                "doc_year": int(rng.integers(2005, 2024)),
            }
        )
    return pd.DataFrame(rows)


def test_partition_is_exhaustive_and_disjoint(toy_df: pd.DataFrame) -> None:
    for split in (random_split(toy_df), scaffold_split(toy_df), cold_target_split(toy_df)):
        idx = np.concatenate([split.train, split.valid, split.test])
        assert len(idx) == len(np.unique(idx)), "index dupliqué entre partitions"
        assert set(idx) == set(range(len(toy_df))), "partition non exhaustive"


def test_scaffold_split_has_zero_scaffold_overlap(toy_df: pd.DataFrame) -> None:
    s = scaffold_split(toy_df)
    tr = {murcko_scaffold(x) for x in toy_df.iloc[s.train]["smiles"]}
    te = {murcko_scaffold(x) for x in toy_df.iloc[s.test]["smiles"]}
    assert tr.isdisjoint(te)


def test_cold_target_split_has_zero_target_overlap(toy_df: pd.DataFrame) -> None:
    s = cold_target_split(toy_df, by="manning_group")
    tr = set(toy_df.iloc[s.train]["manning_group"])
    te = set(toy_df.iloc[s.test]["manning_group"])
    assert tr.isdisjoint(te)


def test_temporal_split_respects_arrow_of_time(toy_df: pd.DataFrame) -> None:
    s = temporal_split(toy_df, valid_year=2018, test_year=2021)
    assert toy_df.iloc[s.train]["doc_year"].max() < toy_df.iloc[s.test]["doc_year"].min()


def test_random_split_leaks_and_scaffold_split_does_not(toy_df: pd.DataFrame) -> None:
    """Test de contrôle : documente positivement l'ampleur de la fuite du split
    aléatoire. C'est le résultat central du projet, il doit être testé."""
    leaky = audit_leakage(toy_df, random_split(toy_df), sample=200)
    clean = audit_leakage(toy_df, scaffold_split(toy_df), sample=200)
    assert leaky.shared_scaffolds > 0
    assert clean.shared_scaffolds == 0
    clean.assert_clean()


def test_splits_are_reproducible(toy_df: pd.DataFrame) -> None:
    a, b = scaffold_split(toy_df, seed=7), scaffold_split(toy_df, seed=7)
    assert np.array_equal(a.test, b.test)


def test_butina_never_mixes_two_disjoint_chemical_series() -> None:
    """`cutoff` est une *distance* : 0.6 correspond à une similarité Tanimoto de 0.4.

    La similarité inter-séries mesurée ici ne dépasse pas 0.11, très en dessous du
    seuil : aucun cluster ne doit chevaucher les deux séries. On n'exige pas que
    chaque série forme un cluster unique — l'ajout d'un méthyle sur un naphtalène
    nu ne suffit pas à atteindre 0.4 de similarité ECFP4.
    """
    serie_a = ["c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1CC"]
    serie_b = ["C1CCNCC1", "C1CCNCC1C", "C1CCNCC1CC"]
    labels = butina_clusters(serie_a + serie_b, cutoff=0.6)

    assert len(labels) == 6
    assert set(labels[:3]).isdisjoint(set(labels[3:])), "les deux séries se mélangent"


def test_butina_cluster_count_decreases_as_the_cutoff_widens() -> None:
    """Un seuil de distance plus permissif ne peut que fusionner des clusters."""
    smiles = [
        "c1ccc2ccccc2c1",
        "c1ccc2ccccc2c1C",
        "c1ccc2ccccc2c1CC",
        "C1CCNCC1",
        "C1CCNCC1C",
        "C1CCNCC1CC",
    ]
    counts = [len(set(butina_clusters(smiles, cutoff=c))) for c in (0.4, 0.6, 0.8)]
    assert counts[0] >= counts[1] >= counts[2]
    assert counts[0] > counts[2], "le seuil n'a aucun effet : clustering suspect"


def test_butina_split_partitions_exhaustively(toy_df: pd.DataFrame) -> None:
    split = butina_split(toy_df)
    idx = np.concatenate([split.train, split.valid, split.test])
    assert set(idx) == set(range(len(toy_df)))
    assert len(idx) == len(np.unique(idx))


def test_cold_pair_split_is_included_in_both_parent_splits(toy_df: pd.DataFrame) -> None:
    """Le split cold-pair est l'intersection du cold-drug et du cold-target :
    chacune de ses partitions est incluse dans celle des deux splits parents."""
    pair = cold_pair_split(toy_df)
    drug = scaffold_split(toy_df)
    target = cold_target_split(toy_df)
    for name in ("train", "valid", "test"):
        got = set(getattr(pair, name).tolist())
        assert got <= set(getattr(drug, name).tolist())
        assert got <= set(getattr(target, name).tolist())


def test_cold_target_split_rejects_an_unknown_grouping_column(toy_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="colonne de regroupement absente"):
        cold_target_split(toy_df, by="colonne_inexistante")


def test_audit_on_a_degenerate_partition_reports_nan_not_zero(toy_df: pd.DataFrame) -> None:
    """Un test set vide ne doit pas faire lever l'audit, ni renvoyer 0.0 —
    valeur qui se lirait à tort comme « aucune fuite détectée »."""
    from kinoscope.data.splits import Split

    empty = Split(
        train=np.arange(len(toy_df), dtype=np.int64),
        valid=np.array([], dtype=np.int64),
        test=np.array([], dtype=np.int64),
    )
    report = audit_leakage(toy_df, empty, sample=50)
    assert np.isnan(report.max_tanimoto_train_test)
    assert np.isnan(report.frac_test_above_0_7)
    assert report.n_test == 0
