"""Invariants des splits. Ces assertions sont la garantie scientifique du projet :
si elles tombent, tous les chiffres du README sont invalides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinoscope.data.splits import (
    audit_leakage,
    cold_target_split,
    murcko_scaffold,
    random_split,
    scaffold_split,
    temporal_split,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cores = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1cnc2ccccc2n1"]
    rows = []
    for i in range(300):
        core = cores[i % len(cores)]
        smi = f"{core}" + "C" * (i % 4)
        rows.append({
            "smiles": smi,
            "inchikey": f"IK{i:05d}",
            "uniprot": f"P{i % 12:05d}",
            "manning_group": ["TK", "CMGC", "AGC", "CAMK", "STE", "CK1", "TKL"][i % 7],
            "p_affinity": float(rng.normal(7, 1.2)),
            "doc_year": int(rng.integers(2005, 2024)),
        })
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
