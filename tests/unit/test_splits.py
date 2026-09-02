"""Split invariants. These assertions are the project's scientific guarantee: if
they fall, every figure in the README is invalid."""

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

# Seventeen distinct ring systems, each declined in four alkyl-chain lengths.
# Three properties of this list are deliberate.
#
# 1. The Bemis-Murcko scaffold strips the alkyl chain, so the four variants of one
#    core share a single scaffold. The number of cores therefore fixes the number
#    of groups available to greedy bin-packing. With only five cores, an 80/10/10
#    split over 300 rows produced train=240, valid=60, test=0: the test set was
#    empty and the disjointness assertions passed vacuously.
# 2. Quinoxaline is written from the benzene ring (`c1ccc2nccnc2c1`) rather than
#    from the nitrogen (`c1cnc2ccccc2n1`). Both spellings denote the same molecule
#    — InChIKey XSCHRSMBECNVNS-UHFFFAOYSA-N — but the second puts an aromatic
#    nitrogen last, so appending the alkyl suffix yields a non-kekulisable
#    substituted N and `MolFromSmiles` returns `None`.
# 3. Unsubstituted cyclopentane, cyclohexane and cycloheptane have **identical**
#    ECFP4 fingerprints: every one of their atoms has the same radius-2
#    environment. Their Tanimoto similarity is 1.0 despite disjoint scaffolds.
#    They are kept on purpose: this is exactly the phenomenon a leakage audit must
#    catch, and the illustration of the project's claim — scaffold disjointness
#    does not guarantee a cold-drug test set.
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
        # Guard: an invalid SMILES in the fixture would make the splits fail for a
        # reason unrelated to the logic under test.
        assert Chem.MolFromSmiles(smi) is not None, f"invalid fixture SMILES: {smi}"
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
        assert len(idx) == len(np.unique(idx)), "index duplicated across partitions"
        assert set(idx) == set(range(len(toy_df))), "partition is not exhaustive"


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
    """Control test: positively documents how much the random split leaks. This is
    the project's central result, so it must be tested."""
    leaky = audit_leakage(toy_df, random_split(toy_df), sample=200)
    clean = audit_leakage(toy_df, scaffold_split(toy_df), sample=200)
    assert leaky.shared_scaffolds > 0
    assert clean.shared_scaffolds == 0
    clean.assert_clean()


def test_splits_are_reproducible(toy_df: pd.DataFrame) -> None:
    a, b = scaffold_split(toy_df, seed=7), scaffold_split(toy_df, seed=7)
    assert np.array_equal(a.test, b.test)


def test_butina_never_mixes_two_disjoint_chemical_series() -> None:
    """`cutoff` is a *distance*: 0.6 corresponds to a Tanimoto similarity of 0.4.

    The cross-series similarity measured here never exceeds 0.11, well below the
    threshold: no cluster may straddle the two series. We do not require each
    series to form a single cluster — adding a methyl to a bare naphthalene is not
    enough to reach 0.4 ECFP4 similarity.
    """
    serie_a = ["c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1CC"]
    serie_b = ["C1CCNCC1", "C1CCNCC1C", "C1CCNCC1CC"]
    labels = butina_clusters(serie_a + serie_b, cutoff=0.6)

    assert len(labels) == 6
    assert set(labels[:3]).isdisjoint(set(labels[3:])), "the two series are mixed"


def test_butina_cluster_count_decreases_as_the_cutoff_widens() -> None:
    """A more permissive distance threshold can only merge clusters."""
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
    assert counts[0] > counts[2], "the threshold has no effect: clustering is suspect"


def test_butina_split_partitions_exhaustively(toy_df: pd.DataFrame) -> None:
    split = butina_split(toy_df)
    idx = np.concatenate([split.train, split.valid, split.test])
    assert set(idx) == set(range(len(toy_df)))
    assert len(idx) == len(np.unique(idx))


def test_cold_pair_split_is_included_in_both_parent_splits(toy_df: pd.DataFrame) -> None:
    """The cold-pair split is the intersection of cold-drug and cold-target: each of
    its partitions is contained in those of both parent splits."""
    pair = cold_pair_split(toy_df)
    drug = scaffold_split(toy_df)
    target = cold_target_split(toy_df)
    for name in ("train", "valid", "test"):
        got = set(getattr(pair, name).tolist())
        assert got <= set(getattr(drug, name).tolist())
        assert got <= set(getattr(target, name).tolist())


def test_cold_target_split_rejects_an_unknown_grouping_column(toy_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="grouping column not found"):
        cold_target_split(toy_df, by="colonne_inexistante")


def test_audit_on_a_degenerate_partition_reports_nan_not_zero(toy_df: pd.DataFrame) -> None:
    """An empty test set must not make the audit raise, nor return 0.0 — a value
    that would wrongly read as "no leakage detected"."""
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
