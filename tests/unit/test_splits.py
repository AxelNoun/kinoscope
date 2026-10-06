"""Split invariants. These assertions are the project's scientific guarantee: if
they fall, every figure in the README is invalid."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, RDLogger

from kinoscope.data.splits import (
    LeakageReport,
    LeakageThresholds,
    audit_leakage,
    butina_clusters,
    butina_split,
    cold_pair_split,
    cold_target_split,
    load_leakage_thresholds,
    murcko_scaffold,
    random_split,
    scaffold_split,
    temporal_split,
)

_SPLIT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "split"
_SPLIT_CONFIGS = sorted(_SPLIT_CONFIG_DIR.glob("*.yaml"))

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


# --------------------------------------------------------------------------- #
# Leakage gate: thresholds declared in configs/split/*.yaml
# --------------------------------------------------------------------------- #
_THRESHOLDS = LeakageThresholds(
    max_shared_inchikeys=0,
    max_shared_uniprots=2,
    max_shared_scaffolds=0,
    max_tanimoto_train_test=0.85,
)


def _report_at(thresholds: LeakageThresholds) -> LeakageReport:
    """A report sitting exactly on every threshold: the gate must accept it."""
    return LeakageReport(
        n_train=80,
        n_valid=10,
        n_test=10,
        shared_inchikeys=thresholds.max_shared_inchikeys or 0,
        shared_uniprots=thresholds.max_shared_uniprots or 0,
        shared_scaffolds=thresholds.max_shared_scaffolds or 0,
        max_tanimoto_train_test=thresholds.max_tanimoto_train_test or 0.0,
        frac_test_above_0_7=0.0,
    )


def test_split_configs_are_found() -> None:
    """Guards the parametrisation below: an empty glob would skip it silently."""
    assert _SPLIT_CONFIGS, f"no split config under {_SPLIT_CONFIG_DIR}"


@pytest.mark.parametrize("path", _SPLIT_CONFIGS, ids=lambda p: p.stem)
def test_every_split_config_declares_valid_audit_thresholds(path: Path) -> None:
    thresholds = load_leakage_thresholds(path)
    _report_at(thresholds).assert_clean(thresholds=thresholds)


def test_scaffold_config_leaves_targets_unchecked_and_cold_target_does_not() -> None:
    """The one deliberate difference between the two configs: targets are shared by
    design in the cold-drug regime, forbidden in the cold-target one."""
    scaffold = load_leakage_thresholds(_SPLIT_CONFIG_DIR / "scaffold.yaml")
    cold_target = load_leakage_thresholds(_SPLIT_CONFIG_DIR / "cold_target.yaml")
    assert scaffold.max_shared_uniprots is None
    assert cold_target.max_shared_uniprots == 0


def test_gate_accepts_a_report_on_every_threshold() -> None:
    _report_at(_THRESHOLDS).assert_clean(thresholds=_THRESHOLDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shared_inchikeys", 1),
        ("shared_uniprots", 3),
        ("shared_scaffolds", 1),
        ("max_tanimoto_train_test", 0.86),
    ],
)
def test_gate_rejects_each_exceeded_threshold(field: str, value: float) -> None:
    report = replace(_report_at(_THRESHOLDS), **{field: value})
    threshold_name = field if field.startswith("max_") else f"max_{field}"
    with pytest.raises(AssertionError, match=f"above {threshold_name}="):
        report.assert_clean(thresholds=_THRESHOLDS)


def test_gate_lists_every_violation_at_once() -> None:
    report = replace(_report_at(_THRESHOLDS), shared_scaffolds=4, max_tanimoto_train_test=1.0)
    assert len(report.violations(_THRESHOLDS)) == 2


def test_null_threshold_disables_only_its_own_check() -> None:
    open_targets = replace(_THRESHOLDS, max_shared_uniprots=None)
    report = replace(_report_at(_THRESHOLDS), shared_uniprots=500)
    report.assert_clean(thresholds=open_targets)
    with pytest.raises(AssertionError, match="max_shared_scaffolds"):
        replace(report, shared_scaffolds=1).assert_clean(thresholds=open_targets)


def test_gate_fails_when_the_similarity_could_not_be_measured() -> None:
    """``audit_leakage`` reports NaN on an empty train or test partition. NaN compares
    False with everything, so a plain ``>`` would let it through the ceiling."""
    report = replace(_report_at(_THRESHOLDS), max_tanimoto_train_test=math.nan)
    with pytest.raises(AssertionError, match="cannot be verified"):
        report.assert_clean(thresholds=_THRESHOLDS)
    report.assert_clean(thresholds=replace(_THRESHOLDS, max_tanimoto_train_test=None))


def test_assert_clean_without_thresholds_keeps_the_historical_rules() -> None:
    report = replace(_report_at(_THRESHOLDS), shared_uniprots=3, max_tanimoto_train_test=1.0)
    report.assert_clean()
    with pytest.raises(AssertionError, match="targets shared"):
        report.assert_clean(allow_shared_targets=False)
    with pytest.raises(AssertionError, match="scaffolds shared"):
        replace(report, shared_scaffolds=1).assert_clean()


def test_threshold_keys_must_match_exactly() -> None:
    """A misspelt key must raise, not leave its check silently disabled."""
    misspelt = asdict(_THRESHOLDS)
    misspelt["max_shared_scaffold"] = misspelt.pop("max_shared_scaffolds")
    with pytest.raises(ValueError, match="max_shared_scaffold"):
        LeakageThresholds.from_mapping(misspelt)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_shared_inchikeys", -1),
        ("max_shared_inchikeys", True),
        ("max_shared_scaffolds", 0.5),
        ("max_tanimoto_train_test", 1.5),
        ("max_tanimoto_train_test", "0.85"),
    ],
)
def test_threshold_values_are_validated(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        LeakageThresholds.from_mapping({**asdict(_THRESHOLDS), key: value})


def test_config_without_an_audit_block_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "no_audit.yaml"
    config.write_text("name: scaffold\nstrategy: scaffold\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
        load_leakage_thresholds(config)


def test_configured_gate_rejects_the_random_split(toy_df: pd.DataFrame) -> None:
    """The gate tested in failure, with the thresholds actually declared for the
    default regime: a random split shares scaffolds between train and test."""
    thresholds = load_leakage_thresholds(_SPLIT_CONFIG_DIR / "scaffold.yaml")
    report = audit_leakage(toy_df, random_split(toy_df), sample=200)
    with pytest.raises(AssertionError, match="above max_shared_scaffolds="):
        report.assert_clean(thresholds=thresholds)


def test_configured_tanimoto_ceiling_catches_what_scaffold_disjointness_misses(
    toy_df: pd.DataFrame,
) -> None:
    """The fixture keeps cyclohexane and cycloheptane, whose ECFP4 fingerprints are
    identical. At the default seed the cycloheptane series lands in test and the
    cyclohexane series in train: no scaffold is shared, yet the nearest-neighbour
    similarity is 1.0. The historical rules pass this split; the configured ceiling
    does not.
    """
    thresholds = load_leakage_thresholds(_SPLIT_CONFIG_DIR / "scaffold.yaml")
    report = audit_leakage(toy_df, scaffold_split(toy_df), sample=200)
    report.assert_clean()
    assert report.violations(thresholds) == [
        "nearest-neighbour Tanimoto 1.000, above max_tanimoto_train_test=0.85"
    ]
