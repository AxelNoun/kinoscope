"""Split strategies and data-leakage audit.

This is the central module of the project: KinoScope's scientific claim is that
published DTI performance collapses as soon as the split forbids the model from
memorising an already-seen ligand or target.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

logger = logging.getLogger(__name__)

SplitName = Literal["random", "scaffold", "butina", "cold_target", "cold_pair", "temporal"]
_MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class Split:
    train: npt.NDArray[np.int64]
    valid: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]

    def as_dict(self) -> dict[str, list[int]]:
        return {k: getattr(self, k).tolist() for k in ("train", "valid", "test")}


# --------------------------------------------------------------------------- #
# Fingerprints & scaffolds
# --------------------------------------------------------------------------- #
def morgan_fingerprints(smiles: list[str]) -> list[Any]:
    return [_MFPGEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def murcko_scaffold(smiles: str, *, include_chirality: bool = False) -> str:
    return cast(
        str,
        MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality),
    )


def butina_clusters(smiles: list[str], cutoff: float = 0.6) -> list[int]:
    """Butina clustering on ECFP4 Tanimoto distance.

    ``cutoff`` is a *distance*: 0.6 corresponds to a Tanimoto similarity of 0.4,
    the threshold above which two compounds are generally taken to belong to the
    same chemical series.
    """
    fps = morgan_fingerprints(smiles)
    n = len(fps)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = [0] * n
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def _group_split(
    groups: list[int] | list[str], frac: tuple[float, float, float], seed: int
) -> Split:
    """Partition whole *groups*, largest first (greedy bin-packing).

    Greedy filling prevents a massive cluster from unbalancing the test set, which
    is what a plain shuffle of groups does.

    ``seed`` breaks ties between groups of equal size, and that is the only degree
    of freedom the greedy filling can be given: the size ordering itself must stay
    deterministic, otherwise the guarantee above is lost. Shuffling the group list
    before a *stable* sort keeps largest-first while randomising equal-sized
    groups.

    This matters scientifically, not cosmetically. Until this was fixed, each
    bucket was shuffled and then sorted — which annuls the shuffle — and the
    group-to-bucket assignment was otherwise fully deterministic, so every
    group-based strategy silently ignored its ``seed``. A confidence interval
    computed over several seeds then aggregated identical replicates and was
    artificially narrow: it measured the variability of the model initialisation
    while being reported as covering the variability of the split as well.
    """
    rng = np.random.default_rng(seed)
    members: dict[object, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        members[g].append(idx)

    # Shuffle by index permutation rather than `rng.shuffle`: the group labels are
    # heterogeneous (int or str), so the list is `list[object]`, which the NumPy
    # generator does not accept.
    order = [key for _, key in sorted(zip(rng.permutation(len(members)), members, strict=True))]
    order.sort(key=lambda g: len(members[g]), reverse=True)  # stable sort: size wins

    n = len(groups)
    targets = [frac[0] * n, frac[1] * n, frac[2] * n]
    buckets: list[list[int]] = [[], [], []]

    for g in order:
        deficits = [targets[k] - len(buckets[k]) for k in range(3)]
        k = int(np.argmax(deficits))
        buckets[k].extend(members[g])

    # Indices are returned sorted: a split is a set, and a stable order keeps diffs
    # between runs readable. The previous `rng.shuffle(b)` on this line was dead —
    # the sort erased it — and was the reason `seed` had no observable effect.
    return Split(*[np.asarray(sorted(b), dtype=np.int64) for b in buckets])


def random_split(
    df: pd.DataFrame, frac: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> Split:
    """Optimistic control. Used solely to quantify how large the leakage is."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_tr, n_va = int(frac[0] * len(df)), int(frac[1] * len(df))
    return Split(np.sort(idx[:n_tr]), np.sort(idx[n_tr : n_tr + n_va]), np.sort(idx[n_tr + n_va :]))


def scaffold_split(
    df: pd.DataFrame, frac: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> Split:
    """Cold-drug by Bemis-Murcko scaffold."""
    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    return _group_split(scaffolds, frac, seed)


def butina_split(
    df: pd.DataFrame,
    cutoff: float = 0.6,
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Split:
    """Stricter cold-drug: similarity clusters, not just scaffolds."""
    return _group_split(butina_clusters(list(df["smiles"]), cutoff), frac, seed)


def cold_target_split(
    df: pd.DataFrame,
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    *,
    by: str = "manning_group",
) -> Split:
    """Cold-target. ``by='manning_group'`` is leave-group-out, the hardest regime;
    ``by='uniprot'`` gives unseen targets but possibly already-seen families."""
    if by not in df.columns:
        raise KeyError(f"grouping column not found: {by}")
    return _group_split(list(df[by]), frac, seed)


def cold_pair_split(
    df: pd.DataFrame, frac: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> Split:
    """Cold-drug x cold-target intersection: the only genuinely prospective regime."""
    drug = scaffold_split(df, frac, seed)
    target = cold_target_split(df, frac, seed)
    d = {k: set(getattr(drug, k).tolist()) for k in ("train", "valid", "test")}
    t = {k: set(getattr(target, k).tolist()) for k in ("train", "valid", "test")}
    return Split(
        np.asarray(sorted(d["train"] & t["train"]), dtype=np.int64),
        np.asarray(sorted(d["valid"] & t["valid"]), dtype=np.int64),
        np.asarray(sorted(d["test"] & t["test"]), dtype=np.int64),
    )


def temporal_split(
    df: pd.DataFrame, *, valid_year: int, test_year: int, year_col: str = "doc_year"
) -> Split:
    """Prospective proxy: train on the past, test on the future."""
    y = df[year_col].to_numpy()
    return Split(
        np.flatnonzero(y < valid_year),
        np.flatnonzero((y >= valid_year) & (y < test_year)),
        np.flatnonzero(y >= test_year),
    )


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
@dataclass
class LeakageReport:
    n_train: int
    n_valid: int
    n_test: int
    shared_inchikeys: int
    shared_uniprots: int
    shared_scaffolds: int
    max_tanimoto_train_test: float
    frac_test_above_0_7: float

    def assert_clean(self, *, allow_shared_targets: bool = True) -> None:
        assert self.shared_inchikeys == 0, f"{self.shared_inchikeys} ligands shared train/test"
        assert self.shared_scaffolds == 0, f"{self.shared_scaffolds} scaffolds shared train/test"
        if not allow_shared_targets:
            assert self.shared_uniprots == 0, f"{self.shared_uniprots} targets shared train/test"


def audit_leakage(
    df: pd.DataFrame, split: Split, *, sample: int = 2000, seed: int = 0
) -> LeakageReport:
    """Quantify the residual leakage between train and test.

    The maximum train-test similarity is *sampled* (an exact O(n*m) computation
    would be prohibitive at 1e5 x 1e4): it is an honest lower bound, not a
    guarantee, and the report says so. Written to ``docs/leakage_analysis.md``.
    """
    rng = np.random.default_rng(seed)
    tr, te = df.iloc[split.train], df.iloc[split.test]

    shared_ik = len(set(tr["inchikey"]) & set(te["inchikey"]))
    shared_up = len(set(tr["uniprot"]) & set(te["uniprot"]))
    sc_tr = {murcko_scaffold(s) for s in tr["smiles"].unique()}
    sc_te = {murcko_scaffold(s) for s in te["smiles"].unique()}
    shared_sc = len(sc_tr & sc_te)

    tr_smi = tr["smiles"].unique()
    te_smi = te["smiles"].unique()
    tr_s = rng.choice(tr_smi, size=min(sample, len(tr_smi)), replace=False)
    te_s = rng.choice(te_smi, size=min(sample, len(te_smi)), replace=False)
    fp_tr, fp_te = morgan_fingerprints(list(tr_s)), morgan_fingerprints(list(te_s))

    # Degenerate partition (empty train or test): nearest-neighbour similarity is
    # undefined. Report NaN rather than raising, and above all rather than returning
    # 0.0 — which would wrongly read as 'no leakage detected'.
    if not fp_tr or not fp_te:
        logger.warning(
            "leakage audit on a degenerate partition (train=%d, test=%d unique "
            "compounds): nearest-neighbour similarity is undefined",
            len(fp_tr),
            len(fp_te),
        )
        nn_sims = np.array([], dtype=np.float64)
    else:
        nn_sims = np.array([max(DataStructs.BulkTanimotoSimilarity(f, fp_tr)) for f in fp_te])

    return LeakageReport(
        n_train=len(split.train),
        n_valid=len(split.valid),
        n_test=len(split.test),
        shared_inchikeys=shared_ik,
        shared_uniprots=shared_up,
        shared_scaffolds=shared_sc,
        max_tanimoto_train_test=float(nn_sims.max()) if nn_sims.size else float("nan"),
        frac_test_above_0_7=float((nn_sims > 0.7).mean()) if nn_sims.size else float("nan"),
    )


SPLIT_REGISTRY = {
    "random": random_split,
    "scaffold": scaffold_split,
    "butina": butina_split,
    "cold_target": cold_target_split,
    "cold_pair": cold_pair_split,
    "temporal": temporal_split,
}
