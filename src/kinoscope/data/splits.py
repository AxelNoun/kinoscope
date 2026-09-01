"""Stratégies de partitionnement et audit de fuite de données.

C'est le module central du projet : la thèse scientifique de KinoScope est que
les performances DTI publiées s'effondrent dès que la partition interdit au
modèle de mémoriser un ligand ou une cible déjà vus.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
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
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {k: getattr(self, k).tolist() for k in ("train", "valid", "test")}


# --------------------------------------------------------------------------- #
# Fingerprints & scaffolds
# --------------------------------------------------------------------------- #
def morgan_fingerprints(smiles: list[str]):  # noqa: ANN201
    return [_MFPGEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def murcko_scaffold(smiles: str, *, include_chirality: bool = False) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def butina_clusters(smiles: list[str], cutoff: float = 0.6) -> list[int]:
    """Clustering Butina sur distance Tanimoto ECFP4.

    ``cutoff`` est une *distance* : 0.6 correspond à une similarité Tanimoto de
    0.4, seuil au-delà duquel deux composés sont généralement considérés comme
    appartenant à la même série chimique.
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
# Stratégies
# --------------------------------------------------------------------------- #
def _group_split(groups: list[int] | list[str], frac: tuple[float, float, float], seed: int) -> Split:
    """Partitionne des *groupes* entiers, du plus gros au plus petit (greedy bin-packing).

    Le remplissage glouton évite qu'un cluster massif déséquilibre le test set,
    ce qui arrive avec un simple shuffle de groupes.
    """
    rng = np.random.default_rng(seed)
    members: dict[object, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        members[g].append(idx)

    order = sorted(members, key=lambda g: len(members[g]), reverse=True)
    n = len(groups)
    targets = [frac[0] * n, frac[1] * n, frac[2] * n]
    buckets: list[list[int]] = [[], [], []]

    for g in order:
        deficits = [targets[k] - len(buckets[k]) for k in range(3)]
        k = int(np.argmax(deficits))
        buckets[k].extend(members[g])

    for b in buckets:
        rng.shuffle(b)
    return Split(*[np.asarray(sorted(b), dtype=np.int64) for b in buckets])


def random_split(df: pd.DataFrame, frac=(0.8, 0.1, 0.1), seed: int = 42) -> Split:
    """Témoin optimiste. Sert uniquement à quantifier l'ampleur de la fuite."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_tr, n_va = int(frac[0] * len(df)), int(frac[1] * len(df))
    return Split(np.sort(idx[:n_tr]), np.sort(idx[n_tr : n_tr + n_va]), np.sort(idx[n_tr + n_va :]))


def scaffold_split(df: pd.DataFrame, frac=(0.8, 0.1, 0.1), seed: int = 42) -> Split:
    """Cold-drug par squelette de Bemis-Murcko."""
    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    return _group_split(scaffolds, frac, seed)


def butina_split(df: pd.DataFrame, cutoff: float = 0.6, frac=(0.8, 0.1, 0.1), seed: int = 42) -> Split:
    """Cold-drug plus strict : clusters de similarité, pas seulement de squelette."""
    return _group_split(butina_clusters(list(df["smiles"]), cutoff), frac, seed)


def cold_target_split(df: pd.DataFrame, frac=(0.8, 0.1, 0.1), seed: int = 42, *, by: str = "manning_group") -> Split:
    """Cold-target. ``by='manning_group'`` = leave-group-out (le régime le plus dur) ;
    ``by='uniprot'`` = cibles inédites mais familles potentiellement vues."""
    if by not in df.columns:
        raise KeyError(f"colonne de regroupement absente: {by}")
    return _group_split(list(df[by]), frac, seed)


def cold_pair_split(df: pd.DataFrame, frac=(0.8, 0.1, 0.1), seed: int = 42) -> Split:
    """Intersection cold-drug x cold-target : le seul régime réellement prospectif."""
    drug = scaffold_split(df, frac, seed)
    target = cold_target_split(df, frac, seed)
    d = {k: set(getattr(drug, k).tolist()) for k in ("train", "valid", "test")}
    t = {k: set(getattr(target, k).tolist()) for k in ("train", "valid", "test")}
    return Split(
        np.asarray(sorted(d["train"] & t["train"]), dtype=np.int64),
        np.asarray(sorted(d["valid"] & t["valid"]), dtype=np.int64),
        np.asarray(sorted(d["test"] & t["test"]), dtype=np.int64),
    )


def temporal_split(df: pd.DataFrame, *, valid_year: int, test_year: int, year_col: str = "doc_year") -> Split:
    """Proxy prospectif : on entraîne sur le passé, on teste sur le futur."""
    y = df[year_col].to_numpy()
    return Split(
        np.flatnonzero(y < valid_year),
        np.flatnonzero((y >= valid_year) & (y < test_year)),
        np.flatnonzero(y >= test_year),
    )


# --------------------------------------------------------------------------- #
# Audit de fuite
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
        assert self.shared_inchikeys == 0, f"{self.shared_inchikeys} ligands partagés train/test"
        assert self.shared_scaffolds == 0, f"{self.shared_scaffolds} scaffolds partagés train/test"
        if not allow_shared_targets:
            assert self.shared_uniprots == 0, f"{self.shared_uniprots} cibles partagées train/test"


def audit_leakage(df: pd.DataFrame, split: Split, *, sample: int = 2000, seed: int = 0) -> LeakageReport:
    """Quantifie la fuite résiduelle entre train et test.

    La similarité maximale train-test est échantillonnée (O(n*m) exact serait
    prohibitif sur 10^5 x 10^4) : c'est une borne inférieure honnête, pas une
    garantie. Le rapport est écrit dans ``docs/leakage_analysis.md``.
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

    nn_sims = np.array([max(DataStructs.BulkTanimotoSimilarity(f, fp_tr)) for f in fp_te])
    return LeakageReport(
        n_train=len(split.train),
        n_valid=len(split.valid),
        n_test=len(split.test),
        shared_inchikeys=shared_ik,
        shared_uniprots=shared_up,
        shared_scaffolds=shared_sc,
        max_tanimoto_train_test=float(nn_sims.max()),
        frac_test_above_0_7=float((nn_sims > 0.7).mean()),
    )


SPLIT_REGISTRY = {
    "random": random_split,
    "scaffold": scaffold_split,
    "butina": butina_split,
    "cold_target": cold_target_split,
    "cold_pair": cold_pair_split,
    "temporal": temporal_split,
}
