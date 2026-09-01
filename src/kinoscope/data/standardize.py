"""Pipeline de standardisation chimique (RDKit).

Toute molécule entrant dans le projet (entraînement comme inférence) traverse
exactement cette fonction. Une divergence entre la standardisation d'entraînement
et celle d'inférence est la première cause de dégradation silencieuse en
production pour un modèle QSAR/DTI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

ALLOWED_ELEMENTS: frozenset[int] = frozenset({1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53})
MAX_MOLWT: float = 900.0
MIN_HEAVY_ATOMS: int = 6
MAX_HEAVY_ATOMS: int = 80


@dataclass(frozen=True)
class StandardizedMolecule:
    smiles: str
    inchikey: str
    inchikey_skeleton: str  # sans couche stéréo : clé de déduplication du clustering
    mol_weight: float
    n_heavy_atoms: int


class RejectionReason(str):
    """Motifs de rejet, journalisés pour l'audit du data card."""

    UNPARSABLE = "unparsable"
    DISCONNECTED = "disconnected_after_fragment_choice"
    FORBIDDEN_ELEMENT = "forbidden_element"
    SIZE_OUT_OF_RANGE = "size_out_of_range"
    SANITIZATION_FAILED = "sanitization_failed"


_normalizer = rdMolStandardize.Normalizer()
_uncharger = rdMolStandardize.Uncharger()
_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_tautomer_enumerator.SetMaxTautomers(256)


def standardize(smiles: str, *, canonical_tautomer: bool = True) -> StandardizedMolecule | None:
    """Retourne la molécule standardisée, ou ``None`` si elle est rejetée.

    Étapes, dans l'ordre (l'ordre importe) :
      1. parsing + sanitization
      2. sélection du plus gros fragment (retire contre-ions et solvats)
      3. normalisation des groupes fonctionnels (nitro, N-oxydes, ...)
      4. neutralisation des charges
      5. tautomère canonique (coûteux ; désactivable pour l'inférence temps réel)
      6. filtres de domaine (éléments, taille, masse)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.debug("rejet %s: %s", smiles, RejectionReason.UNPARSABLE)
        return None

    try:
        mol = _fragment_chooser.choose(mol)
        mol = _normalizer.normalize(mol)
        mol = _uncharger.uncharge(mol)
        if canonical_tautomer:
            mol = _tautomer_enumerator.Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError, RuntimeError):
        logger.debug("rejet %s: %s", smiles, RejectionReason.SANITIZATION_FAILED)
        return None

    if "." in Chem.MolToSmiles(mol):
        return None
    if any(a.GetAtomicNum() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
        return None

    n_heavy = mol.GetNumHeavyAtoms()
    mw = Descriptors.MolWt(mol)
    if not (MIN_HEAVY_ATOMS <= n_heavy <= MAX_HEAVY_ATOMS) or mw > MAX_MOLWT:
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    inchikey = Chem.MolToInchiKey(mol)

    flat = Chem.MolFromSmiles(canonical)
    Chem.RemoveStereochemistry(flat)
    skeleton = Chem.MolToInchiKey(flat).split("-")[0]

    return StandardizedMolecule(
        smiles=canonical,
        inchikey=inchikey,
        inchikey_skeleton=skeleton,
        mol_weight=mw,
        n_heavy_atoms=n_heavy,
    )


def standardize_batch(
    smiles_list: list[str], *, n_jobs: int = 1
) -> list[StandardizedMolecule | None]:
    """Version batch. ``n_jobs > 1`` requiert joblib (RDKit n'est pas thread-safe :
    utiliser des *processus*, jamais des threads)."""
    if n_jobs == 1:
        return [standardize(s) for s in smiles_list]
    from joblib import Parallel, delayed

    return list(
        Parallel(n_jobs=n_jobs, backend="loky")(delayed(standardize)(s) for s in smiles_list)
    )
