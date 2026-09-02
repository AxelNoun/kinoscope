"""Chemical standardisation pipeline (RDKit).

Every molecule entering the project — training and inference alike — goes through
exactly this function. A divergence between training-time and inference-time
standardisation is the leading cause of silent production degradation for a
QSAR/DTI model.
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
    inchikey_skeleton: str  # stereo layer stripped: the deduplication key for clustering
    mol_weight: float
    n_heavy_atoms: int


class RejectionReason(str):
    """Rejection reasons, logged for the data-card audit."""

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
    """Return the standardised molecule, or ``None`` if it is rejected.

    Steps, in order (the order matters):
      1. parsing + sanitisation
      2. largest-fragment selection (strips counter-ions and solvates)
      3. functional-group normalisation (nitro, N-oxides, ...)
      4. charge neutralisation
      5. canonical tautomer (costly; disableable for real-time inference)
      6. domain filters (elements, size, mass)

    Fragment selection must precede neutralisation: the other way round, a
    counter-ion would be neutralised and then kept.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.debug("reject %s: %s", smiles, RejectionReason.UNPARSABLE)
        return None

    try:
        mol = _fragment_chooser.choose(mol)
        mol = _normalizer.normalize(mol)
        mol = _uncharger.uncharge(mol)
        if canonical_tautomer:
            mol = _tautomer_enumerator.Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError, RuntimeError):
        logger.debug("reject %s: %s", smiles, RejectionReason.SANITIZATION_FAILED)
        return None

    if "." in Chem.MolToSmiles(mol):
        logger.debug("reject %s: %s", smiles, RejectionReason.DISCONNECTED)
        return None
    if any(a.GetAtomicNum() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
        logger.debug("reject %s: %s", smiles, RejectionReason.FORBIDDEN_ELEMENT)
        return None

    n_heavy = mol.GetNumHeavyAtoms()
    mw = Descriptors.MolWt(mol)
    if not (MIN_HEAVY_ATOMS <= n_heavy <= MAX_HEAVY_ATOMS) or mw > MAX_MOLWT:
        logger.debug("reject %s: %s", smiles, RejectionReason.SIZE_OUT_OF_RANGE)
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
    """Batch version. ``n_jobs > 1`` requires joblib (RDKit is not thread-safe:
    use *processes*, never threads)."""
    if n_jobs == 1:
        return [standardize(s) for s in smiles_list]
    from joblib import Parallel, delayed

    return list(
        Parallel(n_jobs=n_jobs, backend="loky")(delayed(standardize)(s) for s in smiles_list)
    )
