"""Test de sortie de l'étape 0.1 — packaging et importabilité.

Vérifie deux propriétés distinctes :

1. Chaque sous-module de `kinoscope` s'importe sans `ImportError`. Une dépendance
   manquante ou mal épinglée dans `pyproject.toml` se manifeste ici, et non trois
   phases plus loin au milieu d'un entraînement.
2. Le paquet est importable depuis un répertoire de travail arbitraire, ce qui
   distingue une véritable installation (`pip install -e .`, layout `src/`) d'un
   simple ajout du dépôt au `sys.path` courant.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import kinoscope

# Sous-paquets attendus d'après §4.2 du MASTER_PLAN. La liste sert de garde-fou
# contre une arborescence amputée : `walk_packages` ne signale pas ce qui manque.
EXPECTED_SUBPACKAGES = (
    "kinoscope.data",
    "kinoscope.data.sources",
    "kinoscope.eval",
    "kinoscope.models",
    "kinoscope.models.layers",
    "kinoscope.serving",
    "kinoscope.utils",
)


def _walk_submodules() -> tuple[list[str], dict[str, str]]:
    """Énumère les sous-modules de `kinoscope`.

    Renvoie `(noms, erreurs_de_parcours)`. `pkgutil.walk_packages` importe les
    paquets pour accéder à leur `__path__` ; sans `onerror`, un échec y est
    silencieusement avalé, ce qui masquerait précisément ce que ce test cherche.
    """
    walk_errors: dict[str, str] = {}

    def _on_error(name: str) -> None:
        exc = sys.exc_info()[1]
        walk_errors[name] = f"{type(exc).__name__}: {exc}"

    names = [
        module.name
        for module in pkgutil.walk_packages(
            kinoscope.__path__, prefix=f"{kinoscope.__name__}.", onerror=_on_error
        )
    ]
    return sorted(names), walk_errors


def test_expected_subpackages_are_present() -> None:
    names, walk_errors = _walk_submodules()
    assert not walk_errors, f"parcours du paquet en échec : {walk_errors}"
    missing = [pkg for pkg in EXPECTED_SUBPACKAGES if pkg not in names]
    assert not missing, f"sous-paquets absents de l'arborescence : {missing}"


def test_every_submodule_imports_without_error() -> None:
    names, walk_errors = _walk_submodules()
    assert names, "aucun sous-module découvert : le paquet est-il bien installé ?"

    failures: dict[str, str] = dict(walk_errors)
    for name in names:
        try:
            importlib.import_module(name)
        # On capture largement : le but est le rapport exhaustif des échecs.
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    report = "\n".join(f"  {name}: {err}" for name, err in sorted(failures.items()))
    assert not failures, f"{len(failures)} sous-module(s) non importable(s) :\n{report}"


def test_importable_from_an_arbitrary_directory(tmp_path: Path) -> None:
    """`import kinoscope` doit fonctionner hors du répertoire du dépôt."""
    # Le processus fils est lancé hors du dépôt : il n'y trouverait pas
    # `pyproject.toml` et instrumenterait donc la couverture avec les réglages
    # par défaut (sans `branch`), ce qui fait échouer la fusion des données de
    # coverage du processus parent. On neutralise l'instrumentation héritée :
    # ce sous-processus sert à vérifier l'installation, pas à mesurer le code.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_"))
    }
    completed = subprocess.run(
        [sys.executable, "-c", "import kinoscope; print(kinoscope.__file__)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("__init__.py")
