"""Test de sortie de l'étape 0.3 — déterminisme global.

Un résultat non reproductible est un bug bloquant. Ce module vérifie que
`set_global_seed` remet effectivement chaque générateur dans le même état, et
que le mode déterministe de torch est bien activé.

L'état global de torch (`use_deterministic_algorithms`, drapeaux cuDNN) est
sauvegardé puis restauré autour de chaque test : ces réglages sont propres au
processus, et les laisser actifs modifierait le comportement des autres tests
de la session.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from kinoscope.utils.seed import set_global_seed


@pytest.fixture(autouse=True)
def _restore_torch_global_state() -> Iterator[None]:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_hashseed = os.environ.get("PYTHONHASHSEED")
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        if previous_hashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous_hashseed


def test_reproducible_forward() -> None:
    """Deux exécutions à graine identique produisent des tenseurs bit-à-bit égaux.

    Le mini-forward couvre les deux sources d'aléa d'un entraînement : l'initialisation
    des poids et le tirage des données. `torch.equal` compare bit à bit, pas à une
    tolérance près — c'est le seul critère acceptable pour du déterminisme.
    """

    def run() -> torch.Tensor:
        set_global_seed(1234)
        layer = torch.nn.Linear(16, 8)
        x = torch.randn(4, 16)
        return layer(x)

    first, second = run(), run()
    assert torch.equal(first, second)


def test_distinct_seeds_produce_distinct_draws() -> None:
    """Garde-fou contre un `set_global_seed` qui ne sèmerait rien du tout."""
    set_global_seed(0)
    first = torch.randn(8)
    set_global_seed(1)
    second = torch.randn(8)
    assert not torch.equal(first, second)


def test_seeds_python_and_numpy_streams() -> None:
    # `np.random.rand` (NPY002) est délibéré : c'est le générateur global hérité
    # que `set_global_seed` sème, et le seul moyen de vérifier qu'il l'est bien.
    set_global_seed(7)
    py_first, np_first = random.random(), np.random.rand()  # noqa: NPY002
    set_global_seed(7)
    assert random.random() == py_first
    assert np.random.rand() == np_first  # noqa: NPY002


def test_exports_pythonhashseed_for_child_processes() -> None:
    """`PYTHONHASHSEED` n'agit qu'au démarrage d'un interpréteur.

    Le poser ici ne change donc rien au processus courant : cela ne sert qu'aux
    processus fils (workers de `DataLoader` lancés par `spawn`). On vérifie
    l'export, pas un effet sur le hachage local — qui n'existe pas.
    """
    set_global_seed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_deterministic_flag_is_honoured() -> None:
    set_global_seed(3, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    set_global_seed(3, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
