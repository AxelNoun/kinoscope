"""Exit test for step 0.3 — global determinism.

A non-reproducible result is a blocking bug. This module checks that
`set_global_seed` really returns every generator to the same state, and that
torch's deterministic mode is actually enabled.

Torch's global state (`use_deterministic_algorithms`, cuDNN flags) is saved and
restored around each test: those settings are process-wide, and leaving them
active would change the behaviour of the rest of the session.
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
    """Two runs at the same seed produce bit-for-bit identical tensors.

    The mini forward pass covers both sources of randomness in a training run:
    weight initialisation and data sampling. `torch.equal` compares bit for bit,
    not within a tolerance — the only acceptable criterion for determinism.
    """

    def run() -> torch.Tensor:
        set_global_seed(1234)
        layer = torch.nn.Linear(16, 8)
        x = torch.randn(4, 16)
        return layer(x)

    first, second = run(), run()
    assert torch.equal(first, second)


def test_distinct_seeds_produce_distinct_draws() -> None:
    """Guard against a `set_global_seed` that seeds nothing at all."""
    set_global_seed(0)
    first = torch.randn(8)
    set_global_seed(1)
    second = torch.randn(8)
    assert not torch.equal(first, second)


def test_seeds_python_and_numpy_streams() -> None:
    # `np.random.rand` (NPY002) is deliberate: the legacy global generator is what
    # `set_global_seed` seeds, and the only way to check that it does.
    set_global_seed(7)
    py_first, np_first = random.random(), np.random.rand()  # noqa: NPY002
    set_global_seed(7)
    assert random.random() == py_first
    assert np.random.rand() == np_first  # noqa: NPY002


def test_exports_pythonhashseed_for_child_processes() -> None:
    """`PYTHONHASHSEED` only takes effect at interpreter startup.

    Setting it here therefore changes nothing for the current process: it only
    serves child processes (`DataLoader` workers started by `spawn`). What is
    checked is the export, not a local hashing effect — there is none.
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
