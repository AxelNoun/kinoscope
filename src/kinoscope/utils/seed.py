"""Global seeding of pseudo-random generators.

Every stochastic execution in this project goes through `set_global_seed`. A
non-reproducible result is treated as a blocking bug, not as a fact of life.

Two limitations are deliberately left visible rather than papered over:

- `PYTHONHASHSEED` is read by CPython **at interpreter startup**. Setting it from
  an already-running process changes nothing for that process; it only matters
  for child processes (`DataLoader` workers created by `spawn`, `subprocess`
  calls). Setting it is still worthwhile, but it does not make the current
  process's `set` iteration order deterministic.
- `np.random.seed` only seeds the legacy global generator. Functions in this
  project that instantiate their own `np.random.default_rng(seed)` — in
  `data/splits.py` and `eval/metrics.py` — are independent of that state by
  design: their reproducibility rests on their `seed` argument, which is the
  property we want.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Required by cuBLAS when deterministic algorithms are enabled on GPU. Without
# this variable, `torch.use_deterministic_algorithms(True)` raises a RuntimeError
# on the first matrix operation under CUDA >= 10.2.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed `random`, `numpy`, `torch` and `torch.cuda`, and set deterministic mode.

    Parameters
    ----------
    seed
        Seed applied to every generator.
    deterministic
        If true, require deterministic kernels. The mode is **strict**: an
        operation with no deterministic implementation raises a RuntimeError
        rather than silently producing an irreproducible result. The error is the
        useful signal here; `torch.use_deterministic_algorithms` accepts
        `warn_only=True`, but using it would make undetectable exactly what we
        are trying to detect.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    # NPY002 recommends `np.random.Generator`, rightly so for new code. But the
    # whole point of this line is to seed the legacy global generator: there is no
    # way to do that through the `Generator` API, and that state is what every
    # `np.random.*` call in this project or its dependencies consumes.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no effect when no GPU is visible

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    # `benchmark` picks the convolution algorithm by timing it: the choice can
    # vary from one run to the next, and so can the result.
    torch.backends.cudnn.benchmark = not deterministic
