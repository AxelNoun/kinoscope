"""Exit test for step 0.1 — packaging and importability.

Two distinct properties are checked:

1. Every submodule of `kinoscope` imports without an `ImportError`. A missing or
   badly pinned dependency shows up here, rather than three phases later in the
   middle of a training run.
2. The package is importable from an arbitrary working directory, which
   distinguishes a real installation (`pip install -e .`, `src/` layout) from
   merely having the repository on the current `sys.path`.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import kinoscope

# Subpackages expected from §4.2 of the MASTER_PLAN. The list guards against an
# amputated tree: `walk_packages` does not report what is missing.
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
    """Enumerate the submodules of `kinoscope`.

    Returns `(names, walk_errors)`. `pkgutil.walk_packages` imports packages to
    reach their `__path__`; without `onerror`, a failure there is swallowed
    silently, which would hide precisely what this test is looking for.
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
    assert not walk_errors, f"package walk failed: {walk_errors}"
    missing = [pkg for pkg in EXPECTED_SUBPACKAGES if pkg not in names]
    assert not missing, f"subpackages missing from the tree: {missing}"


def test_every_submodule_imports_without_error() -> None:
    names, walk_errors = _walk_submodules()
    assert names, "no submodule discovered: is the package actually installed?"

    failures: dict[str, str] = dict(walk_errors)
    for name in names:
        try:
            importlib.import_module(name)
        # Broad on purpose: the point is an exhaustive report of every failure.
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    report = "\n".join(f"  {name}: {err}" for name, err in sorted(failures.items()))
    assert not failures, f"{len(failures)} submodule(s) failed to import:\n{report}"


def test_importable_from_an_arbitrary_directory(tmp_path: Path) -> None:
    """`import kinoscope` must work outside the repository directory."""
    # The child process runs outside the repository, where it would not find
    # `pyproject.toml` and would therefore instrument coverage with the default
    # settings (no `branch`), which breaks the parent's coverage merge. Strip the
    # inherited instrumentation: this subprocess exists to check the install, not
    # to measure code.
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
