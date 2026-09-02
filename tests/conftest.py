"""Shared test configuration.

Two Hypothesis profiles, because the two use cases have different goals: ``dev``
must return quickly while code is being written, ``ci`` must actually search for
counterexamples. The profile is selected by the ``HYPOTHESIS_PROFILE``
environment variable (default ``dev``), set to ``ci`` in the GitHub Actions
workflow.

``deadline=None`` in both cases: RDKit featurisation and Butina clustering have
highly variable runtimes from one example to the next, and a deadline failure
would be a false positive — not the information we are after.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, Verbosity, settings

settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    # This repository's strategies build RDKit molecules and DataFrames:
    # generation is slow by nature, not by design flaw. `too_slow` would only
    # contribute spurious failures here.
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.normal,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
