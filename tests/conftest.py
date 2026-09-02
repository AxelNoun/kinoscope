"""Configuration commune des tests.

Deux profils Hypothesis, parce que les deux usages n'ont pas le même objectif :
``dev`` doit rendre la main vite pendant l'écriture du code, ``ci`` doit
chercher réellement des contre-exemples. Le profil est choisi par la variable
d'environnement ``HYPOTHESIS_PROFILE`` (défaut : ``dev``), positionnée à ``ci``
dans le workflow GitHub Actions.

``deadline=None`` dans les deux cas : la featurisation RDKit et le clustering
Butina ont des temps d'exécution très variables d'un exemple à l'autre, et un
échec pour dépassement de délai serait un faux positif — pas l'information
qu'on cherche.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, Verbosity, settings

settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    # Les stratégies de ce dépôt construisent des molécules RDKit et des
    # DataFrames : la génération est lente par nature, pas par défaut de
    # conception. `too_slow` n'apporterait ici qu'un échec parasite.
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.normal,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
