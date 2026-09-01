"""Semis global des générateurs pseudo-aléatoires.

Toute exécution stochastique du projet passe par `set_global_seed`. Un résultat
non reproductible est traité comme un bug bloquant, pas comme une fatalité.

Deux limites sont volontairement laissées visibles plutôt que masquées :

- `PYTHONHASHSEED` est lu par CPython **au démarrage** de l'interpréteur. Le
  poser depuis un processus déjà lancé ne change rien à ce processus ; cela ne
  sert qu'aux processus fils (workers `DataLoader` créés par `spawn`, appels à
  `subprocess`). Le poser reste utile, mais il ne rend pas déterministe l'ordre
  d'itération d'un `set` du processus courant.
- `np.random.seed` ne sème que le générateur global hérité. Les fonctions du
  projet qui instancient leur propre `np.random.default_rng(seed)` — dans
  `data/splits.py` et `eval/metrics.py` — sont indépendantes de cet état, par
  construction : leur reproductibilité tient à leur argument `seed`, ce qui est
  la propriété recherchée.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Exigé par cuBLAS lorsque les algorithmes déterministes sont actifs sur GPU.
# Sans cette variable, `torch.use_deterministic_algorithms(True)` lève une
# RuntimeError au premier appel matriciel sur CUDA >= 10.2.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Sème `random`, `numpy`, `torch` et `torch.cuda`, et fixe le mode déterministe.

    Parameters
    ----------
    seed
        Graine appliquée à tous les générateurs.
    deterministic
        Si vrai, exige des noyaux déterministes. Le mode est **strict** : une
        opération dépourvue d'implémentation déterministe lève une RuntimeError
        au lieu de produire silencieusement un résultat irreproductible. C'est
        l'erreur qui est le signal utile ; `torch.use_deterministic_algorithms`
        accepte `warn_only=True`, mais l'utiliser ici reviendrait à rendre
        indétectable exactement ce qu'on cherche à détecter.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    # NPY002 recommande `np.random.Generator`, à raison pour du code neuf. Mais
    # l'objet de cette ligne est précisément de semer le générateur global hérité :
    # il n'existe aucun moyen de le faire via l'API `Generator`, et c'est cet état
    # que consomme tout appel `np.random.*` du projet ou de ses dépendances.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # sans effet si aucun GPU n'est visible

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    # `benchmark` sélectionne l'algorithme de convolution par mesure de temps :
    # le choix peut varier d'une exécution à l'autre, donc le résultat aussi.
    torch.backends.cudnn.benchmark = not deterministic
