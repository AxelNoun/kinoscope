# KinoScope

**Prédiction d'affinité kinase–inhibiteur par modèle hybride GNN + ESM-2, sous protocole d'évaluation sans fuite de données.**

[![CI](https://github.com/AxelNoun/kinoscope/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelNoun/kinoscope/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

---

## Le problème

Les performances DTI publiées sur Davis et KIBA sont obtenues sous partition aléatoire.
Un même ligand se retrouve alors des deux côtés de la frontière train/test, et le modèle
apprend un **prior de composé** plutôt qu'une fonction d'interaction. Le chiffre publié
ne dit rien de ce qui arrive sur un chémotype ou une kinase inédits — c'est-à-dire sur le
seul cas d'usage réel.

KinoScope fait deux choses :

1. Un modèle hybride **GINE (ligand) × ESM-2 (poche KLIFS 85 résidus)** avec fusion par
   attention bilinéaire, sortie hétéroscédastique et intervalles conformes calibrés.
2. Un **protocole d'évaluation à quatre régimes** (aléatoire, cold-drug, cold-target,
   temporel) avec audit de fuite automatisé en CI, qui mesure honnêtement ce qui généralise.

## Résultats

> Régénéré automatiquement par `.github/workflows/benchmark.yml`. Toutes les valeurs sont
> accompagnées de leur IC 95 % bootstrap (1000 rééchantillonnages).

| Régime de split | RMSE ↓ | Pearson r ↑ | Spearman ρ ↑ | CI ↑ | PICP@90 % |
|---|---|---|---|---|---|
| `random` (témoin optimiste) | TBD | TBD | TBD | TBD | TBD |
| `scaffold` (cold-drug) | TBD | TBD | TBD | TBD | TBD |
| `cold_target` (leave-group-out) | TBD | TBD | TBD | TBD | TBD |
| `temporal` (prospectif) | TBD | TBD | TBD | TBD | TBD |

Ablations : `no_plm` (one-hot séquence), `no_ban` (concaténation simple),
`full_sequence` (ESM-2 sur la séquence entière au lieu de la poche KLIFS).

## Démarrage rapide

```bash
git clone https://github.com/AxelNoun/kinoscope && cd kinoscope
make setup            # uv venv + deps + hooks pre-commit
make data             # pipeline DVC : ingestion -> standardisation -> splits
make embeddings       # cache ESM-2 des poches (une seule fois, ~110 Mo)
make train EXPERIMENT=main
make eval             # les 4 régimes + bootstrap
make serve            # API sur http://localhost:8000/docs
```

## Architecture

```
SMILES ──RDKit──> graphe 2D ──GINE×5 (résiduel + JK)──> [B, Na, 256] ─┐
                                                                      ├─> BAN ─> MLP ─> (μ, log σ²)
UniProt ──KLIFS──> 85 résidus ──ESM-2 (gelé, cache)──> [B, 85, 256] ─┘         └─> carte de contacts
```

Détails, compromis et justifications : [`docs/architecture.md`](docs/architecture.md).

## Limites déclarées

- Restreint aux **inhibiteurs de site ATP** (types I/II). Les allostériques de type III/IV
  sortent du domaine d'applicabilité par construction.
- Kinases humaines uniquement, poche KLIFS obligatoirement résolue.
- L'affinité prédite est un `pAffinity` conditionné sur le type d'essai. Aucune conversion
  Cheng-Prusoff n'est appliquée.
- Sur split cold-target, l'échangeabilité requise par la prédiction conforme est violée :
  la couverture rapportée est empirique, pas garantie.

Voir [`MODEL_CARD.md`](MODEL_CARD.md).

## Citer

```bibtex
@software{kinoscope,
  title  = {KinoScope: leakage-free evaluation of hybrid GNN--PLM models for kinase inhibitor affinity},
  author = {Coiffard-Donaty, Axel},
  year   = {2026},
  url    = {https://github.com/AxelNoun/kinoscope}
}
```

## Sources de données

ChEMBL (CC BY-SA 3.0) · BindingDB (CC BY 3.0) · KLIFS · UniProt (CC BY 4.0) ·
PDBbind · Davis & KIBA via Therapeutics Data Commons. Détail dans [`docs/data_card.md`](docs/data_card.md).
