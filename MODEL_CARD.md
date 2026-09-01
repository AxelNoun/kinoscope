# Model Card — KinoScope v0.1.0

## Détails

- **Architecture** : GINE (5 couches, 256d, Jumping Knowledge) × ESM-2 650M gelé sur poche KLIFS 85 résidus, fusion par Bilinear Attention Network (4 têtes, rang 96).
- **Sortie** : `pAffinity` (μ) et variance aléatorique (log σ²), logit d'activité, carte d'interaction atome–résidu.
- **Paramètres entraînables** : ~XX M (ESM-2 exclu, gelé).
- **Licence** : Apache-2.0.

## Usage prévu

Triage de librairies et priorisation de composés en phase hit-finding sur kinases humaines.
**Non destiné** à un usage clinique, réglementaire, ou à la substitution d'un essai biochimique.

## Données d'entraînement

Voir `docs/data_card.md`. Sources : ChEMBL (assay_type='B', confidence ≥ 8, relation '='),
BindingDB, Davis, KIBA. Agrégation par médiane de pChEMBL, rejet si σ inter-mesures > 1.0 log.

## Évaluation

Quatre régimes de partition, métriques accompagnées d'IC 95 % bootstrap. Le régime
`random` est rapporté uniquement comme témoin de fuite, **jamais** comme performance du modèle.

## Limites et biais connus

| Limite | Conséquence |
|---|---|
| Domaine restreint au site ATP (types I/II) | Prédictions non fiables pour les allostériques III/IV |
| Biais de publication ChEMBL vers les actifs | Sous-représentation des vrais négatifs ; Davis fournit une matrice complète en contrepoids |
| Kinome humain uniquement | Aucune extrapolation inter-espèces validée |
| Conformation DFG non modélisée explicitement | Les inhibiteurs de type II sont traités comme des type I |
| Échangeabilité violée sous cold-target | Couverture conforme empirique, non garantie |
| Hétérogénéité IC50/Ki/Kd | Traitée par covariable d'essai ; comparaisons inter-endpoints à éviter |

## Considérations éthiques

Le modèle est un outil de priorisation. Une prédiction hors domaine d'applicabilité
(Tanimoto NN < 0.35) est signalée et doit être écartée, pas arrondie à la hausse.
