# Model Card — KinoScope v0.1.0

> **Status: no model has been trained yet.** This card describes the intended model and the
> commitments made about how it will be reported. Every quantitative field is a placeholder
> until a training run exists.

## Details

- **Architecture**: GINE (5 layers, 256d, Jumping Knowledge) × frozen ESM-2 650M over the
  85-residue KLIFS pocket, fused by a Bilinear Attention Network (4 heads, rank 96).
- **Output**: `pAffinity` (μ) with aleatoric variance (log σ²), an activity logit, and an
  atom–residue interaction map.
- **Trainable parameters**: not yet measured (ESM-2 excluded, frozen).
- **Licence**: Apache-2.0.

## Intended use

Library triage and compound prioritisation during hit-finding on human kinases.
**Not intended** for clinical or regulatory use, nor as a substitute for a biochemical assay.

## Training data

See `docs/data_card.md`. Sources: ChEMBL (`assay_type='B'`, confidence ≥ 8, relation `'='`),
BindingDB, Davis, KIBA. Aggregated by median pChEMBL, rejecting pairs whose inter-measurement
standard deviation exceeds 1.0 log unit.

## Evaluation

Four split regimes, with every metric accompanied by a bootstrap 95 % confidence interval.
The `random` regime is reported solely as a leakage control, **never** as model performance.

## Known limitations and biases

| Limitation | Consequence |
|---|---|
| Domain restricted to the ATP site (types I/II) | Predictions unreliable for type III/IV allosteric binders |
| ChEMBL publication bias towards actives | True negatives under-represented; Davis provides a complete matrix as counterweight |
| Human kinome only | No validated cross-species extrapolation |
| DFG conformation not modelled explicitly | Type II inhibitors are treated as type I |
| Exchangeability violated under cold-target splits | Conformal coverage is empirical, not guaranteed |
| IC50/Ki/Kd heterogeneity | Handled as an assay covariate; cross-endpoint comparisons should be avoided |

## Ethical considerations

The model is a prioritisation tool. A prediction outside the applicability domain
(nearest-neighbour Tanimoto < 0.35) is flagged and must be discarded, not rounded up.
