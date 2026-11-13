# cold_drug transfer validation

Numbers are computed from the project's saved test predictions; no scores are supplied by this helper.

## Seed means and sample standard deviations

| Dataset | Variant | MSE | MAE | Concordance index |
|---|---|---|---|---|
| kiba | scratch | 0.65265 ± 0.01497 | 0.57358 ± 0.00558 | 0.73711 ± 0.00112 |
| kiba | frozen | 0.67051 ± 0.00736 | 0.57810 ± 0.00451 | 0.73495 ± 0.00191 |
| kiba | finetune | 0.63136 ± 0.01706 | 0.56065 ± 0.01063 | 0.74404 ± 0.00426 |
| davis | scratch | 0.90039 ± 0.04690 | 0.56848 ± 0.08921 | 0.65523 ± 0.08203 |
| davis | frozen | 0.75006 ± 0.05486 | 0.52848 ± 0.00999 | 0.69731 ± 0.02580 |
| davis | finetune | 0.63529 ± 0.02416 | 0.47571 ± 0.01646 | 0.73936 ± 0.00536 |

## Per-seed fine-tuned minus scratch MSE

| Dataset | Training seed | Difference (negative favors fine-tuning) |
|---|---|---|
| kiba | 42 | -0.03648 |
| kiba | 43 | 0.01222 |
| kiba | 44 | -0.03960 |
| davis | 42 | -0.26815 |
| davis | 43 | -0.22909 |
| davis | 44 | -0.29804 |

## Test-set exposure and independent group counts

| Dataset | Pairs | Drugs | Scaffolds | Drugs in PCBA train | Drugs in PCBA validation |
|---|---|---|---|---|---|
| kiba | 11769 | 460 | 348 | 8 | 1 |
| davis | 2653 | 7 | 7 | 1 | 2 |

Across the two curated datasets: 9 shared drugs, 179 shared exact proteins and 1043 shared drug-protein pairs. These benchmarks need not be wholly independent.

## Primary paired comparison: fine-tuned minus scratch MSE

Negative differences favor fine-tuning. These are conditional, pointwise 95% percentile intervals; not p-values or proof of causation.

- kiba: difference -0.02129; interval [-0.06803, 0.02074]; 460 held-out drug_id groups.
  The interval includes zero; these data do not clearly separate the models at this interval level.
- davis: difference -0.26509; interval [-0.75730, 0.01476]; 7 held-out drug_id groups.
  Few groups (<20): treat the percentile interval cautiously; inspect group errors and additional fixed splits.
  The interval includes zero; these data do not clearly separate the models at this interval level.

## Scope and safeguards

- Predictions are aligned by pair IDs, with exact entity and saved-label checks across models/seeds.
- Data and test-checkpoint fingerprints and saved errors are checked before analysis.
- Errors remain pair-weighted; a drug with more pairs contributes more, as in the original benchmark metric.
- Seeds remain fixed in the cluster bootstrap. Their sample SD is reported separately; this is not a combined seed-and-drug interval.
- Scaffold reviews resample scaffolds rather than treating related drugs as independent.
- Review data_and_exposure_review.json before claiming pretraining-unseen drugs or independent benchmarks.
- Do not change hyperparameters based on these test reports.
