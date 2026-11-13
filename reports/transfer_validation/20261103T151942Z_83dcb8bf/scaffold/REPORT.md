# scaffold transfer validation

Numbers are computed from the project's saved test predictions; no scores are supplied by this helper.

## Seed means and sample standard deviations

| Dataset | Variant | MSE | MAE | Concordance index |
|---|---|---|---|---|
| kiba | scratch | 0.64422 ± 0.02709 | 0.55196 ± 0.01707 | 0.69412 ± 0.00598 |
| kiba | frozen | 0.69612 ± 0.01313 | 0.57750 ± 0.00404 | 0.68645 ± 0.00438 |
| kiba | finetune | 0.65586 ± 0.00880 | 0.56312 ± 0.00557 | 0.69642 ± 0.00136 |
| davis | scratch | 0.82622 ± 0.03509 | 0.49544 ± 0.01309 | 0.76477 ± 0.02057 |
| davis | frozen | 0.68818 ± 0.01927 | 0.46918 ± 0.02185 | 0.71384 ± 0.00946 |
| davis | finetune | 0.61880 ± 0.02401 | 0.45098 ± 0.02859 | 0.75380 ± 0.01058 |

## Per-seed fine-tuned minus scratch MSE

| Dataset | Training seed | Difference (negative favors fine-tuning) |
|---|---|---|
| kiba | 42 | 0.03007 |
| kiba | 43 | 0.02137 |
| kiba | 44 | -0.01654 |
| davis | 42 | -0.19867 |
| davis | 43 | -0.26954 |
| davis | 44 | -0.15405 |

## Test-set exposure and independent group counts

| Dataset | Pairs | Drugs | Scaffolds | Drugs in PCBA train | Drugs in PCBA validation |
|---|---|---|---|---|---|
| kiba | 11762 | 361 | 296 | 14 | 5 |
| davis | 2653 | 7 | 7 | 1 | 2 |

Across the two curated datasets: 9 shared drugs, 179 shared exact proteins and 1043 shared drug-protein pairs. These benchmarks need not be wholly independent.

## Primary paired comparison: fine-tuned minus scratch MSE

Negative differences favor fine-tuning. These are conditional, pointwise 95% percentile intervals; not p-values or proof of causation.

- kiba: difference 0.01163; interval [-0.02068, 0.04363]; 296 held-out scaffold groups.
  The interval includes zero; these data do not clearly separate the models at this interval level.
- davis: difference -0.20742; interval [-0.60204, 0.03189]; 7 held-out scaffold groups.
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
