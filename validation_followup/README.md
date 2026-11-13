# Transfer validation follow-up

Additive helper for the existing `attentivefp_dta_final` project. It does not
replace source files, reinstall packages, retrain PCBA, rebuild ESM features,
delete runs, or tune settings from test results. Actual server results are NOT
bundled. Local tests use explicitly synthetic prediction fixtures.

## Install and run

Place `validation_followup.zip` in the project directory on your GPU server:

```bash
source /home/dmlab/miniconda3/etc/profile.d/conda.sh
conda activate attentivefp-pcba
cd /home/dmlab/Aviral/attentivefp_dta_final
unzip -n validation_followup.zip
bash validation_followup/start.sh "$PWD"
```

Run the launcher once and do not start other training on the GPU concurrently.
It follows a unique log immediately; Ctrl+C stops following only. `nohup`
continues the pipeline after SSH disconnects. Keep the printed log path.
Existing NumPy, pandas, PyArrow, PyTorch/PyG and project dependencies suffice;
do not replace the working CUDA environment.

## What it does

1. Reviews all 18 existing KIBA/DAVIS cold-drug test runs (three variants ×
   three seeds × two datasets). Checks data fingerprints, config consistency,
   checkpoint fingerprints, exact pair identities, labels, finite predictions,
   test coverage, and recomputed MSE/MAE/RMSE. Shows per-seed results and counts
   held-out drugs/scaffolds, their memberships in each PCBA partition, and
   exact entity/pair overlap between the two downstream datasets.
2. Computes paired drug-cluster bootstrap intervals for MSE and MAE differences
   and relative error reductions. Defaults: 5,000 replicates, RNG seed 20261103,
   pointwise 95% percentile intervals. The same sampled drugs and all their
   pairs are used for every model/seed. Original pair-weighted errors are
   preserved; predictions are NOT ensembled. Primary comparison is mean-across-
   fixed-seeds MSE of fine-tuning minus scratch. Per-seed contrasts and other
   model comparisons are also saved. No p-values or automatic significance
   declarations are produced.
3. Adds a `scaffold_seed42.json` split for each dataset if absent, or audits an
   existing one without modifying it. Copies each completed cold-drug run's
   saved config, changing only data paths, the split and output path. Trains
   18 new scaffold runs sequentially, then evaluates all 18. It reviews these
   results too; scaffold intervals resample scaffolds, not individual drugs.

New runs are under `runs/{kiba,davis}/scaffold/{scratch,frozen,finetune}/seed*`.
Existing cold-drug runs remain unchanged. Scientific settings are inherited
from saved configs rather than potentially edited YAML defaults. The PCBA
checkpoint and mean-pooled ESM features are reused.

## Reports

Every invocation gets a new directory below `reports/transfer_validation/`.
It contains `cold_drug/REPORT.md`, CSV per-seed/summary metrics, JSON exposure
and bootstrap reports, per-group error totals, input SHA256 manifests, and
locked scaffold configs. A successful full run adds `scaffold/REPORT.md` and
`results_bundle.zip`. Upload the bundle for interpretation; it contains no
model weights or full molecular prediction tables. Source data hashes and
aggregate group errors remain available for reproducibility.

For steps 1–2 only (no new training or splits):

```bash
bash validation_followup/start.sh "$PWD" --review-only
```

For reviewed results plus a scaffold plan, without creating scaffold splits
or launching training:

```bash
bash validation_followup/start.sh "$PWD" --plan-only
```

## Recovery and safety

Failures stop the sequence. Previously completed matching scaffold runs and
their existing test evaluations are reused on another invocation; changed
configs, writer locks and partial evaluations are explicit stop conditions.
Never delete results or a writer lock just to bypass a failure. For a genuine
interrupted scaffold run, first confirm no trainer is active, inspect its log,
then opt into epoch-boundary recovery with:

```bash
bash validation_followup/start.sh "$PWD" --resume-incomplete
```

The original trainer still enforces its lock and dataset/RNG resume checks.
If a writer lock remains after a crash, investigate it manually; the helper
never removes it. A second active follow-up pipeline is refused automatically.

## Statistical interpretation

- Seed sample SD measures training randomness; the cluster interval conditions
  on those fixed fitted models, seeds, split and pretraining checkpoint. It is
  NOT a combined interval over training, split and pretraining variability.
- Held-out groups are resampled uniformly with replacement; all their pairs
  remain together, including repeated copies. Different proteins can still
  be shared across drugs. Drug-only resampling is appropriate to this declared
  cold-drug question but not to a claim about unseen targets.
- Groups may not be exchangeable and chemical similarities can remain. Few
  held-out drugs/scaffolds make percentile intervals fragile; the reports warn
  below 20 groups. With one group no informative interval is reported.
- Intervals are pointwise, not adjusted for the multiple comparisons reported.
  The primary contrast was declared above. An interval excluding zero alone is
  not sufficient to establish broad scientific significance or causality.
- Compound overlap does not resolve assay/affinity endpoint leakage. PCBA
  validation affected checkpoint selection. Historical transfer-file hashes
  were not persisted by the original trainer; current paths do not prove that
  historical PCBA bytes never changed.
- DAVIS/KIBA labels use different scales. Do not compare their raw MSEs or
  claim wholly independent replication if entities/pairs overlap.
- This is still a single fixed scaffold split with three training seeds. Do
  not tune on the new test results or silently discard unfavorable runs.

## Local verification

```bash
cd validation_followup
python -m unittest -v test_validation
bash -n start.sh
```

Tests exercise prediction alignment, fingerprints, label/metric consistency,
split leakage, weighted cluster resampling against brute force, no-overwrite
checks, plan-only behavior, exact config inheritance, stop-on-failure, safe
completed-run reuse, and the ordering of all training before any test calls.
GPU training itself must run on your server; it is not claimed as locally run.
