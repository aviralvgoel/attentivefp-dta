# AttentiveFP for Drug–Target Affinity Prediction

This project tests whether supervised pretraining on **MoleculeNet PCBA** improves drug–protein binding-affinity prediction on **KIBA** and **DAVIS**. The drug branch is an AttentiveFP graph neural network. The protein branch uses cached, mean-pooled ESM-2 embeddings. A regression head takes the concatenated drug and protein representations and predicts one affinity score per pair.

The downstream comparison keeps the architecture fixed and varies how the drug encoder is trained: **scratch** starts from random weights, **frozen** keeps the pretrained weights fixed, and **fine-tuned** starts from pretrained weights and updates them on affinity data. The reported experiments use concatenation; the optional atom–residue cross-attention implementation is a separate experiment.

## Results

Test MSE, reported as mean ± sample standard deviation across training seeds 42, 43, and 44 on each fixed split. Lower is better. Relative changes compare fine-tuning with scratch within the same dataset and split.

| Dataset / split | Scratch | Frozen | Fine-tuned | Fine-tuning vs. scratch |
| --- | ---: | ---: | ---: | ---: |
| KIBA / cold-drug | 0.65265 ± 0.01497 | 0.67051 ± 0.00736 | 0.63136 ± 0.01706 | 3.3% lower |
| KIBA / scaffold | 0.64422 ± 0.02709 | 0.69612 ± 0.01313 | 0.65586 ± 0.00880 | 1.8% higher |
| DAVIS / cold-drug | 0.90039 ± 0.04690 | 0.75006 ± 0.05486 | 0.63529 ± 0.02416 | 29.4% lower |
| DAVIS / scaffold | 0.82622 ± 0.03509 | 0.68818 ± 0.01927 | 0.61880 ± 0.02401 | 25.1% lower |

Fine-tuning lowered mean MSE in three of four settings, but **all four paired 95% group-bootstrap intervals for the MSE difference included zero**. DAVIS held out only seven drugs or scaffolds per protocol, and one group accounted for about 92% of the net MSE gain in each protocol. The curated DAVIS drugs and scaffolds also map one-to-one, so the two protocols should not be treated as independent evidence for harder scaffold generalization without checking their partition IDs. These results suggest that the value of pretraining depends on the dataset and split.

Cold-drug and scaffold splits hold out drugs or chemical cores from **downstream affinity training**. They do not hold out proteins, and some downstream test compounds occurred in PCBA pretraining. KIBA scores and DAVIS pKd values use different scales; compare MSE within each benchmark. This is a binding-affinity benchmark, not a cancer drug-response or clinical efficacy model.

## Setup

Run commands from the project root. The package declares Python 3.10+ and PyTorch 2.4+. Use a PyTorch build that works with your GPU and driver before installing the project.

```bash
python -m pip install -e '.[esm,dev]'
python doctor.py --cuda
python -m pytest -q
python smoke.py --device cuda --output runs/smoke_gpu
```

On a machine without CUDA, run the smoke test with `--device cpu --output runs/smoke_cpu` and omit the CUDA diagnostic. Smoke runs use synthetic data and are execution checks, not benchmark results.

## Datasets and input formats

Raw datasets, ESM-2 weights, and trained checkpoints are not bundled with the project source. The commands below assume this layout:

| Input | Expected path | Source / format |
| --- | --- | --- |
| PCBA | `data/raw/pcba.csv.gz` | [MoleculeNet PCBA CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz); `smiles` and 128 `PCBA-` assay columns. This is not LIT-PCBA. |
| KIBA | `data/raw/kiba.csv` | [DeepDTA KIBA files](https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba); project CSV columns: `smiles,sequence,affinity`. |
| DAVIS | `data/raw/davis.csv` | [DeepDTA DAVIS files](https://github.com/hkmztrk/DeepDTA/tree/master/data/davis); project CSV columns: `smiles,sequence,affinity`, with raw affinity in nM for the command below. |

The DeepDTA folders contain `ligands_can.txt`, `proteins.txt`, and a `Y` affinity matrix, not the project CSVs. To prepare a trusted DeepDTA folder directly, replace `--input data/raw/kiba.csv` with `--deepdta-dir data/raw/kiba --trust-local-pickle`, or use the corresponding DAVIS folder. Only load `Y` from a source you trust: it is a Python pickle. The adapter keeps finite measurements and generates the requested project splits rather than importing the paper's folds.

## 1. Prepare PCBA and pretrain the drug encoder

```bash
python prepare_pcba.py \
  --input data/raw/pcba.csv.gz \
  --output data/pcba/processed \
  --smiles-column smiles --task-prefix PCBA- \
  --shard-size 2048 --seed 42

python audit.py pcba \
  --processed-dir data/pcba/processed \
  --split-file data/pcba/processed/splits/scaffold_seed42.json \
  --output reports/pcba_audit.json

python train_pcba.py --config configs/pcba.yaml
```

RDKit canonicalizes SMILES while retaining stereochemistry, charges, and fragments. The shared graph schema uses 45 atom features and 14 bond features, with each bond stored in both directions. PCBA duplicates are merged by canonical SMILES. Missing assay values and conflicting duplicate labels are masked rather than treated as inactive. The full-data preprocessing run produced 437,927 molecules and 128 tasks.

The preprocessor writes graph shards, `index.parquet`, `labels.npy`, `label_mask.npy`, and split JSON files. Pretraining learns a 512-dimensional drug representation with masked binary cross-entropy and selects a checkpoint by validation macro average precision. The default checkpoint used for downstream transfer is `runs/pcba/scaffold_seed42/best.pt`. Only the drug encoder is transferred, not the assay classification head.

If you already have compatible processed data and the checkpoint, reuse them. The feature schema must match across pretraining and downstream data; matching tensor dimensions alone is not enough. See [legacy PCBA notes](docs/LEGACY_PCBA.md) for older graph files.

## 2. Prepare the downstream datasets and protein cache

```bash
python prepare_dta.py --input data/raw/kiba.csv \
  --dataset kiba --label-transform none \
  --split-mode cold_drug --seed 42 --output data/kiba/processed

python prepare_dta.py --input data/raw/davis.csv \
  --dataset davis --label-transform neg_log10_nM \
  --split-mode cold_drug --seed 42 --output data/davis/processed
```

KIBA scores are retained as supplied. For DAVIS, `neg_log10_nM` computes `pKd = 9 - log10(Kd in nM)`. If your input already contains pKd, use `--label-transform none` to avoid transforming it twice. The script validates molecules and protein sequences and merges duplicate canonical drug–sequence pairs by the median of the transformed labels **before splitting**. It stores one graph per drug, one entry per unique protein, and the processed pair table.

Create scaffold splits and cache a mean-pooled ESM-2 vector for every unique protein:

```bash
for dataset in kiba davis; do
  python make_split.py --processed-dir "data/$dataset/processed" \
    --mode scaffold --seed 42 || break

  python precompute_esm.py --processed-dir "data/$dataset/processed" \
    --output "data/$dataset/esm2_mean.pt" \
    --model facebook/esm2_t12_35M_UR50D --pooling mean --device cuda || break
done
```

Cold-drug splitting keeps all pairs for a drug together. Scaffold splitting groups drugs by their Murcko chemical core. Both aim for 80/10/10 train/validation/test proportions without breaking groups to hit the exact ratios. ESM-2 pooling excludes special and padding tokens. The default extraction processes sequences longer than 1,000 residues in non-overlapping chunks rather than truncating them. ESM-2 remains frozen during downstream training.

## 3. Train the affinity models

The default KIBA configurations use the cold-drug split and training seed 42:

```bash
python train_dta.py --config configs/kiba_scratch.yaml
python train_dta.py --config configs/kiba_frozen.yaml
python train_dta.py --config configs/kiba_finetune.yaml
```

For DAVIS, use `configs/davis_scratch.yaml` or `configs/davis_finetune.yaml`. The frozen DAVIS condition uses the latter configuration with the transfer mode and output path overridden:

```bash
python train_dta.py --config configs/davis_finetune.yaml \
  --set transfer.mode=frozen --set transfer.warmup_epochs=0 \
  --set training.output_dir=runs/davis/cold_drug/frozen/seed42
```

To repeat a run with a different training seed or split, change both the corresponding setting and the output directory. For example:

```bash
python train_dta.py --config configs/kiba_finetune.yaml \
  --set seed=43 \
  --set data.split_file=data/kiba/processed/splits/scaffold_seed42.json \
  --set training.output_dir=runs/kiba/scaffold/finetune/seed43
```

Here, `seed=43` changes the training randomness, while `scaffold_seed42.json` keeps the data partition fixed. The reported comparison uses the same split and training seeds across variants. Label standardization uses training-set statistics only; test metrics are reported on the original KIBA-score or pKd scale.

For an SSH session, `bash scripts/run_training.sh dta configs/kiba_finetune.yaml` launches training under `nohup` and follows the log. Use it **instead of** the corresponding foreground command. Ctrl+C stops log following, not training. Choose a new output directory for a new experiment; use `--resume` with the run's `last.pt` to resume at an epoch boundary.

## 4. Evaluate and audit

Training selects `best.pt` on validation data. Evaluate that checkpoint separately on the held-out test split:

```bash
python evaluate.py \
  --checkpoint runs/kiba/cold_drug/finetune/seed42/best.pt \
  --split test --device cuda
```

The evaluator writes metrics and pair-level predictions under `evaluation_test/` in the run directory. It reports regression metrics including MSE, RMSE, MAE, and concordance index. Use validation results to select configurations, not the held-out test results.

Audit a downstream split and record whether its drugs or scaffolds appear in the PCBA training partition:

```bash
python audit.py dta \
  --processed-dir data/kiba/processed \
  --split-file data/kiba/processed/splits/scaffold_seed42.json \
  --output reports/kiba_scaffold_audit.json

python audit_overlap.py \
  --pcba-dir data/pcba/processed \
  --pcba-split data/pcba/processed/splits/scaffold_seed42.json \
  --dta-dir data/kiba/processed \
  --dta-split data/kiba/processed/splits/scaffold_seed42.json \
  --output reports/kiba_scaffold_pcba_exposure.json
```

Compound exposure is not by itself evidence of affinity-label leakage. For a new evaluation, compare variants on the same test pairs and resample whole held-out drugs or scaffolds when estimating uncertainty.

## Project files and reproducibility

| Location | Purpose |
| --- | --- |
| `src/dta_project/` | Data loaders, graph features, models, training, and evaluation |
| `configs/` | PCBA and downstream training settings |
| [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | Processed formats and feature compatibility |
| `runs/` | Checkpoints, resolved configurations, logs, and evaluation outputs |
| `reports/` | Data and split audit reports |

To reproduce the reported runs, keep the exact raw inputs, processed files, split JSONs, protein cache, transferred checkpoint, and resolved run configurations. Record the software environment and checkpoint hashes alongside the outputs. Downloading the public benchmarks again does not guarantee the same input rows or split membership.

If an older checkout fails on a resolved JSON configuration with an optimizer value parsed as a string, use the YAML configuration with decimal-valued overrides, or update the configuration loader to parse JSON with `json.loads`. If RDKit reports a bond-stereo error when building a non-chiral scaffold, use the corrected scaffold routine that clears stereo on a temporary molecule used only for grouping. Keep the original stereochemistry in canonical identities and graph features.
