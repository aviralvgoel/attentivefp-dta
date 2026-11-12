# AttentiveFP drug target affinity training project

This is the canonical model-training folder for the attached DTA roadmap.
It includes PCBA molecular pretraining, KIBA/DAVIS curation, frozen ESM-2 features,
scratch/frozen/fine-tuned AttentiveFP variants, optional atom-residue attention,
leakage audits, epoch-boundary resume, logging, and explicit test evaluation.

The primary task is **continuous drug-target affinity regression**, not cancer
cell-line drug response. No patient-level or clinical claims are made. This is a
training implementation, not evidence of an accuracy improvement.

## Start here on your server

Extract the archive into a new folder. Do not replace the old demo or interrupt
an existing run just to change folders. Activate the environment in which your
Blackwell GPU already works:

```bash
conda activate attentivefp-pcba
cd /home/dmlab/Aviral/Attentive_FP/attentivefp_dta_final
python -m pip install -e '.[esm,dev]'
python doctor.py --cuda
python -m pytest -q
python smoke.py --device cuda --output runs/smoke_gpu
```

`doctor.py` tests an actual AttentiveFP forward/backward, including a zero-bond
molecule. It is a stronger check than only inspecting the architecture list.
Do not uninstall a working PyTorch build. The project requires PyTorch >=2.4;
your CUDA build and driver must also support your actual GPU. If necessary use
the [official PyTorch installer](https://pytorch.org/get-started/locally/) and
rerun the doctor. `torchvision`, `torchaudio`, and optional compiled PyG
extensions are not required for this implementation.

The smoke run uses explicit **synthetic labels and synthetic protein vectors**.
It tests execution only; its scores must not be reported as scientific results.
For a CPU-only installation, install CPU PyTorch first and use `--device cpu`.

## 1 Continue with existing PCBA data

You do not need to rerun preprocessing merely to train. The adapter config uses
the real server paths from the previous demo:

```bash
python audit.py pcba \
  --processed-dir /home/dmlab/Aviral/Attentive_FP/data/processed \
  --split-file /home/dmlab/Aviral/Attentive_FP/data/splits/scaffold_seed42.json \
  --benchmark-batches 200 --batch-size 512 --workers 0 \
  --output reports/existing_pcba_audit.json

bash scripts/run_training.sh pcba configs/pcba_existing_server.yaml
```

The launcher starts a background `nohup` process, follows its log immediately,
records the PID, and logs GPU utilization/memory/power when `nvidia-smi` is
available. Ctrl+C stops log following; training continues. It logs every 100
batches and every validation epoch. The first pass computes class weights and
hashes the data, which can take some time on a large dataset.

Use a new run name or `--resume`; nonempty output folders are never silently
overwritten. To reduce memory pressure:

```bash
bash scripts/run_training.sh pcba configs/pcba_existing_server.yaml \
  --set training.batch_size=128 --set training.workers=2 \
  --set training.output_dir=runs/pcba/existing_small_batch
```

Important: existing PCBA graphs may use a different featurizer. Pretraining on
them is supported, but **transfer into new DTA graphs is blocked until feature
compatibility is established**. See [legacy PCBA instructions](docs/LEGACY_PCBA.md).
Matching node/edge dimensions alone is insufficient.

## 2 Prepare PCBA with the shared featurizer if needed

This path makes new compatible data without modifying existing preprocessing:

```bash
python prepare_pcba.py \
  --input data/raw/pcba.csv.gz \
  --output data/pcba/processed \
  --smiles-column smiles --task-prefix PCBA- \
  --shard-size 2048 --seed 42

python audit.py pcba \
  --processed-dir data/pcba/processed \
  --split-file data/pcba/processed/splits/scaffold_seed42.json \
  --check-features --output reports/pcba_audit.json

bash scripts/run_training.sh pcba configs/pcba.yaml
```

This is MoleculeNet PCBA, normally 128 assays. **LIT-PCBA is a different dataset**;
do not relabel it or infer its format from the name. The loader/model infer the
task count rather than hard-coding 128. `--task-columns` supports explicit task
names. The preprocessor retains an in-memory label table; budget several GB of
host RAM for a full PCBA collection. Graphs are built and saved in shards.

Duplicate canonical molecules are merged. Conflicting observed 0/1 labels are
masked as unknown. Missing assay labels are never treated as inactive. Invalid
SMILES raise by default; `--invalid drop` records rejected input rows.

The bundled features are versioned (`aviral_graph_v1`, 45 atom and 14 bond
features). They retain salts, stereochemistry and formal charges; there is no
silent largest-fragment selection or tautomer normalization.

## 3 Prepare KIBA and DAVIS

No research datasets, trained weights, or ESM model weights are bundled.
Provide a CSV with `smiles,sequence,affinity`; see `examples/dta_input.csv` for
the schema (its labels are illustrative). For curated KIBA scores:

```bash
python prepare_dta.py --input data/raw/kiba.csv \
  --dataset kiba --label-transform none \
  --split-mode cold_drug --seed 42 --output data/kiba/processed

python make_split.py --processed-dir data/kiba/processed --mode random --seed 42
python make_split.py --processed-dir data/kiba/processed --mode scaffold --seed 42

python audit.py dta --processed-dir data/kiba/processed \
  --split-file data/kiba/processed/splits/cold_drug_seed42.json \
  --output reports/kiba_cold_drug_audit.json
```

For raw DAVIS Kd values in **nM**, explicitly request the transform
`pKd = 9 - log10(Kd_nM)`:

```bash
python prepare_dta.py --input data/raw/davis.csv \
  --dataset davis --label-transform neg_log10_nM \
  --split-mode cold_drug --seed 42 --output data/davis/processed
```

If the DAVIS labels are already pKd, use `--label-transform none` instead.
Never apply the logarithm twice or mix Ki, Kd, IC50 and KIBA scores as though
they had the same units. BindingDB input must already be curated to one endpoint
and consistent units. Censored values such as `>10000` are rejected, not parsed
as exact measurements.

For trusted local DeepDTA-format files (`ligands_can.txt`, `proteins.txt`, `Y`):

```bash
python prepare_dta.py --deepdta-dir data/raw/kiba \
  --trust-local-pickle --dataset kiba --label-transform none \
  --split-mode cold_drug --output data/kiba/processed
```

The explicit trust flag is required because `Y` is a Python pickle. Do not use
it on arbitrary untrusted files. This adapter creates our declared splits, not
the original paper's folds; its results are not automatically comparable to
published numbers.

Exact canonical drug/sequence duplicates are aggregated by median transformed
label before splitting. Drug IDs are derived from canonical SMILES; protein IDs
from exact normalized sequences. Raw labels and replicate counts are retained.

Also report whether downstream test drugs appeared in PCBA pretraining:

```bash
python audit_overlap.py --pcba-dir data/pcba/processed \
  --pcba-split data/pcba/processed/splits/scaffold_seed42.json \
  --dta-dir data/kiba/processed \
  --dta-split data/kiba/processed/splits/cold_drug_seed42.json \
  --output reports/pretraining_exposure.json
```

A drug absent from downstream training is not necessarily absent from PCBA.
This report makes that distinction explicit; it does not resolve overlapping
assay endpoints automatically or remove/retrain any data without your choice.

## 4 Cache frozen ESM-2 representations

The default 35M ESM-2 is a practical starting model. The first call downloads
public pretrained weights; later calls reuse the local cache. Pin a Hugging
Face model commit with `--revision` for final reproducibility.

```bash
python precompute_esm.py --processed-dir data/kiba/processed \
  --output data/kiba/esm2_mean.pt \
  --model facebook/esm2_t12_35M_UR50D --pooling mean --device cuda

python precompute_esm.py --processed-dir data/davis/processed \
  --output data/davis/esm2_mean.pt \
  --model facebook/esm2_t12_35M_UR50D --pooling mean --device cuda
```

CLS/EOS/padding tokens are excluded. Sequences longer than 1000 residues are
processed as **independent non-overlapping chunks**, not silently truncated.
The resulting representation lacks full cross-chunk context; this policy is
recorded in the cache. Sequence hashes reject stale protein/embedding matches.
Cached ESM remains frozen; end-to-end ESM fine-tuning is not implemented.

## 5 Run the controlled transfer experiments

All model variants use the same split and feature schema:

```bash
bash scripts/run_training.sh dta configs/kiba_scratch.yaml
bash scripts/run_training.sh dta configs/kiba_frozen.yaml
bash scripts/run_training.sh dta configs/kiba_finetune.yaml
```

For the pretrained variants, PCBA `hidden_dim`, `embedding_dim`, `num_layers`
and `num_timesteps` must match the drug architecture in the DTA config.
Override `transfer.checkpoint` if using a different compatible checkpoint.
Only the encoder weights transfer; the PCBA assay head is discarded.

Fine-tuning freezes the encoder for the first 5 epochs, then unfreezes it with
a smaller learning rate. A frozen encoder also remains in evaluation mode, so
dropout cannot introduce random fixed features.

Run three training seeds (42, 43, 44) on fixed random-pair and cold-drug splits:

```bash
bash scripts/run_matrix.sh
```

The matrix is validation-only and sequential. Ensure both split files and the
compatible PCBA checkpoint exist first. DAVIS configs are also included.

## 6 Optional atom residue cross attention

First finish the pooled-concatenation baseline. For attention, save residue
features rather than a single pooled protein vector:

```bash
python precompute_esm.py --processed-dir data/kiba/processed \
  --output data/kiba/esm2_residues.pt --pooling residues --device cuda
bash scripts/run_training.sh dta configs/kiba_attention.yaml
```

This extension uses final AttentiveFP atom states as queries and ESM residue
states as keys/values, then combines the pooled interaction with the molecular
embedding. It is **not a reproduction of DrugBAN**. It uses a tested PyG GRU
hook to expose atom states without renaming pretrained weights; it is not
designed for `torch.compile`, DDP or multi-GPU training. Padded atoms/residues
are masked. Memory scales with atoms × residues; start with batch size 16.
Attention maps are model diagnostics, not experimentally verified contacts.

## 7 Resume an interrupted run

```bash
bash scripts/run_training.sh dta configs/kiba_finetune.yaml \
  --resume runs/kiba/cold_drug/finetune/seed42/last.pt
```

Resume is at epoch boundaries. A partially completed epoch restarts from the
last completed epoch; it does not resume from the exact batch. Model, optimizer,
scheduler, scaler, RNG state, early stopping and dataset fingerprints are saved.
Do not change scientific settings during resume. Device, workers and progress
settings may change, but CUDA nondeterminism/hardware changes can affect exact
bitwise reproducibility. A `.writer.lock` prevents concurrent writers; after a
hard crash, confirm its PID is no longer running before removing that exact
stale lock file. Keep both `last.pt` and `best.pt` in the original run folder.

## 8 Evaluate test only after selecting configurations

```bash
python evaluate.py \
  --checkpoint runs/kiba/cold_drug/finetune/seed42/best.pt \
  --split test --device cuda
```

Training does **not** load test pairs into a training/validation loader or
compute their metrics. Evaluation is a separate explicit command. Dataset,
split, graph and protein-cache fingerprints are checked. Move a dataset using
`--processed-dir`, `--split-file` and `--protein-cache` overrides; its bytes and
split must remain unchanged. Evaluation output is not overwritten silently.

Each training run contains:

| File | Purpose |
|---|---|
| `best.pt` | Validation-selected checkpoint and complete training state |
| `last.pt` | Last completed epoch for recovery |
| `config.json`, `environment.json` | Resolved settings and package/GPU versions |
| `dataset_signature.json` | Exact input fingerprints |
| `history.jsonl`, `train.log` | Per-epoch metrics and progress |
| `validation_metrics.json`, `metrics.json` | Selected validation results; no test score |
| `evaluation_test/metrics.json` | Explicit held-out result |
| `evaluation_test/predictions.parquet` | DTA predictions with pair/drug/protein IDs |

PCBA exports `predictions.npz` with molecule IDs, labels, masks and probabilities.
The PCBA ranking metric is **macro average precision**, not trapezoidal PR-AUC;
ROC-AUC is secondary and only assays with both observed classes are included.
Undefined values are JSON `null`, with evaluable task counts reported. BCE
fallback is disabled in full runs and enabled only in the smoke fixture.

DTA reports MSE, RMSE, MAE, CI, Pearson/Spearman and a training-mean baseline.
Normalization uses training labels only; predictions/metrics are on the original
transformed-label scale (e.g. pKd or KIBA score). CI uses an O(N log N) algorithm,
not a quadratic pair matrix. Compare mean ± SD across seeds; for cold-start
uncertainty use grouped resampling by held-out drug/scaffold rather than treating
all drug-target pairs as independent observations.

## Project files

Root launchers have one job each: `prepare_pcba.py`, `train_pcba.py`,
`prepare_dta.py`, `make_split.py`, `precompute_esm.py`, `train_dta.py`,
`evaluate.py`, `audit.py`, `import_legacy.py`, `doctor.py`, `smoke.py`.
Implementation modules live under `src/dta_project/`; editable YAMLs under
`configs/`; regression tests under `tests/`. `docs/DATA_CONTRACT.md` describes
on-disk formats. `docs/LEGACY_PCBA.md` covers the previous project.

## Limits of this release

- Real full-data training, convergence and Blackwell execution must be checked
  on your server; local integration testing is CPU and uses synthetic labels.
- ESM extraction is tested with a locally initialized tiny ESM model, not a
  claim that downloaded pretrained ESM weights improve downstream performance.
- Cold-target splits enforce exact-sequence separation, not sequence-cluster
  separation; homologous proteins may remain across partitions.
- Cold-both splits exclude cross-partition endpoint pairs explicitly; fractions
  describe entity assignments, not final retained pair counts.
- Protein similarity clustering, 3D chemistry, full BindingDB curation,
  trainable ESM, distributed training and clinical validation are out of scope.

API references: [PyG AttentiveFP](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html),
[Hugging Face ESM](https://huggingface.co/docs/transformers/model_doc/esm),
[GraphDTA author repository](https://github.com/thinng/GraphDTA),
[DeepDTA author repository](https://github.com/hkmztrk/DeepDTA).
