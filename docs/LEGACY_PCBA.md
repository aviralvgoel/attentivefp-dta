# Reusing the earlier PCBA project

The earlier folder and data are never edited by this project. The existing-server
config reads their absolute data paths but writes checkpoints to this new folder.
Do not use the small `smoke_pcba` fixture as the real dataset.

## What remains compatible

- `index.parquet` with `global_index`, `shard_filename`, `offset`.
- `graphs/shard_*.pt` containing lists of PyG Data, `{'graphs': list}`, or
  `(data, slices)` collated objects.
- `labels.npy` and boolean `label_mask.npy`, or labels/masks inside each graph.
- Split JSON with `train`, `validation` (or `val`), `test` global IDs.

The label batch bug is fixed: each graph's targets/mask are `[1, tasks]`, so a
batch is `[B, tasks]`. Observed NaNs fail loudly rather than becoming negatives.
The graph's stable ID is called `molecule_id`, because PyG automatically increments
attribute names containing `index` during batching. Exported IDs are now preserved.
Positive weights use label arrays, without an expensive graph pass when arrays
exist. Shard-local batches reduce random disk rereads; they change the training
order relative to the old demo and should be used consistently across comparisons.

## Feature compatibility before transfer

An old 512-dimensional embedding is not a featurizer specification. The input
atom/bond order and meaning must match. This release uses a specific versioned
45/14-dimensional schema, shared by both preprocessors. If the old features differ,
do not force-load the encoder or pad/truncate features to make the dimensions fit.

Start with a read-only sample audit:

```bash
python audit.py pcba \
  --processed-dir /home/dmlab/Aviral/Attentive_FP/data/processed \
  --split-file /home/dmlab/Aviral/Attentive_FP/data/splits/scaffold_seed42.json \
  --check-features --sample-size 64 --output reports/legacy_feature_sample.json
```

If there is a mismatch, either reprocess/pretrain using the shared featurizer into
a NEW data directory, or port the known original featurizer into both pipelines
with a new schema version and tests. No automatic conversion can recover semantic
feature definitions just from tensor dimensions.

If every sample matches, run a complete verification before importing a checkpoint:

```bash
python audit.py pcba \
  --processed-dir /home/dmlab/Aviral/Attentive_FP/data/processed \
  --split-file /home/dmlab/Aviral/Attentive_FP/data/splits/scaffold_seed42.json \
  --check-features --all-graphs --output reports/legacy_feature_full.json

python import_legacy.py \
  --checkpoint /home/dmlab/Aviral/Attentive_FP/pcba_attentivefp_demo/runs/pcba_full_scaffold_seed42/best.pt \
  --verified-audit reports/legacy_feature_full.json \
  --processed-dir /home/dmlab/Aviral/Attentive_FP/data/processed \
  --output runs/imported_pcba/encoder_transfer.pt
```

Use `--set transfer.checkpoint=runs/imported_pcba/encoder_transfer.pt` in downstream
training. This creates a separate transfer checkpoint; it does not rewrite the
old checkpoint or claim to resume its optimizer. You must confirm that the source
checkpoint was actually trained on the audited data: legacy checkpoints without
dataset fingerprints cannot prove that provenance automatically. A feature audit
does not rule out PCBA/downstream supervised assay-label overlap; document any
overlapping compounds/endpoints in your research protocol before cold-start claims.

## No need to restart an existing healthy training process

Let it finish. This new code is for subsequent runs and downstream modeling.
Do not replace modules underneath a running Python process or try to resume the
old checkpoint with the new trainer. Use the original script to finish the old
run, or begin a new canonical run with the new trainer.
