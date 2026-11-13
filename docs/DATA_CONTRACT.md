# On disk data contracts

All PyG graph files must be trusted local preprocessing output. Deserializing
arbitrary graph/pickle files can execute Python. Model/cache checkpoints use
`torch.load(weights_only=True)`; graph objects necessarily use trusted object loading.

## PCBA processed directory

| File | Schema |
|---|---|
| `index.parquet` | Unique nonnegative integer global_index; shard_filename; offset; canonical_smiles; scaffold |
| `labels.npy` | float32 `[N, tasks]`; observed entries exactly 0/1 |
| `label_mask.npy` | bool `[N, tasks]`; false means unknown, not inactive |
| `graphs/shard_*.pt` | Trusted lists of Data with x `[atoms,45]`, edge_index `[2,E]`, edge_attr `[E,14]` for the bundled schema |
| `metadata.json` | schema_hash, explicit feature definitions, task order, source and shard hashes |
| `splits/*.json` | Lists of global sample IDs, with no overlap or silent omissions |

Old data may have other feature dimensions for pretraining, but cannot be
transferred with the bundled DTA featurizer until schema equivalence is verified.
An assay without positive/negative observations is excluded from AP/ROC-AUC;
evaluation fails if the entire selected partition has no observed labels.

## DTA input

Required CSV columns: `smiles`, `sequence`, `affinity`. Optional `target_id` is
checked for conflicting sequences. Blank affinity, nonnumeric bounds, invalid
SMILES and invalid sequences fail by default. `--invalid drop` explicitly records
input row rejections. Exact pair duplicates are merged before splitting.

DeepDTA JSON/matrix format is supported with explicit local-pickle trust. We do
not download research datasets, combine incompatible endpoints or infer units.

## DTA processed directory

| File | Schema |
|---|---|
| `pairs.parquet` | pair_id; drug_id; protein_id; canonical_smiles; protein_sequence; raw_label; label; scaffold; replicate_count |
| `drugs.parquet` | Unique drug_id; canonical_smiles; scaffold |
| `proteins.parquet` | Unique protein_id; protein_sequence |
| `drug_graphs.pt` | Trusted mapping drug_id to a molecular PyG graph |
| `metadata.json` | dataset, endpoint transformation, schema_hash and source provenance |
| `splits/*.json` | Pair IDs, split mode, seed, fractions, pair-table fingerprint |

Random-pair split groups only exact pairs. `cold_drug` groups exact canonical
drugs. `scaffold` additionally groups Bemis-Murcko cores; all acyclic molecules
share the empty scaffold and remain together. Group size can make fractions
imbalanced; inspect audit counts. `cold_target` groups exact sequences, not
sequence similarity clusters. `cold_both` retains only pairs with both endpoints
in the same assigned partition; discarded cross pairs are listed under `excluded`.

## Frozen protein cache

The `.pt` payload contains:

- `embeddings`: protein_id to float16 pooled vector `[H]` or residue states `[L,H]`.
- `sequence_sha256`: hash of the exact normalized full sequence for every entry.
- `metadata`: model/revision, pooling policy, chunked proteins, input fingerprint,
  and `synthetic=false` for genuine ESM extraction.

Atom-residue attention requires `pooling=residues`. The scratch/pooled model
supports either representation, using a masked mean for residue caches. No
padding, CLS or EOS tokens contribute to pooling. Synthetic fixture caches are
blocked unless explicitly allowed in a smoke config.

## Run and evaluation

Resolved config paths refer to the shell's project directory, not to the YAML
directory. `extends` paths are relative to their YAML file. `--set KEY=VALUE`
uses YAML scalars and rejects unknown dotted keys. Run names are user-controlled;
different splits, hyperparameters or transfer modes require different directories.

`best.pt` and `last.pt` save task/model architecture, weights, optimizer/scheduler/
scaler, RNG, selection policy, training-label normalization and dataset/cache
fingerprints. Resume is only from the same run's `last.pt` with scientific config
unchanged. History is append-only. JSON undefined metrics use `null`.

Test exports are written separately. DTA rows retain stable IDs and transformed
label units. Keep test predictions for paired comparisons and group-aware
uncertainty estimates; do not tune on their scores.
