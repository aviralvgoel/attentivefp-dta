"""Convert MoleculeNet PCBA CSV into compatible graph shards and sparse labels.

The full input label table is held in host RAM; graph construction is sharded.
This is MoleculeNet PCBA (128 tasks), not LIT-PCBA (a different benchmark).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .chemistry import SCHEMA, SCHEMA_HASH, canonicalize, scaffold, smiles_to_graph
from .common import sha256, write_json
from .splits import group_partition


def prepare(frame, tasks, output, seed=42, shard_size=2048, invalid="error", source=None):
    if not tasks or "smiles" not in frame or not set(tasks) <= set(frame):
        raise ValueError("Require smiles and the declared assay label columns")
    if len(tasks) != len(set(tasks)) or shard_size < 1:
        raise ValueError("Task names must be unique and shard_size positive")
    frame = frame[["smiles"] + tasks].copy()
    frame[tasks] = frame[tasks].apply(pd.to_numeric, errors="raise")
    values = frame[tasks].to_numpy()
    observed = ~pd.isna(values)
    if not np.isfinite(values[observed]).all() or not np.isin(values[observed], [0, 1]).all():
        raise ValueError("PCBA observed labels must be exactly 0 or 1; blank labels mean missing")
    canonical, bad = [], []
    for i, smi in enumerate(tqdm(frame.smiles, desc="Canonicalizing")):
        try:
            canonical.append(canonicalize(smi))
        except ValueError as exc:
            if invalid == "error":
                raise ValueError(f"Invalid row {i}: {exc}") from exc
            canonical.append(None)
            bad.append(i)
    frame["canonical_smiles"] = canonical
    frame = frame[frame.canonical_smiles.notna()]
    grouped = frame.groupby("canonical_smiles", sort=True)[tasks]
    lo, hi = grouped.min(), grouped.max()
    mask = lo.notna().to_numpy() & (lo.to_numpy() == hi.to_numpy())
    labels = np.where(mask, lo.fillna(0).to_numpy(), 0).astype(np.float32)
    smiles = lo.index.tolist()
    if len(smiles) < 3:
        raise ValueError("Too few unique valid compounds")
    scaffolds = [scaffold(s) for s in smiles]
    # Compute split before writing any output, so too few scaffold groups fail cleanly.
    scaffold_split = group_partition(scaffolds, seed)
    random_split = group_partition(list(range(len(smiles))), seed)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output not empty: {out}")
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    np.save(out / "labels.npy", labels, allow_pickle=False)
    np.save(out / "label_mask.npy", mask, allow_pickle=False)
    rows, pending, shard_no, hashes = [], [], 0, {}

    def flush():
        nonlocal shard_no, pending
        name = f"shard_{shard_no:05d}.pt"
        torch.save(pending, out / "graphs" / name)
        hashes[name] = sha256(out / "graphs" / name)
        shard_no += 1
        pending = []

    for i, smi in enumerate(tqdm(smiles, desc="Graph features")):
        rows.append(
            {
                "global_index": i,
                "shard_filename": f"shard_{shard_no:05d}.pt",
                "offset": len(pending),
                "canonical_smiles": smi,
                "scaffold": scaffolds[i],
            }
        )
        pending.append(smiles_to_graph(smi))
        if len(pending) == shard_size:
            flush()
    if pending:
        flush()
    pd.DataFrame(rows).to_parquet(out / "index.parquet", index=False)
    for name, split in (("scaffold", scaffold_split), ("random", random_split)):
        split["metadata"] = {
            "mode": name,
            "seed": seed,
            "index_sha256": sha256(out / "index.parquet"),
        }
        write_json(out / "splits" / f"{name}_seed{seed}.json", split)
    write_json(
        out / "metadata.json",
        {
            "format": "pcba_v1",
            "schema": SCHEMA,
            "schema_hash": SCHEMA_HASH,
            "tasks": tasks,
            "molecules": len(rows),
            "invalid_rows": bad,
            "source": source,
            "conflicting_labels_masked": int(
                ((lo != hi) & lo.notna() & hi.notna()).to_numpy().sum()
            ),
            "shard_hashes": hashes,
        },
    )
    print(f"Saved {len(rows):,} molecules and {len(tasks)} tasks to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--task-prefix", default="PCBA-")
    parser.add_argument("--task-columns", nargs="+", help="Overrides --task-prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--invalid", choices=["error", "drop"], default="error")
    args = parser.parse_args()
    columns = pd.read_csv(args.input, nrows=0).columns.tolist()
    tasks = args.task_columns or [c for c in columns if c.startswith(args.task_prefix)]
    if not tasks:
        parser.error("No task columns selected")
    frame = pd.read_csv(args.input, usecols=[args.smiles_column] + tasks).rename(
        columns={args.smiles_column: "smiles"}
    )
    prepare(
        frame,
        tasks,
        args.output,
        args.seed,
        args.shard_size,
        args.invalid,
        {"sha256": sha256(args.input)},
    )


if __name__ == "__main__":
    main()
