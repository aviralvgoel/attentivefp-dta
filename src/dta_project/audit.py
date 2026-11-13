"""Audit PCBA masks/features or DTA pair/entity leakage; optional throughput test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chemistry import SCHEMA_HASH, smiles_to_graph
from .common import sha256, write_json
from .pcba import PCBADataset, make_pcba_loader
from .splits import PARTITIONS, audit_dta_split, check_indices


def audit_pcba(directory, split_file, sample_size=64, check_features=False, all_graphs=False):
    full = PCBADataset(directory)
    split = json.loads(Path(split_file).read_text())
    partitions = check_indices(split, full.index.global_index)
    report = {
        "task": "pcba",
        "molecules": len(full),
        "split_counts": {k: len(v) for k, v in partitions.items()},
        "index_sha256": sha256(Path(directory) / "index.parquet"),
        "split_sha256": sha256(split_file),
        "schema_hash": full.metadata.get("schema_hash"),
        "verified_features": 0,
        "all_graph_features_verified": False,
    }
    if "canonical_smiles" in full.index and full.index.canonical_smiles.duplicated().any():
        raise ValueError("Duplicate canonical SMILES in PCBA index")
    if split.get("metadata", {}).get("mode") == "scaffold" or "scaffold" in Path(split_file).name:
        if "scaffold" not in full.index:
            from .chemistry import scaffold

            if "canonical_smiles" not in full.index:
                raise ValueError("Scaffold audit needs scaffold or canonical_smiles column")
            full.index["scaffold"] = full.index.canonical_smiles.map(scaffold)
        groups = {
            part: set(full.index.loc[full.index.global_index.isin(ids), "scaffold"])
            for part, ids in partitions.items()
        }
        from itertools import combinations

        for a, b in combinations(PARTITIONS, 2):
            if groups[a] & groups[b]:
                raise ValueError(f"Scaffold overlap in PCBA {a}/{b}")
        report["scaffold_disjoint"] = True
    if full.labels is not None:
        report["label_counts"] = {}
        for part, ids in partitions.items():
            counts = np.zeros((3, full.labels.shape[1]), dtype=np.int64)
            ids = np.array(sorted(ids))
            for start in range(0, len(ids), 8192):
                ix = ids[start : start + 8192]
                y, m = full.labels[ix], full.mask[ix]
                if not np.isfinite(y[m]).all() or not np.isin(y[m], [0, 1]).all():
                    raise ValueError(f"Invalid observed labels in {part}")
                counts += np.stack([m.sum(0), ((y == 1) & m).sum(0), ((y == 0) & m).sum(0)])
            report["label_counts"][part] = {
                "observed": counts[0].tolist(),
                "positive": counts[1].tolist(),
                "negative": counts[2].tolist(),
            }
    selected = (
        range(len(full))
        if all_graphs
        else sorted(
            np.random.default_rng(42).choice(len(full), min(len(full), sample_size), replace=False)
        )
    )
    for i in selected:
        graph = full[int(i)]
        if check_features:
            if "canonical_smiles" not in full.index:
                raise ValueError("Feature verification requires canonical_smiles in index.parquet")
            expected = smiles_to_graph(full.index.iloc[int(i)].canonical_smiles)
            for key in ("x", "edge_index", "edge_attr"):
                a, b = graph[key], expected[key]
                if a.shape != b.shape or not torch.allclose(
                    a.float(), b.float(), atol=1e-6, rtol=1e-6
                ):
                    raise ValueError(
                        f"Feature mismatch at global_index={int(graph.molecule_id[0])}, tensor={key}. Do not transfer this checkpoint using the bundled featurizer."
                    )
            report["verified_features"] += 1
    if check_features and all_graphs:
        report["all_graph_features_verified"] = True
        report["verified_schema_hash"] = SCHEMA_HASH
        report["shard_hashes"] = {
            name: sha256(Path(directory) / "graphs" / name)
            for name in full.index.shard_filename.unique()
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["pcba", "dta"])
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/audit.json"))
    parser.add_argument("--check-features", action="store_true")
    parser.add_argument(
        "--all-graphs",
        action="store_true",
        help="Full legacy feature verification (potentially slow)",
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--benchmark-batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.task == "dta":
        frame = pd.read_parquet(args.processed_dir / "pairs.parquet")
        result = audit_dta_split(frame, json.loads(args.split_file.read_text()))
    else:
        result = audit_pcba(
            args.processed_dir,
            args.split_file,
            args.sample_size,
            args.check_features,
            args.all_graphs,
        )
        if args.benchmark_batches:
            dataset = PCBADataset(args.processed_dir, args.split_file, "train")
            loader = make_pcba_loader(dataset, args.batch_size, args.workers, True)
            start, graphs = time.perf_counter(), 0
            for i, batch in enumerate(loader):
                graphs += batch.num_graphs
                if i + 1 >= args.benchmark_batches:
                    break
            seconds = time.perf_counter() - start
            result["loader_benchmark"] = {
                "graphs": graphs,
                "seconds": seconds,
                "graphs_per_second": graphs / seconds,
            }
    write_json(args.output, result)
    print(f"Audit passed; saved {args.output}")


if __name__ == "__main__":
    main()
