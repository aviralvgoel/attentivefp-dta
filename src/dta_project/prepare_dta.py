"""Curate a single homogeneous affinity endpoint; build graphs and grouped splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chemistry import SCHEMA, SCHEMA_HASH, canonicalize, scaffold, smiles_to_graph
from .common import sha256, write_json
from .splits import audit_dta_split, make_dta_split

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def stable_id(prefix, value):
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:24]


def normalize_sequence(value):
    if not isinstance(value, str):
        raise TypeError("Protein sequence must be a string")
    value = "".join(value.split()).upper()
    if not value or set(value) - AMINO_ACIDS:
        raise ValueError("Invalid amino acid sequence; gaps/stops are not silently removed")
    return value


def read_deepdta(directory, trust_pickle=False):
    if not trust_pickle:
        raise ValueError(
            "Y is a Python pickle. Only for a trusted source, pass --trust-local-pickle."
        )
    root = Path(directory)
    ligands = json.loads((root / "ligands_can.txt").read_text())
    proteins = json.loads((root / "proteins.txt").read_text())
    with (root / "Y").open("rb") as stream:
        y = np.asarray(pickle.load(stream, encoding="latin1"), dtype=float)
    if y.shape != (len(ligands), len(proteins)):
        raise ValueError("DeepDTA Y matrix dimensions disagree with JSON entities")
    d, p = np.where(np.isfinite(y))
    ds, ps = list(ligands.values()), list(proteins.values())
    return pd.DataFrame(
        {"smiles": [ds[i] for i in d], "sequence": [ps[i] for i in p], "affinity": y[d, p]}
    )


def prepare(
    frame,
    output,
    *,
    dataset,
    transform="none",
    mode="cold_drug",
    seed=42,
    invalid="error",
    source=None,
):
    if not {"smiles", "sequence", "affinity"} <= set(frame):
        raise ValueError("Input must have smiles, sequence, affinity columns")
    if dataset.lower() not in ("kiba", "davis", "bindingdb", "custom", "synthetic"):
        raise ValueError("dataset must identify the data source")
    # Do not silently merge IDs whose supplied protein identifier maps to multiple sequences.
    if "target_id" in frame and frame.groupby("target_id").sequence.nunique().max() > 1:
        raise ValueError("A target_id maps to conflicting protein sequences")
    records, rejected = [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        try:
            smi, seq, label = (
                canonicalize(row.smiles),
                normalize_sequence(row.sequence),
                float(row.affinity),
            )
            if not np.isfinite(label):
                raise ValueError("Nonfinite affinity")
            if transform == "neg_log10_nM":
                if label <= 0:
                    raise ValueError("nM affinity must be positive")
                label = 9.0 - np.log10(label)
            elif transform != "none":
                raise ValueError(f"Unsupported transformation {transform}")
            records.append(
                {
                    "canonical_smiles": smi,
                    "protein_sequence": seq,
                    "label": float(label),
                    "raw_label": float(row.affinity),
                }
            )
        except (ValueError, TypeError) as exc:
            if invalid == "error":
                raise ValueError(f"Invalid input row {i}: {exc}") from exc
            rejected.append({"input_row": i, "reason": str(exc)})
    if not records:
        raise ValueError("No valid pairs")
    clean = pd.DataFrame(records)
    # Merge exact pair duplicates BEFORE splitting; split on canonical chemical identity.
    clean = clean.groupby(["canonical_smiles", "protein_sequence"], sort=True, as_index=False).agg(
        label=("label", "median"),
        raw_label=("raw_label", "median"),
        replicate_count=("label", "size"),
    )
    clean["drug_id"] = clean.canonical_smiles.map(lambda s: stable_id("d_", s))
    clean["protein_id"] = clean.protein_sequence.map(lambda s: stable_id("p_", s))
    clean["scaffold"] = clean.canonical_smiles.map(scaffold)
    clean.insert(0, "pair_id", np.arange(len(clean), dtype=np.int64))
    split = make_dta_split(clean, mode, seed)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output is not empty: {out}; choose a new dataset directory")
    out.mkdir(parents=True, exist_ok=True)
    drugs = (
        clean[["drug_id", "canonical_smiles", "scaffold"]].drop_duplicates().reset_index(drop=True)
    )
    proteins = clean[["protein_id", "protein_sequence"]].drop_duplicates().reset_index(drop=True)
    graphs = {row.drug_id: smiles_to_graph(row.canonical_smiles) for row in drugs.itertuples()}
    torch.save(graphs, out / "drug_graphs.pt")
    clean.to_parquet(out / "pairs.parquet", index=False)
    drugs.to_parquet(out / "drugs.parquet", index=False)
    proteins.to_parquet(out / "proteins.parquet", index=False)
    split["metadata"]["pairs_sha256"] = sha256(out / "pairs.parquet")
    split_path = out / "splits" / f"{mode}_seed{seed}.json"
    write_json(split_path, split)
    write_json(out / "audit.json", audit_dta_split(clean, split))
    write_json(out / "rejected_rows.json", rejected)
    metadata = {
        "format": "dta_v1",
        "dataset": dataset,
        "label_transform": transform,
        "label_units": "pK" if transform == "neg_log10_nM" else "as_supplied",
        "schema_hash": SCHEMA_HASH,
        "schema": SCHEMA,
        "pairs": len(clean),
        "drugs": len(drugs),
        "proteins": len(proteins),
        "source": source,
        "input_rows": len(frame),
        "rejected_rows": len(rejected),
        "graphs_sha256": sha256(out / "drug_graphs.pt"),
    }
    write_json(out / "metadata.json", metadata)
    return out, split_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="CSV: smiles, sequence, affinity")
    inputs.add_argument(
        "--deepdta-dir", type=Path, help="Trusted DeepDTA-format ligands_can.txt/proteins.txt/Y"
    )
    parser.add_argument("--trust-local-pickle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=["kiba", "davis", "bindingdb", "custom"], required=True
    )
    parser.add_argument("--label-transform", choices=["none", "neg_log10_nM"], required=True)
    parser.add_argument(
        "--split-mode",
        choices=["random", "cold_drug", "scaffold", "cold_target", "cold_both"],
        default="cold_drug",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid", choices=["error", "drop"], default="error")
    args = parser.parse_args()
    frame = (
        pd.read_csv(args.input)
        if args.input
        else read_deepdta(args.deepdta_dir, args.trust_local_pickle)
    )
    source = (
        {"csv_sha256": sha256(args.input)}
        if args.input
        else {
            "deepdta_files": {
                name: sha256(args.deepdta_dir / name)
                for name in ("Y", "ligands_can.txt", "proteins.txt")
            }
        }
    )
    out, split = prepare(
        frame,
        args.output,
        dataset=args.dataset,
        transform=args.label_transform,
        mode=args.split_mode,
        seed=args.seed,
        invalid=args.invalid,
        source=source,
    )
    print(f"Processed dataset: {out}\nSplit: {split}")


if __name__ == "__main__":
    main()
