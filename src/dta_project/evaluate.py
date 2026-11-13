"""Explicit held-out evaluation and stable-ID prediction export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import choose_device, load_checkpoint, sha256, write_json
from .dta import DTADataset, make_dta_loader
from .engine import evaluate_dta, evaluate_pcba
from .models import DTARegressor, PCBAClassifier
from .pcba import PCBADataset, make_pcba_loader


def evaluate(
    checkpoint_path,
    split="test",
    device="auto",
    batch_size=256,
    workers=0,
    output=None,
    processed_dir=None,
    split_file=None,
    protein_cache=None,
):
    checkpoint = load_checkpoint(checkpoint_path)
    dc = checkpoint["config"]["data"].copy()
    if processed_dir:
        dc["processed_dir"] = str(processed_dir)
    if split_file:
        dc["split_file"] = str(split_file)
    if protein_cache:
        dc["protein_cache"] = str(protein_cache)
    signature = checkpoint["data_signature"]
    if sha256(dc["split_file"]) != signature["split_sha256"]:
        raise ValueError("Evaluation split differs from checkpoint's locked split")
    root = Path(dc["processed_dir"])
    if checkpoint["task"] == "dta":
        for key, filename in (
            ("pairs_sha256", "pairs.parquet"),
            ("graphs_sha256", "drug_graphs.pt"),
            ("metadata_sha256", "metadata.json"),
        ):
            if sha256(root / filename) != signature[key]:
                raise ValueError(f"Evaluation data changed: {filename}")
        if sha256(dc["protein_cache"]) != checkpoint["protein_signature"]:
            raise ValueError("Protein cache differs from the training cache")
        dataset = DTADataset(
            root, dc["split_file"], split, dc["protein_cache"], dc["allow_synthetic"]
        )
        loader = make_dta_loader(dataset, batch_size, workers)
        model = DTARegressor(checkpoint["model_config"])
    else:
        for key, filename in (
            ("index_sha256", "index.parquet"),
            ("labels_sha256", "labels.npy"),
            ("mask_sha256", "label_mask.npy"),
            ("metadata_sha256", "metadata.json"),
        ):
            if signature.get(key) and sha256(root / filename) != signature[key]:
                raise ValueError(f"Evaluation PCBA data changed: {filename}")
        dataset = PCBADataset(root, dc["split_file"], split)
        for name, digest in signature.get("shard_hashes", {}).items():
            if sha256(root / "graphs" / name) != digest:
                raise ValueError(f"PCBA shard changed since training: {name}")
        loader = make_pcba_loader(dataset, batch_size, workers)
        model = PCBAClassifier(checkpoint["model_config"])
    device = choose_device(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    if checkpoint["task"] == "dta":
        metrics, predictions = evaluate_dta(model, loader, device, checkpoint["label_stats"])
    else:
        weights = checkpoint.get("pos_weight")
        metrics, predictions = evaluate_pcba(
            model, loader, device, weights.to(device) if weights is not None else None
        )
    out = Path(output) if output else Path(checkpoint_path).parent / f"evaluation_{split}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Evaluation output exists: {out}. Choose --output for a new evaluation."
        )
    out.mkdir(parents=True, exist_ok=True)
    if checkpoint["task"] == "dta":
        pred = pd.DataFrame(predictions)
        entities = dataset.frame[
            ["pair_id", "drug_id", "protein_id", "canonical_smiles", "scaffold"]
        ]
        pred = pred.merge(entities, on="pair_id", validate="one_to_one")
        pred.to_parquet(out / "predictions.parquet", index=False)
    else:
        np.savez_compressed(out / "predictions.npz", **predictions)
    write_json(
        out / "metrics.json",
        {
            "split": split,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"],
            **metrics,
        },
    )
    print(
        f"{split} metrics: {metrics if checkpoint['task'] == 'dta' else {k: v for k, v in metrics.items() if k != 'per_task'}}"
    )
    print(f"Saved predictions and metrics to {out}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--protein-cache", type=Path)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.split,
        args.device,
        args.batch_size,
        args.workers,
        args.output,
        args.processed_dir,
        args.split_file,
        args.protein_cache,
    )


if __name__ == "__main__":
    main()
