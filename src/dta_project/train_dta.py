"""Train scratch, frozen-PCBA, or staged-fine-tuned drug-target affinity models."""

from __future__ import annotations

import copy
from pathlib import Path

from .common import choose_device, load_config, parser_for_training, resolve_paths, seed_all, sha256
from .dta import DTADataset, make_dta_loader
from .engine import fit
from .models import DTARegressor, transfer_encoder


def train(config, resume=None):
    config = resolve_paths(config)
    dc, tc, transfer = config["data"], config["training"], config["transfer"]
    seed_all(config["seed"])
    device = choose_device(tc["device"])
    train_data = DTADataset(
        dc["processed_dir"], dc["split_file"], "train", dc["protein_cache"], dc["allow_synthetic"]
    )
    val_data = DTADataset(
        dc["processed_dir"],
        dc["split_file"],
        "validation",
        dc["protein_cache"],
        dc["allow_synthetic"],
    )
    model_config = copy.deepcopy(config["model"])
    graph = train_data[0][0]
    model_config["drug"].update(node_dim=graph.x.shape[-1], edge_dim=graph.edge_attr.shape[-1])
    model_config["protein_dim"] = train_data.protein_dim
    if (
        model_config["fusion"] == "cross_attention"
        and train_data.cache_metadata.get("pooling") != "residues"
    ):
        raise ValueError(
            "Atom-residue attention requires an ESM cache created with --pooling residues"
        )
    model = DTARegressor(model_config).to(device)
    if transfer["mode"] != "scratch" and not resume:
        if not transfer["checkpoint"]:
            raise ValueError("Set transfer.checkpoint for pretrained variants")
        transfer_encoder(model, transfer["checkpoint"], train_data.metadata["schema_hash"])
    if transfer["mode"] == "scratch" and transfer["checkpoint"]:
        raise ValueError("Scratch baseline must not specify a pretrained checkpoint")
    values = train_data.frame.label.to_numpy(dtype=float)
    stats = {"mean": float(values.mean()), "std": float(max(values.std(ddof=0), 1e-8))}
    if not tc["normalize_labels"]:
        stats = {"mean": 0.0, "std": 1.0, "train_mean": float(values.mean())}
    signature = {
        "pairs_sha256": sha256(Path(dc["processed_dir"]) / "pairs.parquet"),
        "graphs_sha256": sha256(Path(dc["processed_dir"]) / "drug_graphs.pt"),
        "metadata_sha256": sha256(Path(dc["processed_dir"]) / "metadata.json"),
        "split_sha256": sha256(dc["split_file"]),
    }
    loader = lambda epoch: make_dta_loader(
        train_data, tc["batch_size"], tc["workers"], True, config["seed"], epoch
    )
    val_loader = make_dta_loader(val_data, tc["batch_size"], tc["workers"])
    return fit(
        model,
        "dta",
        config,
        model_config,
        device,
        loader,
        val_loader,
        signature,
        train_data.metadata["schema_hash"],
        label_stats=stats,
        resume=resume,
        protein_signature=sha256(dc["protein_cache"]),
    )


def main():
    args = parser_for_training(__doc__).parse_args()
    train(load_config(args.config, args.set), args.resume)


if __name__ == "__main__":
    main()
