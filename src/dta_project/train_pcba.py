"""Train or resume PCBA pretraining on existing or newly prepared graph shards."""

from __future__ import annotations

import copy
from pathlib import Path

from .common import choose_device, load_config, parser_for_training, resolve_paths, seed_all, sha256
from .engine import fit
from .models import PCBAClassifier
from .pcba import PCBADataset, class_weights, make_pcba_loader


def train(config, resume=None):
    config = resolve_paths(config)
    dc, tc = config["data"], config["training"]
    seed_all(config["seed"])
    device = choose_device(tc["device"])
    train_data = PCBADataset(dc["processed_dir"], dc["split_file"], "train", dc["cache_shards"])
    val_data = PCBADataset(dc["processed_dir"], dc["split_file"], "validation", dc["cache_shards"])
    graph = train_data[0]
    model_config = copy.deepcopy(config["model"])
    model_config.update(
        node_dim=graph.x.shape[-1], edge_dim=graph.edge_attr.shape[-1], num_tasks=graph.y.shape[-1]
    )
    model = PCBAClassifier(model_config).to(device)
    weights, positive, negative = class_weights(train_data, tc["positive_weight_cap"])
    if not (positive + negative).sum():
        raise ValueError("No observed training labels")
    pos_weight = weights.to(device) if tc["class_weighting"] else None
    signature = {
        "index_sha256": sha256(Path(dc["processed_dir"]) / "index.parquet"),
        "split_sha256": sha256(dc["split_file"]),
        "metadata_sha256": sha256(Path(dc["processed_dir"]) / "metadata.json")
        if (Path(dc["processed_dir"]) / "metadata.json").exists()
        else None,
        "labels_sha256": sha256(Path(dc["processed_dir"]) / "labels.npy")
        if train_data.labels is not None
        else None,
        "mask_sha256": sha256(Path(dc["processed_dir"]) / "label_mask.npy")
        if train_data.labels is not None
        else None,
    }
    signature["shard_hashes"] = {
        name: sha256(Path(dc["processed_dir"]) / "graphs" / name)
        for name in train_data.index_all.shard_filename.unique()
    }
    loader = lambda epoch: make_pcba_loader(
        train_data, tc["batch_size"], tc["workers"], True, config["seed"], epoch
    )
    val_loader = make_pcba_loader(val_data, tc["batch_size"], tc["workers"])
    return fit(
        model,
        "pcba",
        config,
        model_config,
        device,
        loader,
        val_loader,
        signature,
        train_data.metadata.get("schema_hash"),
        pos_weight=pos_weight,
        resume=resume,
    )


def main():
    args = parser_for_training(__doc__).parse_args()
    train(load_config(args.config, args.set), args.resume)


if __name__ == "__main__":
    main()
