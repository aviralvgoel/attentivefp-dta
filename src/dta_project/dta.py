"""Drug-target dataset with explicit pair IDs and sequence-verified protein cache."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from .chemistry import SCHEMA_HASH
from .common import sha256, worker_seed
from .pcba import validate_graph
from .splits import audit_dta_split


class DTADataset(Dataset):
    def __init__(self, directory, split_file, split, protein_cache, allow_synthetic=False):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata.get("schema_hash") != SCHEMA_HASH:
            raise ValueError(
                "DTA graph feature schema mismatch; reprocess with the shared featurizer"
            )
        self.all_pairs = pd.read_parquet(self.directory / "pairs.parquet")
        if self.all_pairs.pair_id.duplicated().any():
            raise ValueError("Duplicate pair_id")
        split_payload = json.loads(Path(split_file).read_text())
        audit_dta_split(self.all_pairs, split_payload)
        expected = split_payload.get("metadata", {}).get("pairs_sha256")
        if expected and expected != sha256(self.directory / "pairs.parquet"):
            raise ValueError("DTA table changed after splitting")
        self.frame = self.all_pairs[self.all_pairs.pair_id.isin(split_payload[split])].reset_index(
            drop=True
        )
        # Local trusted graphs are PyG objects; do not open arbitrary untrusted .pt files.
        self.graphs = torch.load(
            self.directory / "drug_graphs.pt", map_location="cpu", weights_only=False
        )
        for graph in self.graphs.values():
            validate_graph(graph)
        cache = torch.load(protein_cache, map_location="cpu", weights_only=True)
        self.cache_metadata = cache["metadata"]
        if self.cache_metadata.get("synthetic", False) and not allow_synthetic:
            raise ValueError(
                "Synthetic protein features may only be used with data.allow_synthetic=true"
            )
        self.features = cache["embeddings"]
        manifest = cache["sequence_sha256"]
        for row in (
            self.all_pairs[["protein_id", "protein_sequence"]].drop_duplicates().itertuples()
        ):
            expected = __import__("hashlib").sha256(row.protein_sequence.encode()).hexdigest()
            if manifest.get(row.protein_id) != expected or row.protein_id not in self.features:
                raise ValueError(f"Stale or missing protein representation: {row.protein_id}")
        dims = {v.shape[-1] for v in self.features.values()}
        if len(dims) != 1:
            raise ValueError("Inconsistent protein feature dimensions")
        self.protein_dim = dims.pop()
        for value in self.features.values():
            if value.ndim not in (1, 2) or not value.numel() or not torch.isfinite(value).all():
                raise ValueError("Invalid protein embedding")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        row = self.frame.iloc[item]
        return (
            self.graphs[row.drug_id],
            self.features[row.protein_id],
            float(row.label),
            int(row.pair_id),
        )


def collate_pairs(samples):
    graphs, protein, labels, ids = zip(*samples)
    ranks = {p.ndim for p in protein}
    if len(ranks) != 1:
        raise ValueError("Cannot mix pooled and residue-level cache entries")
    if protein[0].ndim == 1:
        features = torch.stack(protein).unsqueeze(1).float()
        mask = torch.ones((len(samples), 1), dtype=torch.bool)
    else:
        length = max(len(p) for p in protein)
        features = torch.zeros((len(samples), length, protein[0].shape[-1]))
        mask = torch.zeros((len(samples), length), dtype=torch.bool)
        for i, value in enumerate(protein):
            features[i, : len(value)] = value.float()
            mask[i, : len(value)] = True
    return {
        "graph": Batch.from_data_list(list(graphs)),
        "protein": features,
        "protein_mask": mask,
        "y": torch.tensor(labels, dtype=torch.float32),
        "pair_id": torch.tensor(ids, dtype=torch.long),
    }


def make_dta_loader(dataset, batch_size=64, workers=0, shuffle=False, seed=42, epoch=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        collate_fn=collate_pairs,
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed + epoch),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
