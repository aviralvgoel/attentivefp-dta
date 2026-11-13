"""Legacy-compatible PCBA shards, correct graph-label batching, and local I/O sampling."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader

from .common import worker_seed
from .splits import check_indices


def validate_graph(graph):
    if graph.x.ndim != 2 or graph.x.shape[0] < 1 or not torch.isfinite(graph.x).all():
        raise ValueError("Invalid node feature tensor")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E], including for zero-bond graphs")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[0] != graph.edge_index.shape[1]:
        raise ValueError("edge_attr must have shape [E, edge_dim]")
    if not torch.isfinite(graph.edge_attr).all():
        raise ValueError("Non-finite bond features")
    if graph.edge_index.numel() and (
        graph.edge_index.min() < 0 or graph.edge_index.max() >= len(graph.x)
    ):
        raise ValueError("Bond references an out-of-range atom")


class PCBADataset(Dataset):
    def __init__(self, directory, split_file=None, split="train", cache_shards=2):
        self.directory = Path(directory).resolve()
        self.graphs_dir = self.directory / "graphs"
        self.index_all = pd.read_parquet(self.directory / "index.parquet")
        required = {"global_index", "shard_filename", "offset"}
        if not required <= set(self.index_all):
            raise ValueError(f"Missing index columns: {required - set(self.index_all)}")
        if self.index_all.global_index.duplicated().any():
            raise ValueError("Duplicate global_index in PCBA index")
        for col in ("global_index", "offset"):
            values = pd.to_numeric(self.index_all[col], errors="raise")
            if (values < 0).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"Invalid {col}")
            self.index_all[col] = values.astype(np.int64)
        self.index = self.index_all.copy()
        if split_file:
            payload = json.loads(Path(split_file).read_text())
            parts = check_indices(payload, self.index_all.global_index)
            self.index = self.index[self.index.global_index.isin(parts[split])]
        self.index = self.index.sort_values("global_index").reset_index(drop=True)
        if self.index.empty:
            raise ValueError("Empty PCBA split")
        self.labels = self.mask = None
        lp, mp = self.directory / "labels.npy", self.directory / "label_mask.npy"
        if lp.exists() != mp.exists():
            raise ValueError("Both labels.npy and label_mask.npy must exist together")
        if lp.exists():
            self.labels = np.load(lp, mmap_mode="r", allow_pickle=False)
            self.mask = np.load(mp, mmap_mode="r", allow_pickle=False)
            if self.labels.ndim != 2 or self.labels.shape != self.mask.shape:
                raise ValueError("Label arrays must both be [N, tasks]")
            if self.index_all.global_index.max() >= len(self.labels):
                raise ValueError("Labels do not cover all global IDs")
            if self.mask.dtype != np.bool_:
                raise ValueError("label_mask.npy must be boolean")
        self.metadata = (
            json.loads((self.directory / "metadata.json").read_text())
            if (self.directory / "metadata.json").exists()
            else {}
        )
        self.cache_shards = int(cache_shards)
        if self.cache_shards < 1:
            raise ValueError("cache_shards must be positive")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _read(self, filename):
        if filename not in self._cache:
            path = (self.graphs_dir / filename).resolve()
            if self.graphs_dir not in path.parents:
                raise ValueError("Shard path escapes graphs/ directory")
            # Only use graph shards from your OWN trusted preprocessing.
            self._cache[filename] = torch.load(path, map_location="cpu", weights_only=False)
            if len(self._cache) > self.cache_shards:
                self._cache.popitem(last=False)
        self._cache.move_to_end(filename)
        return self._cache[filename]

    def __getitem__(self, item):
        row = self.index.iloc[item]
        source = self._read(str(row.shard_filename))
        if isinstance(source, dict) and "graphs" in source:
            source = source["graphs"]
        if isinstance(source, list):
            original = source[int(row.offset)]
        elif isinstance(source, tuple) and len(source) == 2:
            data, slices = source
            original = separate(data.__class__, data, int(row.offset), slices, decrement=False)
        else:
            raise TypeError(
                "Unsupported shard; expected list[Data], {'graphs': list}, or (data, slices)"
            )
        # Reconstruct standard Data: avoid stale labels and PyG's default increment
        # of ANY key containing 'index'. Stable external ID is named molecule_id.
        graph = Data(
            x=original.x.float(),
            edge_index=original.edge_index.long(),
            edge_attr=original.edge_attr.float(),
        )
        validate_graph(graph)
        gid = int(row.global_index)
        if self.labels is not None:
            mask = np.array(self.mask[gid], dtype=bool, copy=True)
            y = np.array(self.labels[gid], dtype=np.float32, copy=True)
            if not np.isfinite(y[mask]).all() or not np.isin(y[mask], [0, 1]).all():
                raise ValueError(f"Nonbinary or nonfinite OBSERVED label for molecule {gid}")
            y[~mask] = 0
            graph.y = torch.from_numpy(y).view(1, -1)
            graph.label_mask = torch.from_numpy(mask).view(1, -1)
        else:
            if (
                getattr(original, "y", None) is None
                or getattr(original, "label_mask", None) is None
            ):
                raise ValueError("No labels or mask in arrays or graph shard")
            graph.y = original.y.float().view(1, -1).clone()
            graph.label_mask = original.label_mask.bool().view(1, -1).clone()
            observed = graph.y[graph.label_mask]
            if not torch.isfinite(observed).all() or not ((observed == 0) | (observed == 1)).all():
                raise ValueError("Invalid observed labels")
            graph.y[~graph.label_mask] = 0
        if graph.y.shape != graph.label_mask.shape:
            raise ValueError("Label/mask shape mismatch")
        graph.molecule_id = torch.tensor([gid], dtype=torch.long)
        return graph


class ShardBatchSampler(Sampler):
    """All rows exactly once; randomize shards and rows within each shard per epoch."""

    def __init__(self, dataset, batch_size=512, shuffle=True, seed=42):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.groups = [
            list(ids) for ids in dataset.index.groupby("shard_filename", sort=True).indices.values()
        ]
        self.batch_size, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.groups)) if self.shuffle else range(len(self.groups))
        for group in order:
            rows = np.asarray(self.groups[group]).copy()
            if self.shuffle:
                rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                yield rows[start : start + self.batch_size].tolist()


def make_pcba_loader(dataset, batch_size, workers=0, shuffle=False, seed=42, epoch=0):
    sampler = ShardBatchSampler(dataset, batch_size, shuffle, seed)
    sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed + epoch),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def class_weights(dataset, cap=50.0):
    if cap < 1:
        raise ValueError("Positive-class weight cap must be at least 1")
    if dataset.labels is not None:
        positive = np.zeros(dataset.labels.shape[1], dtype=np.float64)
        negative = np.zeros_like(positive)
        ids = dataset.index.global_index.to_numpy()
        for start in range(0, len(ids), 8192):
            rows = ids[start : start + 8192]
            y, m = dataset.labels[rows], dataset.mask[rows]
            if not np.isfinite(y[m]).all() or not np.isin(y[m], [0, 1]).all():
                raise ValueError("Invalid observed training labels")
            positive += ((y == 1) & m).sum(0)
            negative += ((y == 0) & m).sum(0)
    else:
        positive = negative = None
        for batch in make_pcba_loader(dataset, 256):
            p = ((batch.y == 1) & batch.label_mask).sum(0).numpy()
            n = ((batch.y == 0) & batch.label_mask).sum(0).numpy()
            positive = p.astype(float) if positive is None else positive + p
            negative = n.astype(float) if negative is None else negative + n
    weights = np.where(positive > 0, negative / np.maximum(positive, 1), 1)
    return torch.tensor(np.clip(weights, 1, cap), dtype=torch.float32), positive, negative
