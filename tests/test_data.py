import numpy as np
import pandas as pd
import pytest
import torch

from dta_project.chemistry import EDGE_DIM, NODE_DIM, canonicalize, scaffold, smiles_to_graph
from dta_project.dta import DTADataset
from dta_project.pcba import PCBADataset, ShardBatchSampler, class_weights, make_pcba_loader
from dta_project.prepare_dta import prepare
from dta_project.splits import audit_dta_split, check_indices, group_partition, make_dta_split


@pytest.mark.parametrize("smiles", ["CCO", "[Na+]", "c1ccccc1", "Cl.Cl", "C[C@H](O)N"])
def test_graph_shapes(smiles):
    g = smiles_to_graph(smiles)
    assert g.x.shape[1] == NODE_DIM
    assert g.edge_index.shape == (2, g.edge_attr.shape[0])
    assert g.edge_attr.shape[1] == EDGE_DIM


def test_zero_bonds_and_canonicalization():
    g = smiles_to_graph("[Na+]")
    assert g.edge_index.shape == (2, 0) and g.edge_attr.shape == (0, EDGE_DIM)
    assert canonicalize("OCC") == canonicalize("CCO")
    assert scaffold("CCC") == scaffold("CCO") == ""


@pytest.mark.parametrize("s", ["", "definitely_not_smiles", None])
def test_invalid_smiles(s):
    with pytest.raises(ValueError):
        smiles_to_graph(s)


def test_pcba_batch_ids_and_masks(smoke_root):
    ds = PCBADataset(smoke_root / "pcba")
    loader = make_pcba_loader(ds, 5)
    ids = []
    for batch in loader:
        assert batch.y.shape == batch.label_mask.shape == (batch.num_graphs, 4)
        ids.extend(batch.molecule_id.tolist())
    assert sorted(ids) == list(range(20))


def test_sampler_exact_coverage_and_epoch_randomness(smoke_root):
    ds = PCBADataset(smoke_root / "pcba")
    sampler = ShardBatchSampler(ds, batch_size=3, shuffle=True, seed=9)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert sorted(i for batch in first for i in batch) == list(range(len(ds)))
    assert sorted(i for batch in second for i in batch) == list(range(len(ds)))
    assert first != second and len(first) == len(sampler)
    for batch in first:
        assert ds.index.iloc[batch].shard_filename.nunique() == 1


def test_class_weights_training_only(smoke_root):
    ds = PCBADataset(smoke_root / "pcba", smoke_root / "pcba/splits/random_seed42.json", "train")
    weights, p, n = class_weights(ds)
    ids = ds.index.global_index.to_numpy()
    y, mask = ds.labels[ids], ds.mask[ids]
    assert np.array_equal(p, ((y == 1) & mask).sum(0))
    assert np.array_equal(n, ((y == 0) & mask).sum(0))
    assert weights.min() >= 1 and weights.max() <= 50


def test_split_duplicates_and_unknown_ids():
    payload = {"train": [0, 0], "validation": [1], "test": [2]}
    with pytest.raises(ValueError, match="Duplicate"):
        check_indices(payload, [0, 1, 2])
    payload["train"] = [3]
    with pytest.raises(ValueError, match="Unknown"):
        check_indices(payload, [0, 1, 2])


@pytest.mark.parametrize("mode", ["random", "cold_drug", "scaffold", "cold_target", "cold_both"])
def test_group_splits(smoke_root, mode):
    frame = pd.read_parquet(smoke_root / "dta/pairs.parquet")
    split = make_dta_split(frame, mode, seed=17)
    report = audit_dta_split(frame, split)
    assert report["mode"] == mode
    assert split == make_dta_split(frame, mode, seed=17)


def test_detect_entity_leakage(smoke_root):
    frame = pd.read_parquet(smoke_root / "dta/pairs.parquet")
    split = make_dta_split(frame, "random", seed=42)
    split["metadata"]["mode"] = "cold_drug"
    with pytest.raises(ValueError, match="leakage"):
        audit_dta_split(frame, split)


def test_too_few_groups():
    with pytest.raises(ValueError, match="three distinct"):
        group_partition(["a", "a", "b"], 42)


def test_synthetic_cache_guard(smoke_root):
    with pytest.raises(ValueError, match="Synthetic"):
        DTADataset(
            smoke_root / "dta",
            smoke_root / "dta/splits/cold_drug_seed42.json",
            "train",
            smoke_root / "synthetic_proteins.pt",
        )


def test_stale_protein_cache(smoke_root, tmp_path):
    payload = torch.load(smoke_root / "synthetic_proteins.pt", weights_only=True)
    key = next(iter(payload["sequence_sha256"]))
    payload["sequence_sha256"][key] = "stale"
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Stale"):
        DTADataset(
            smoke_root / "dta",
            smoke_root / "dta/splits/cold_drug_seed42.json",
            "train",
            path,
            allow_synthetic=True,
        )


def test_pair_dedup_and_pK_transform(tmp_path):
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN", "CCC", "CCCl", "CCF"],
            "sequence": ["ACDE"] * 6,
            "affinity": [10, 100, 10, 20, 30, 40],
        }
    )
    out, _ = prepare(
        frame, tmp_path / "data", dataset="custom", transform="neg_log10_nM", mode="cold_drug"
    )
    pairs = pd.read_parquet(out / "pairs.parquet")
    ethanol = pairs[pairs.canonical_smiles == "CCO"].iloc[0]
    assert len(pairs) == 5 and ethanol.replicate_count == 2
    assert ethanol.label == pytest.approx(7.5)


def test_observed_nan_not_silently_dropped(smoke_root, tmp_path):
    # Copy only the small fixture; source files are never modified by tests.
    import shutil

    shutil.copytree(smoke_root / "pcba", tmp_path / "pcba")
    y = np.load(tmp_path / "pcba/labels.npy")
    mask = np.load(tmp_path / "pcba/label_mask.npy")
    i, j = np.argwhere(mask)[0]
    y[i, j] = np.nan
    np.save(tmp_path / "pcba/labels.npy", y)
    ds = PCBADataset(tmp_path / "pcba")
    with pytest.raises(ValueError, match="OBSERVED"):
        ds[int(i)]
