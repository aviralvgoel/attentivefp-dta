import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dta_project.audit import audit_pcba
from dta_project.common import RunDirectory, load_checkpoint, load_config, write_json
from dta_project.evaluate import evaluate
from dta_project.train_dta import train
from dta_project.train_pcba import train as train_pcba


def test_training_artifacts_and_no_implicit_test(smoke_root):
    for mode in ("pretrain", "scratch", "frozen", "finetune", "attention"):
        root = smoke_root / mode
        for name in (
            "best.pt",
            "last.pt",
            "config.json",
            "environment.json",
            "metrics.json",
            "train.log",
            "history.jsonl",
        ):
            assert (root / name).exists()
        assert json.loads((root / "metrics.json").read_text())["test_evaluated"] is False
        assert not (root / "evaluation_test").exists()


def test_paused_resume_matches_uninterrupted(smoke_root, tmp_path):
    checkpoint = load_checkpoint(smoke_root / "scratch/last.pt")
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["training"]["output_dir"] = str(tmp_path / "uninterrupted")
    train(cfg)
    uninterrupted = load_checkpoint(tmp_path / "uninterrupted/last.pt")
    for key, value in checkpoint["model_state"].items():
        assert torch.equal(value, uninterrupted["model_state"][key]), key


def test_frozen_weights_unchanged(smoke_root):
    before = load_checkpoint(smoke_root / "pretrain/best.pt")["model_state"]
    after = load_checkpoint(smoke_root / "frozen/last.pt")["model_state"]
    for key, value in before.items():
        if key.startswith("encoder."):
            assert torch.equal(value, after[key])


def test_finetune_unfreezes_after_warmup(smoke_root):
    rows = [
        json.loads(line)
        for line in (smoke_root / "finetune/history.jsonl").read_text().splitlines()
    ]
    assert [r["encoder_frozen"] for r in rows] == [True, False, False]


def test_prediction_mapping_is_one_to_one(smoke_root):
    predictions = pd.read_parquet(smoke_root / "heldout_test/predictions.parquet")
    split = json.loads((smoke_root / "dta/splits/cold_drug_seed42.json").read_text())
    assert not predictions.pair_id.duplicated().any()
    assert sorted(predictions.pair_id) == sorted(split["test"])


def test_no_accidental_run_overwrite(tmp_path):
    with RunDirectory(tmp_path / "run") as root:
        (root / "existing.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="not empty"):
        RunDirectory(tmp_path / "run")
    assert (tmp_path / "run/existing.txt").read_text() == "preserve"


def test_config_overrides_and_typo_guard():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/kiba_finetune.yaml", ["seed=7", "training.workers=2"])
    assert cfg["transfer"]["mode"] == "finetune" and cfg["seed"] == 7
    with pytest.raises(KeyError):
        load_config(root / "configs/kiba_finetune.yaml", ["training.workres=2"])


def test_nonstandard_json_nan_is_not_emitted(tmp_path):
    path = tmp_path / "metrics.json"
    write_json(path, {"bad": float("nan"), "inf": float("inf"), "array": np.array([1])})
    assert json.loads(path.read_text()) == {"bad": None, "inf": None, "array": [1]}


def test_pcba_full_feature_audit(smoke_root):
    report = audit_pcba(
        smoke_root / "pcba",
        smoke_root / "pcba/splits/scaffold_seed42.json",
        check_features=True,
        all_graphs=True,
    )
    assert report["all_graph_features_verified"] and report["verified_features"] == 20


def test_pcba_evaluation_exports_unmodified_ids(smoke_root, tmp_path):
    evaluate(smoke_root / "pretrain/best.pt", "test", "cpu", 3, output=tmp_path / "eval")
    pred = np.load(tmp_path / "eval/predictions.npz")
    split = json.loads((smoke_root / "pcba/splits/random_seed42.json").read_text())
    assert sorted(pred["molecule_id"]) == sorted(split["test"])


def test_resume_rejects_scientific_config_change(smoke_root):
    cfg = load_checkpoint(smoke_root / "scratch/last.pt")["config"]
    cfg["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Scientific configuration changed"):
        train(cfg, smoke_root / "scratch/last.pt")


def test_pcba_resume_matches_uninterrupted(smoke_root, tmp_path):
    reference = load_checkpoint(smoke_root / "pretrain/last.pt")
    cfg = copy.deepcopy(reference["config"])
    cfg["training"]["output_dir"] = str(tmp_path / "pcba_resume")
    cfg["training"]["stop_after_epoch"] = 1
    best = train_pcba(cfg)
    cfg["training"]["stop_after_epoch"] = None
    train_pcba(cfg, best.parent / "last.pt")
    resumed = load_checkpoint(best.parent / "last.pt")
    for key in reference["model_state"]:
        assert torch.equal(reference["model_state"][key], resumed["model_state"][key])


def test_pcba_smoke_bce_fallback_is_explicit(smoke_root, tmp_path):
    import shutil

    shutil.copytree(smoke_root / "pcba", tmp_path / "pcba")
    values = np.load(tmp_path / "pcba/labels.npy")
    mask = np.load(tmp_path / "pcba/label_mask.npy")
    split_path = tmp_path / "pcba/splits/random_seed42.json"
    split = json.loads(split_path.read_text())
    values[split["validation"]] = 0
    mask[split["validation"]] = True
    np.save(tmp_path / "pcba/labels.npy", values)
    np.save(tmp_path / "pcba/label_mask.npy", mask)
    cfg = copy.deepcopy(load_checkpoint(smoke_root / "pretrain/last.pt")["config"])
    cfg["data"].update(processed_dir=str(tmp_path / "pcba"), split_file=str(split_path))
    cfg["training"].update(output_dir=str(tmp_path / "fallback"), smoke_allow_bce_fallback=True)
    path = train_pcba(cfg)
    assert load_checkpoint(path)["selection_metric"] == "negative_masked_bce"
    cfg["training"].update(output_dir=str(tmp_path / "no_fallback"), smoke_allow_bce_fallback=False)
    with pytest.raises(ValueError, match="BCE fallback"):
        train_pcba(cfg)
