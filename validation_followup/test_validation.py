"""CPU tests use explicitly synthetic fixtures; no scientific scores are claimed."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis import (VARIANTS, audit_frame, digest, load_pcba, load_results,
                      paired_bootstrap, read_json, review, write_json)
from run import run_scaffold, scaffold_config, training_action

SEEDS = (42,43,44)


def fixture(root):
    (root / "train_dta.py").write_text("# synthetic fixture, never executes training\n")
    (root / "src/dta_project").mkdir(parents=True)
    (root / "src/dta_project/splits.py").write_text("# fixture sentinel\n")
    pcba = root / "data/pcba/processed"
    (pcba / "splits").mkdir(parents=True)
    (root / "runs/pcba/scaffold_seed42").mkdir(parents=True)
    (root / "runs/pcba/scaffold_seed42/best.pt").write_bytes(b"synthetic-PCBA")
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCN"]
    cores = ["A", "A", "B", "C", "D", "E"]
    pd.DataFrame({"global_index": range(6), "canonical_smiles": smiles,
                  "scaffold": cores}).to_parquet(pcba / "index.parquet", index=False)
    write_json(pcba / "metadata.json", {"schema_hash": "synthetic-schema"})
    write_json(pcba / "splits/scaffold_seed42.json", {"train": [0,1,4], "validation": [2], "test": [3,5]})
    for dataset in ("kiba", "davis"):
        directory = root / f"data/{dataset}/processed"
        (directory / "splits").mkdir(parents=True)
        frame = pd.DataFrame([{"pair_id": d*4+p, "drug_id": f"d{d}", "protein_id": f"p{p}",
                               "canonical_smiles": smiles[d], "scaffold": cores[d],
                               "label": (d+p)/8 + (6 if dataset == "davis" else 0)}
                              for d in range(6) for p in range(4)])
        frame.to_parquet(directory / "pairs.parquet", index=False)
        (directory / "drug_graphs.pt").write_bytes(b"synthetic-graphs")
        (directory.parent / "esm2_mean.pt").write_bytes(b"synthetic-protein-fixture")
        write_json(directory / "metadata.json", {"schema_hash": "synthetic-schema",
                   "label_transform": "neg_log10_nM" if dataset == "davis" else "none"})
        split = {"train": list(range(12)), "validation": list(range(12,16)), "test": list(range(16,24)),
                 "metadata": {"mode": "cold_drug", "seed": 42}}
        split["metadata"]["pairs_sha256"] = digest(directory / "pairs.parquet")
        write_json(directory / "splits/cold_drug_seed42.json", split)
        for seed in SEEDS:
            for variant in VARIANTS:
                config = {"root": str(root), "seed": seed,
                          "model": {"fusion": "concat", "hidden": 3},
                          "data": {"processed_dir": str(directory),
                                   "protein_cache": str(directory.parent / "esm2_mean.pt"),
                                   "split_file": str(directory / "splits/cold_drug_seed42.json"),
                                   "allow_synthetic": False},
                          "transfer": {"mode": variant, "checkpoint": str(root / "runs/pcba/scaffold_seed42/best.pt") if variant != "scratch" else None,
                                       "warmup_epochs": 5 if variant == "finetune" else 0},
                          "training": {"output_dir": str(root / f"runs/{dataset}/cold_drug/{variant}/seed{seed}"),
                                       "epochs": 100, "patience": 12, "normalize_labels": True,
                                       "stop_after_epoch": None, "learning_rate": .001,
                                       "encoder_learning_rate": .00002 if variant == "finetune" else .0002}}
                make_run(config)


def make_run(config, evaluate=True):
    run = Path(config["training"]["output_dir"])
    run.mkdir(parents=True, exist_ok=True)
    directory = Path(config["data"]["processed_dir"])
    write_json(run / "config.json", config)
    signature = {key: digest(path) for key, path in (
        ("pairs_sha256", directory / "pairs.parquet"), ("graphs_sha256", directory / "drug_graphs.pt"),
        ("metadata_sha256", directory / "metadata.json"), ("split_sha256", Path(config["data"]["split_file"])))}
    write_json(run / "dataset_signature.json", signature)
    (run / "best.pt").write_bytes(f"synthetic:{run}".encode())
    (run / "last.pt").write_bytes(b"synthetic-last")
    (run / "train.log").write_text("Finished; best epoch=2. Test set NOT evaluated.\n")
    write_json(run / "metrics.json", {"best_epoch": 2})
    if evaluate:
        make_evaluation(run)


def make_evaluation(run):
    config = read_json(run / "config.json")
    directory = Path(config["data"]["processed_dir"])
    frame = pd.read_parquet(directory / "pairs.parquet")
    test_ids = read_json(config["data"]["split_file"])["test"]
    pred = frame[frame.pair_id.isin(test_ids)].drop(columns="label").copy()
    pred["y_true"] = frame[frame.pair_id.isin(test_ids)].label.to_numpy(np.float32)
    offset = {"scratch": .8, "frozen": .9, "finetune": .5}[config["transfer"]["mode"]] + (config["seed"]-42)*.01
    pred["y_pred"] = (pred.y_true + offset).astype(np.float32)
    out = run / "evaluation_test"
    out.mkdir(exist_ok=True)
    pred.sample(frac=1, random_state=config["seed"]).to_parquet(out / "predictions.parquet", index=False)
    error = pred.y_true.to_numpy(float) - pred.y_pred.to_numpy(float)
    write_json(out / "metrics.json", {"split": "test", "checkpoint_sha256": digest(run / "best.pt"),
               "checkpoint_epoch": 2, "mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error))),
               "rmse": float(np.sqrt(np.mean(error**2))), "ci": 1., "pearson": 1., "spearman": 1.,
               "pairs": len(pred), "mean_predictor_mse": 1.2})


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dta-validation-test-")
        self.root = Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, dataset="kiba"):
        return load_results(self.root, dataset, "cold_drug", SEEDS, load_pcba(self.root))

    def test_aligns_shuffled_rows_and_counts_exposure(self):
        data = self.load()
        self.assertEqual(data["predictions"].shape, (3,3,8))
        self.assertEqual(data["exposure"]["test"]["pcba"]["train"]["drugs"], 1)

    def test_rejects_changed_labels_and_missing_pairs(self):
        path = self.root / "runs/kiba/cold_drug/finetune/seed43/evaluation_test/predictions.parquet"
        pred = pd.read_parquet(path)
        pred.loc[0,"y_true"] += 1
        pred.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "True labels"):
            self.load()
        pred = pred.iloc[:-1]
        pred.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.load()

    def test_rejects_duplicate_pair(self):
        path = self.root / "runs/kiba/cold_drug/frozen/seed42/evaluation_test/predictions.parquet"
        pred = pd.read_parquet(path)
        pd.concat([pred,pred.iloc[:1]]).to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "identity columns"):
            self.load()

    def test_rejects_wrong_checkpoint_or_fingerprint(self):
        run = self.root / "runs/kiba/cold_drug/scratch/seed42"
        (run / "best.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Wrong test checkpoint"):
            self.load()
        metric = read_json(run / "evaluation_test/metrics.json")
        metric["checkpoint_sha256"] = digest(run / "best.pt")
        write_json(run / "evaluation_test/metrics.json", metric)
        signature = read_json(run / "dataset_signature.json")
        signature["split_sha256"] = "bad"
        write_json(run / "dataset_signature.json", signature)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.load()

    def test_rejects_tuning_changes(self):
        path = self.root / "runs/kiba/cold_drug/finetune/seed43/config.json"
        cfg = read_json(path)
        cfg["training"]["learning_rate"] = .1
        write_json(path, cfg)
        with self.assertRaisesRegex(ValueError, "settings changed across seeds"):
            self.load()

    def test_rejects_mismatched_saved_error(self):
        path = self.root / "runs/kiba/cold_drug/frozen/seed44/evaluation_test/metrics.json"
        metric = read_json(path)
        metric["mse"] += .1
        write_json(path, metric)
        with self.assertRaisesRegex(ValueError, "Saved mse"):
            self.load()

    def test_partition_leakage_is_rejected(self):
        frame = pd.read_parquet(self.root / "data/kiba/processed/pairs.parquet")
        split = read_json(self.root / "data/kiba/processed/splits/cold_drug_seed42.json")
        frame.loc[frame.drug_id == "d4", "drug_id"] = "d0"
        frame.loc[frame.pair_id.between(16,19), "protein_id"] = ["new0","new1","new2","new3"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_frame(frame, split, "cold_drug")

    def test_bootstrap_constant_difference_and_fixed_seed_average(self):
        y = np.zeros(4)
        pred = np.array([[[1.]*4,[2.]*4,[.5]*4], [[2.]*4,[3.]*4,[1.]*4]])
        result, _ = paired_bootstrap(y, pred, ["a","b","b","b"], [42,43], 50, 7)
        row = next(r for r in result["records"] if (r["metric"],r["treatment"],r["control"],r["seed"]) == ("mse","finetune","scratch","mean_fixed_seeds"))
        self.assertAlmostEqual(row["difference_treatment_minus_control"], -1.875)
        np.testing.assert_allclose(row["difference_percentile_95_interval"], [-1.875,-1.875])

    def test_pair_weighting_not_drug_macro_average(self):
        y = np.zeros(10)
        pred = np.zeros((2,3,10))
        pred[:,0,0] = 10
        result, _ = paired_bootstrap(y, pred, ["a"] + ["b"]*9, [42,43], 100, 8)
        row = next(r for r in result["records"] if (r["metric"],r["treatment"],r["control"],r["seed"]) == ("mse","finetune","scratch","mean_fixed_seeds"))
        self.assertEqual(row["control_error"], 10.)
        self.assertEqual(row["difference_treatment_minus_control"], -10.)
        # Brute-force expand the same sampled clusters, including repeated pairs.
        rng = np.random.default_rng(8)
        samples = rng.multinomial(2, [.5,.5], size=100)
        brute = np.array([-100*a/(a+9*b) for a,b in samples])
        np.testing.assert_allclose(row["difference_percentile_95_interval"], np.quantile(brute,[.025,.975]))

    def test_bootstrap_reproducible_and_zero_baseline_is_not_infinite_percent(self):
        pred = np.zeros((2,3,4))
        first, _ = paired_bootstrap(np.zeros(4), pred, ["a","a","b","b"], [42,43], 50, 3)
        second, _ = paired_bootstrap(np.zeros(4), pred, ["a","a","b","b"], [42,43], 50, 3)
        self.assertEqual(first, second)
        self.assertTrue(all(r["relative_error_reduction_percent"] is None for r in first["records"]))

    def test_one_group_gets_warning_not_a_fake_interval(self):
        source = self.load()
        source["test"] = source["test"].copy()
        source["test"]["drug_id"] = "single-held-out-drug"
        with patch("analysis.load_results", return_value=source), contextlib.redirect_stdout(io.StringIO()):
            review(self.root, self.root / "single_group_report", datasets=("kiba",), replicates=500)
        result = read_json(self.root / "single_group_report/paired_group_bootstrap.json")["kiba"]
        self.assertEqual(result["records"], [])
        self.assertIn("no informative", (self.root / "single_group_report/REPORT.md").read_text())

    def test_duplicate_endpoints_are_rejected(self):
        frame = pd.read_parquet(self.root / "data/kiba/processed/pairs.parquet")
        frame.loc[1,"protein_id"] = frame.loc[0,"protein_id"]
        split = read_json(self.root / "data/kiba/processed/splits/cold_drug_seed42.json")
        with self.assertRaisesRegex(ValueError, "Duplicate drug-protein"):
            audit_frame(frame, split, "cold_drug")

    def test_report_outputs_no_overwrite(self):
        with contextlib.redirect_stdout(io.StringIO()):
            review(self.root, self.root / "report", replicates=500)
        self.assertEqual(len(pd.read_csv(self.root / "report/per_seed_metrics.csv")), 18)
        self.assertEqual(len(pd.read_csv(self.root / "report/seed_summary.csv")), 36)
        self.assertIn("PCBA validation", (self.root / "report/REPORT.md").read_text())
        with self.assertRaises(FileExistsError):
            review(self.root, self.root / "report", replicates=500)

    def test_plan_only_cli_does_not_train_or_change_splits(self):
        script = Path(__file__).with_name("run.py")
        proc = subprocess.run([sys.executable, str(script), "--project", str(self.root),
                               "--replicates", "500", "--plan-only"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertFalse((self.root / "runs/kiba/scaffold").exists())
        self.assertFalse((self.root / "data/kiba/processed/splits/scaffold_seed42.json").exists())
        self.assertEqual(len(list((self.root / "reports/transfer_validation").glob("*/results_bundle.zip"))), 1)

    def test_locked_configs_and_all_training_before_testing(self):
        source = {d: self.load(d) for d in ("kiba","davis")}
        output = self.root / "scaffold_job"
        output.mkdir()
        calls = []

        def fake_command(project, *args):
            calls.append(args[0])
            if args[0] == "make_split.py":
                directory = Path(args[2])
                split = read_json(directory / "splits/cold_drug_seed42.json")
                split["metadata"]["mode"] = "scaffold"
                write_json(directory / "splits/scaffold_seed42.json", split)
            elif args[0] == "audit.py":
                write_json(args[-1], {"fixture": True})
            elif args[0] == "train_dta.py":
                make_run(read_json(args[2]), evaluate=False)
            elif args[0] == "evaluate.py":
                make_evaluation(Path(args[2]).parent)
            else:
                raise AssertionError(args)

        with patch("run.command", side_effect=fake_command), contextlib.redirect_stdout(io.StringIO()):
            run_scaffold(self.root, output, source, ("kiba","davis"), SEEDS)
            review(self.root, output / "review", "scaffold", replicates=500)
        self.assertEqual(calls.count("train_dta.py"), 18)
        self.assertEqual(calls.count("evaluate.py"), 18)
        self.assertGreater(calls.index("evaluate.py"), max(i for i,x in enumerate(calls) if x == "train_dta.py"))
        for dataset in ("kiba","davis"):
            for seed in SEEDS:
                for variant in VARIANTS:
                    cfg = read_json(self.root / f"runs/{dataset}/scaffold/{variant}/seed{seed}/config.json")
                    original = source[dataset]["configs"][variant,seed]
                    self.assertEqual(cfg["model"], original["model"])
                    self.assertEqual(cfg["transfer"], original["transfer"])
                    self.assertEqual({k:v for k,v in cfg["training"].items() if k != "output_dir"},
                                     {k:v for k,v in original["training"].items() if k != "output_dir"})
        second = self.root / "scaffold_second"
        second.mkdir()
        calls.clear()
        with patch("run.command", side_effect=fake_command), contextlib.redirect_stdout(io.StringIO()):
            run_scaffold(self.root, second, source, ("kiba","davis"), SEEDS)
        self.assertNotIn("train_dta.py", calls)
        self.assertNotIn("evaluate.py", calls)

    def test_incomplete_run_and_writer_lock_are_stop_conditions(self):
        source = self.load()
        cfg = scaffold_config(self.root, source["configs"]["scratch",42], "kiba", "scratch", 42)
        run = Path(cfg["training"]["output_dir"])
        run.mkdir(parents=True)
        write_json(run / "config.json", cfg)
        (run / "last.pt").write_bytes(b"dummy")
        (run / "best.pt").write_bytes(b"dummy")
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            training_action(run, cfg, False)
        self.assertEqual(training_action(run, cfg, True), "resume")
        (run / ".writer.lock").write_text("123")
        with self.assertRaisesRegex(RuntimeError, "writer lock"):
            training_action(run, cfg, True)

    def test_training_failure_stops_before_any_test(self):
        source = {"kiba": self.load()}
        directory = self.root / "data/kiba/processed"
        split = read_json(directory / "splits/cold_drug_seed42.json")
        split["metadata"]["mode"] = "scaffold"
        write_json(directory / "splits/scaffold_seed42.json", split)
        output = self.root / "failed_job"
        output.mkdir()
        calls = []
        def fail(project, *args):
            calls.append(args[0])
            if args[0] == "train_dta.py":
                raise subprocess.CalledProcessError(1, list(args))
        with patch("run.command", side_effect=fail), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                run_scaffold(self.root, output, source, ("kiba",), SEEDS)
        self.assertNotIn("evaluate.py", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
