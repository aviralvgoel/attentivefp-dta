"""Run current-result review, grouped bootstrap, and locked scaffold experiments."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
import zipfile

from analysis import VARIANTS, audit_frame, digest, read_json, review, write_json


def command(project, *args):
    argv = [sys.executable, "-u", *map(str, args)]
    print("\nRUN " + shlex.join(argv), flush=True)
    subprocess.run(argv, cwd=project, check=True)


def scaffold_config(project, source, dataset, variant, seed):
    cfg = copy.deepcopy(source)
    if cfg["seed"] != seed or cfg["transfer"]["mode"] != variant:
        raise ValueError("Source configuration has wrong seed or variant")
    # Only paths/split change: all model, optimizer, precision, warmup and
    # early-stopping settings remain exactly those of the completed cold run.
    cfg["root"] = str(project)
    cfg["data"]["processed_dir"] = str(project / f"data/{dataset}/processed")
    cfg["data"]["protein_cache"] = str(project / f"data/{dataset}/esm2_mean.pt")
    cfg["data"]["split_file"] = str(project / f"data/{dataset}/processed/splits/scaffold_seed42.json")
    cfg["training"]["output_dir"] = str(project / f"runs/{dataset}/scaffold/{variant}/seed{seed}")
    if variant != "scratch":
        if not Path(cfg["transfer"]["checkpoint"]).is_file():
            raise FileNotFoundError("Original PCBA checkpoint moved. Restore its original path before continuing.")
    return cfg


def training_action(run, expected, resume_incomplete):
    if not run.exists() or not any(run.iterdir()):
        return "train"
    if (run / ".writer.lock").exists():
        raise RuntimeError(f"Run has a writer lock; do not start another writer: {run}")
    if not (run / "config.json").is_file() or read_json(run / "config.json") != expected:
        raise ValueError(f"Existing run settings do not match the locked protocol: {run}")
    finished = ((run / "metrics.json").is_file() and (run / "best.pt").is_file()
                and (run / "train.log").is_file()
                and "Finished; best epoch=" in (run / "train.log").read_text())
    if finished:
        return "skip"
    if resume_incomplete and (run / "last.pt").is_file() and (run / "best.pt").is_file():
        return "resume"
    raise RuntimeError(f"Incomplete scaffold run: {run}. Inspect its log; use --resume-incomplete only after confirming no writer is active. Nothing was deleted.")


def run_scaffold(project, output, source, datasets, seeds, resume_incomplete=False):
    locked = output / "locked_configs"
    locked.mkdir()
    jobs = []
    for dataset in datasets:
        directory = project / f"data/{dataset}/processed"
        split = directory / "splits/scaffold_seed42.json"
        if not split.exists():
            command(project, "make_split.py", "--processed-dir", directory, "--mode", "scaffold", "--seed", "42")
        frame = source[dataset]["frame"]
        payload = read_json(split)
        if payload.get("metadata", {}).get("pairs_sha256") != digest(directory / "pairs.parquet"):
            raise ValueError("Existing scaffold split is not tied to the current pairs table")
        _, counts = audit_frame(frame, payload, "scaffold")
        write_json(output / f"{dataset}_scaffold_counts.json", counts)
        print(f"{dataset} scaffold partition counts: {json.dumps(counts)}", flush=True)
        command(project, "audit.py", "dta", "--processed-dir", directory,
                "--split-file", split, "--output", output / f"{dataset}_scaffold_audit.json")
        for seed in seeds:
            for variant in VARIANTS:
                cfg = scaffold_config(project, source[dataset]["configs"][variant, seed], dataset, variant, seed)
                path = locked / f"{dataset}_{variant}_seed{seed}.json"
                write_json(path, cfg)
                run = Path(cfg["training"]["output_dir"])
                action = training_action(run, cfg, resume_incomplete)
                jobs.append((dataset, seed, variant, path, run, action))
    # All paths/configurations are preflighted before starting GPU jobs.
    for dataset, seed, variant, config, run, action in jobs:
        print(f"\nScaffold {dataset} / {variant} / seed {seed}: {action}", flush=True)
        if action != "skip":
            args = ["train_dta.py", "--config", config]
            if action == "resume":
                args.extend(["--resume", run / "last.pt"])
            command(project, *args)
    # No test evaluation until every training job has succeeded.
    for _, _, _, _, run, _ in jobs:
        evaluation = run / "evaluation_test"
        if evaluation.exists() and any(evaluation.iterdir()):
            if not (evaluation / "metrics.json").is_file() or not (evaluation / "predictions.parquet").is_file():
                raise RuntimeError(f"Partial test evaluation at {evaluation}; inspect it instead of overwriting it.")
            if read_json(evaluation / "metrics.json").get("checkpoint_sha256") != digest(run / "best.pt"):
                raise ValueError(f"Existing test results use a different checkpoint: {run}")
            print(f"Keeping existing evaluation: {evaluation}", flush=True)
        else:
            command(project, "evaluate.py", "--checkpoint", run / "best.pt", "--split", "test", "--device", "cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("kiba", "davis"), default=["kiba", "davis"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--review-only", action="store_true", help="Steps 1 and 2 only; no new training/splits")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="After current-result review, print planned scaffold jobs without writing splits or training")
    args = parser.parse_args()
    project = args.project.resolve()
    if not (project / "train_dta.py").is_file() or not (project / "src/dta_project/splits.py").is_file():
        raise ValueError("--project must be the extracted attentivefp_dta_final project")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Use unique datasets and at least two unique training seeds")
    if args.replicates < 500:
        raise ValueError("Use at least 500 bootstrap replicates (default 5000)")
    parent = project / "reports/transfer_validation"
    parent.mkdir(parents=True, exist_ok=True)
    # An advisory OS lock is released automatically even after process death.
    with (parent / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another validation pipeline is active; do not launch a duplicate") from exc
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"{stamp}_{uuid.uuid4().hex[:8]}"
        output.mkdir()
        print(f"Report directory: {output}", flush=True)
        write_json(output / "request.json", {"project": str(project), "datasets": args.datasets,
                   "seeds": args.seeds, "replicates": args.replicates,
                   "review_only": args.review_only, "plan_only": args.plan_only})
        try:
            source = review(project, output / "cold_drug", "cold_drug", args.datasets, args.seeds, args.replicates)
            if args.plan_only:
                plan = []
                for dataset in args.datasets:
                    for seed in args.seeds:
                        for variant in VARIANTS:
                            cfg = scaffold_config(project, source[dataset]["configs"][variant,seed], dataset, variant, seed)
                            plan.append({"dataset": dataset, "variant": variant, "seed": seed, "config": cfg})
                write_json(output / "scaffold_plan.json", plan)
                print(f"Planned {len(plan)} scaffold runs. No split or training was created.", flush=True)
            elif not args.review_only:
                run_scaffold(project, output, source, args.datasets, args.seeds, args.resume_incomplete)
                review(project, output / "scaffold", "scaffold", args.datasets, args.seeds, args.replicates)
            write_json(output / "status.json", {"status": "complete", "scaffold_executed": not (args.review_only or args.plan_only)})
        except Exception as exc:
            write_json(output / "status.json", {"status": "failed", "error": str(exc)})
            print(f"STOPPED: {exc}\nCompleted reports remain at {output}. No existing outputs were deleted.", flush=True)
            raise
        bundle = output / "results_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path != bundle:
                    archive.write(path, path.relative_to(output))
        print(f"\nCOMPLETE. Results bundle to share: {bundle}", flush=True)


if __name__ == "__main__":
    main()
