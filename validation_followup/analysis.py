"""Read-only verification and paired group bootstrap for saved DTA predictions.

Intervals condition on the fitted models, training seeds and declared split.
They do NOT resample training seeds, select models, or change hyperparameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

VARIANTS = ("scratch", "frozen", "finetune")
METRICS = ("mse", "rmse", "mae", "ci", "pearson", "spearman")
PARTS = ("train", "validation", "test")
CONTRASTS = (("finetune", "scratch"), ("frozen", "scratch"), ("finetune", "frozen"))


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def partitions(payload, available):
    """Independently check a complete, non-overlapping, three-way ID split."""
    available = set(map(int, available))
    result = {}
    for name in PARTS:
        values = payload.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Empty or missing partition: {name}")
        if any(type(x) is not int for x in values):
            raise ValueError(f"Noninteger IDs in {name}")
        result[name] = set(values)
        if len(values) != len(result[name]) or not result[name] <= available:
            raise ValueError(f"Duplicate or unknown IDs in {name}")
    if any(result[a] & result[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Partition IDs overlap")
    if set.union(*result.values()) != available:
        raise ValueError("Partition IDs do not cover the dataset")
    return result


def audit_frame(frame, split, mode):
    required = {"pair_id", "drug_id", "protein_id", "canonical_smiles", "scaffold", "label"}
    if not required <= set(frame) or frame.pair_id.duplicated().any():
        raise ValueError("Invalid or incomplete pairs.parquet")
    if frame.duplicated(["drug_id", "protein_id"]).any():
        raise ValueError("Duplicate drug-protein endpoints should have been merged before splitting")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing pair identity, scaffold, or label")
    if split.get("metadata", {}).get("mode") != mode:
        raise ValueError("Split metadata does not match the requested setting")
    ids = partitions(split, frame.pair_id)
    tables = {p: frame[frame.pair_id.isin(ids[p])] for p in PARTS}
    strict = ("drug_id", "scaffold") if mode == "scaffold" else ("drug_id",)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in strict:
            if set(tables[a][key]) & set(tables[b][key]):
                raise ValueError(f"{mode} leakage between {a}/{b}: {key}")
    counts = {p: {"pairs": len(t), "drugs": int(t.drug_id.nunique()),
                  "proteins": int(t.protein_id.nunique()), "scaffolds": int(t.scaffold.nunique())}
              for p, t in tables.items()}
    return tables, counts


def load_pcba(project):
    root = project / "data/pcba/processed"
    index_path, split_path = root / "index.parquet", root / "splits/scaffold_seed42.json"
    index = pd.read_parquet(index_path)
    if not {"global_index", "canonical_smiles", "scaffold"} <= set(index):
        raise ValueError("PCBA index needs global_index, canonical_smiles and scaffold")
    if index.global_index.duplicated().any() or index[["canonical_smiles", "scaffold"]].isna().any().any():
        raise ValueError("PCBA index has duplicate IDs or missing chemical identities")
    ids = partitions(read_json(split_path), index.global_index)
    return {
        "schema_hash": read_json(root / "metadata.json").get("schema_hash"),
        "hashes": {str(p.relative_to(project)): digest(p) for p in (index_path, split_path, root / "metadata.json")},
        "parts": {p: {"drugs": set(index[index.global_index.isin(v)].canonical_smiles),
                      "scaffolds": set(index[index.global_index.isin(v)].scaffold)} for p, v in ids.items()},
    }


def exposure(tables, pcba):
    report = {}
    for part, table in tables.items():
        drugs, cores = set(table.canonical_smiles), set(table.scaffold)
        report[part] = {"unique_drugs": len(drugs), "unique_scaffolds": len(cores),
                       "pcba": {p: {"drugs": len(drugs & seen["drugs"]),
                                    "scaffolds": len(cores & seen["scaffolds"])}
                                for p, seen in pcba["parts"].items()}}
    return report


def comparable_config(config, across_variants=False):
    config = copy.deepcopy(config)
    config.pop("seed", None)
    config["training"].pop("output_dir", None)
    for key in ("device", "workers", "progress", "log_every"):
        config["training"].pop(key, None)
    if across_variants:
        config.pop("transfer", None)
        config["training"].pop("encoder_learning_rate", None)
    return config


def load_results(project, dataset, mode, seeds, pcba):
    directory = project / f"data/{dataset}/processed"
    split_path = directory / f"splits/{mode}_seed42.json"
    frame = pd.read_parquet(directory / "pairs.parquet")
    metadata = read_json(directory / "metadata.json")
    if not metadata.get("schema_hash") or metadata["schema_hash"] != pcba["schema_hash"]:
        raise ValueError("DTA and PCBA feature schemas differ or are unknown")
    if dataset == "davis" and metadata.get("label_transform") != "neg_log10_nM":
        raise ValueError("Expected raw-DAVIS nM -> pKd preprocessing for this protocol")
    if dataset == "kiba" and metadata.get("label_transform") != "none":
        raise ValueError("KIBA labels must not be log transformed")
    tables, counts = audit_frame(frame, read_json(split_path), mode)
    test = tables["test"].sort_values("pair_id").reset_index(drop=True)
    expected_hashes = {key: digest(path) for key, path in (
        ("pairs_sha256", directory / "pairs.parquet"),
        ("graphs_sha256", directory / "drug_graphs.pt"),
        ("metadata_sha256", directory / "metadata.json"), ("split_sha256", split_path))}
    predictions = np.empty((len(seeds), len(VARIANTS), len(test)), dtype=float)
    rows, configs, files = [], {}, {}
    for filename, key in (("pairs.parquet", "pairs_sha256"), ("drug_graphs.pt", "graphs_sha256"),
                          ("metadata.json", "metadata_sha256"), (f"splits/{mode}_seed42.json", "split_sha256")):
        files[str((directory / filename).relative_to(project))] = expected_hashes[key]
    cache = project / f"data/{dataset}/esm2_mean.pt"
    files[str(cache.relative_to(project))] = digest(cache)
    source_configs, baseline_mses = {}, []
    for si, seed in enumerate(seeds):
        for vi, variant in enumerate(VARIANTS):
            run = project / f"runs/{dataset}/{mode}/{variant}/seed{seed}"
            evaluation = run / "evaluation_test"
            config = read_json(run / "config.json")
            signature = read_json(run / "dataset_signature.json")
            for key, value in expected_hashes.items():
                if signature.get(key) != value:
                    raise ValueError(f"Input fingerprint mismatch: {run}, {key}")
            if config.get("seed") != seed or config["transfer"]["mode"] != variant:
                raise ValueError(f"Mislabelled seed/variant: {run}")
            if config["data"].get("allow_synthetic") or config["training"].get("stop_after_epoch") is not None:
                raise ValueError(f"Synthetic features or intentional partial run: {run}")
            if "Finished; best epoch=" not in (run / "train.log").read_text():
                raise ValueError(f"Training did not finish: {run}")
            reference = source_configs.setdefault(variant, comparable_config(config))
            if reference != comparable_config(config):
                raise ValueError(f"Scientific settings changed across seeds: {variant}")
            metric = read_json(evaluation / "metrics.json")
            if metric.get("split") != "test" or metric.get("checkpoint_sha256") != digest(run / "best.pt"):
                raise ValueError(f"Wrong test checkpoint: {run}")
            pred = pd.read_parquet(evaluation / "predictions.parquet")
            required = {"pair_id", "drug_id", "protein_id", "canonical_smiles", "scaffold", "y_true", "y_pred"}
            if not required <= set(pred) or pred.pair_id.duplicated().any():
                raise ValueError(f"Invalid prediction identity columns: {run}")
            pred = pred.sort_values("pair_id").reset_index(drop=True)
            for key in ("pair_id", "drug_id", "protein_id", "canonical_smiles", "scaffold"):
                if pred[key].tolist() != test[key].tolist():
                    raise ValueError(f"Test pair/identity mismatch: {run}, {key}")
            y, yhat = pred.y_true.to_numpy(float), pred.y_pred.to_numpy(float)
            if not np.isfinite(y).all() or not np.isfinite(yhat).all():
                raise ValueError(f"Nonfinite predictions: {run}")
            if not np.allclose(y, test.label.to_numpy(float), rtol=1e-6, atol=1e-6):
                raise ValueError(f"True labels disagree with curated data: {run}")
            if si == 0 and vi == 0:
                truth = y.copy()
            if not np.array_equal(y, truth):
                raise ValueError(f"Saved true labels differ across fitted models: {run}")
            for name, computed in (("mse", np.mean((y-yhat)**2)),
                                   ("mae", np.mean(np.abs(y-yhat))),
                                   ("rmse", np.sqrt(np.mean((y-yhat)**2)))):
                if not np.isclose(metric[name], computed, rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Saved {name} does not match predictions: {run}")
            if metric.get("pairs") != len(test):
                raise ValueError(f"Saved pair count differs: {run}")
            baseline_mses.append(float(metric["mean_predictor_mse"]))
            predictions[si, vi] = yhat
            rows.append({"dataset": dataset, "setting": mode, "variant": variant, "seed": seed,
                         "checkpoint_epoch": metric["checkpoint_epoch"], **{k: metric[k] for k in METRICS},
                         "pairs": len(test), "mean_predictor_mse": metric["mean_predictor_mse"]})
            configs[variant, seed] = config
            for path in (run / "config.json", run / "dataset_signature.json", run / "best.pt",
                         evaluation / "metrics.json", evaluation / "predictions.parquet"):
                files[str(path.relative_to(project))] = digest(path)
    if not np.allclose(baseline_mses, baseline_mses[0], rtol=1e-6, atol=1e-7):
        raise ValueError("Training-mean baselines differ despite a fixed split")
    references = [comparable_config(configs[v, seeds[0]], True) for v in VARIANTS]
    if any(c != references[0] for c in references[1:]):
        raise ValueError("Unexpected architecture/data/training-setting differences across variants")
    return {"test": test, "truth": truth, "predictions": predictions, "rows": rows,
            "configs": configs, "files": files, "counts": counts,
            "exposure": exposure(tables, pcba), "frame": frame,
            "metadata": metadata, "input_hashes": expected_hashes}


def paired_bootstrap(truth, predictions, groups, seeds, replicates=5000, rng_seed=20261103):
    """Sample G clusters, uniformly with replacement; retain ALL pairs in each.

    The statistic is pair-weighted error, averaged over the SAME fixed seeds.
    It is not an ensemble prediction and not a mean of per-drug MSE values.
    """
    truth, predictions = np.asarray(truth, float), np.asarray(predictions, float)
    if predictions.shape != (len(seeds), len(VARIANTS), len(truth)):
        raise ValueError("Expected [seeds, variants, pairs] predictions")
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("Empty/nonfinite bootstrap input")
    group_names, codes = np.unique(np.asarray(groups, str), return_inverse=True)
    g = len(group_names)
    if len(groups) != len(truth) or g < 2 or replicates < 2:
        raise ValueError("Need at least two groups and two bootstrap replicates")
    counts = np.bincount(codes, minlength=g)
    losses = {"mse": (predictions - truth[None, None, :])**2,
              "mae": np.abs(predictions - truth[None, None, :])}
    totals = {name: np.stack([np.stack([np.bincount(codes, weights=loss[s,v], minlength=g)
                                      for v in range(len(VARIANTS))]) for s in range(len(seeds))])
              for name, loss in losses.items()}
    rng = np.random.default_rng(rng_seed)
    draws = {k: [] for k in losses}
    for offset in range(0, replicates, 256):
        multiplicity = rng.multinomial(g, np.full(g, 1/g), size=min(256, replicates-offset))
        denominator = multiplicity @ counts
        for name in losses:
            draws[name].append(np.einsum("bg,svg->bsv", multiplicity, totals[name], optimize=True)
                               / denominator[:, None, None])
    draws = {name: np.concatenate(chunks) for name, chunks in draws.items()}
    records, error_rows = [], []
    for name in losses:
        point = losses[name].mean(axis=-1)
        for treatment, control in CONTRASTS:
            ti, ci = VARIANTS.index(treatment), VARIANTS.index(control)
            for label, point_values, sampled in (
                *[(str(seed), point[s], draws[name][:, s]) for s, seed in enumerate(seeds)],
                ("mean_fixed_seeds", point.mean(axis=0), draws[name].mean(axis=1)),
            ):
                delta = sampled[:, ti] - sampled[:, ci]
                lo, hi = np.quantile(delta, [0.025, 0.975])
                reduction = None
                reduction_interval = None
                if point_values[ci] > 0 and np.all(sampled[:, ci] > 0):
                    reduction = float(100*(point_values[ci]-point_values[ti])/point_values[ci])
                    reduction_interval = list(map(float, np.quantile(
                        100*(sampled[:,ci]-sampled[:,ti])/sampled[:,ci], [0.025, 0.975])))
                records.append({"metric": name, "treatment": treatment, "control": control,
                                "seed": label, "treatment_error": float(point_values[ti]),
                                "control_error": float(point_values[ci]),
                                "difference_treatment_minus_control": float(point_values[ti]-point_values[ci]),
                                "difference_percentile_95_interval": [float(lo), float(hi)],
                                "relative_error_reduction_percent": reduction,
                                "reduction_percentile_95_interval": reduction_interval})
    for s, seed in enumerate(seeds):
        for v, variant in enumerate(VARIANTS):
            for i, group in enumerate(group_names):
                error_rows.append({"seed": seed, "variant": variant, "group": group,
                                   "pairs": int(counts[i]), "squared_error_sum": float(totals["mse"][s,v,i]),
                                   "absolute_error_sum": float(totals["mae"][s,v,i])})
    return {"replicates": replicates, "rng_seed": rng_seed, "groups": g,
            "small_group_warning": g < 20,
            "method": "Paired percentile cluster bootstrap; uniformly resample groups with replacement; pair-weighted errors; fixed training seeds.",
            "interpretation": "Negative error differences favor treatment. Intervals condition on these models and this split; seed uncertainty and split/pretraining variability are not covered. Not multiplicity-adjusted. Primary contrast: finetune minus scratch, MSE. Other contrasts and MAE are secondary/exploratory. No p-values are reported.",
            "records": records}, error_rows


def review(project, output, mode="cold_drug", datasets=("kiba", "davis"), seeds=(42,43,44), replicates=5000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pcba = load_pcba(project)
    rows, summary_rows, bootstraps, reviews, inputs, loaded = [], [], {}, {}, {}, {}
    for dataset in datasets:
        print(f"Verifying {dataset}/{mode} predictions and fingerprints...", flush=True)
        data = load_results(project, dataset, mode, seeds, pcba)
        loaded[dataset] = data
        rows.extend(data["rows"])
        group_column = "scaffold" if mode == "scaffold" else "drug_id"
        n_groups = int(data["test"][group_column].nunique())
        if n_groups >= 2:
            bootstrap, errors = paired_bootstrap(data["truth"], data["predictions"],
                                                data["test"][group_column], seeds, replicates)
        else:
            bootstrap = {"groups": n_groups, "records": [], "small_group_warning": True,
                         "status": "insufficient_groups_for_an_informative_cluster_interval"}
            errors = []
            for si, seed in enumerate(seeds):
                for vi, variant in enumerate(VARIANTS):
                    err = data["predictions"][si,vi] - data["truth"]
                    errors.append({"seed": seed, "variant": variant,
                                   "group": str(data["test"][group_column].iloc[0]),
                                   "pairs": len(err), "squared_error_sum": float((err**2).sum()),
                                   "absolute_error_sum": float(np.abs(err).sum())})
        bootstrap["group_column"] = group_column
        bootstraps[dataset] = bootstrap
        pd.DataFrame(errors).to_csv(output / f"{dataset}_group_errors.csv", index=False)
        reviews[dataset] = {"label_transform": data["metadata"]["label_transform"],
                            "partitions": data["counts"], "pcba_exposure": data["exposure"]}
        inputs.update(data["files"])
        for variant in VARIANTS:
            for metric in METRICS:
                values = [r[metric] for r in data["rows"] if r["variant"] == variant]
                valid = [x for x in values if x is not None]
                summary_rows.append({"dataset": dataset, "setting": mode, "variant": variant,
                                     "metric": metric, "mean": mean(valid) if valid else None,
                                     "sample_sd": stdev(valid) if len(valid) > 1 else None,
                                     "valid_seeds": len(valid)})
    overlap = {}
    if len(loaded) == 2:
        a, b = (loaded[d]["frame"] for d in datasets)
        overlap = {"datasets": list(datasets), "shared_drugs": len(set(a.drug_id) & set(b.drug_id)),
                   "shared_exact_proteins": len(set(a.protein_id) & set(b.protein_id)),
                   "shared_exact_drug_protein_pairs": len(set(zip(a.drug_id,a.protein_id)) & set(zip(b.drug_id,b.protein_id))),
                   "note": "Labels have different meanings/scales. Shared entities/pairs mean the benchmarks are not wholly independent; no labels are merged."}
    pd.DataFrame(rows).to_csv(output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "seed_summary.csv", index=False)
    write_json(output / "paired_group_bootstrap.json", bootstraps)
    write_json(output / "data_and_exposure_review.json", {"datasets": reviews, "cross_dataset_overlap": overlap,
        "limitations": ["PCBA membership uses stored canonical identities and the shared feature schema.",
                        "PCBA validation compounds influenced checkpoint selection; training, validation and test memberships are reported separately.",
                        "Compound exposure does not establish assay endpoint overlap. Endpoints require a separate domain review.",
                        "Historical PCBA transfer checkpoint hashes were not recorded by the original trainer; current files/configuration paths do not prove historical immutability."]})
    write_json(output / "input_manifest.json", {"project": str(project), "mode": mode, "seeds": list(seeds),
               "files_sha256": {**pcba["hashes"], **inputs}})
    lines = [f"# {mode} transfer validation", "", "Numbers are computed from the project's saved test predictions; no scores are supplied by this helper.", "",
             "## Seed means and sample standard deviations", "",
             "| Dataset | Variant | MSE | MAE | Concordance index |", "|---|---|---|---|---|"]
    for dataset in datasets:
        for variant in VARIANTS:
            cells = []
            for metric in ("mse", "mae", "ci"):
                row = next(r for r in summary_rows if (r["dataset"],r["variant"],r["metric"]) == (dataset,variant,metric))
                cells.append(f"{row['mean']:.5f} ± {row['sample_sd']:.5f}" if row["sample_sd"] is not None else "undefined")
            lines.append(f"| {dataset} | {variant} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-seed fine-tuned minus scratch MSE", "",
              "| Dataset | Training seed | Difference (negative favors fine-tuning) |", "|---|---|---|"]
    for dataset in datasets:
        for seed in seeds:
            by_variant = {r["variant"]: r["mse"] for r in rows if r["dataset"] == dataset and r["seed"] == seed}
            lines.append(f"| {dataset} | {seed} | {by_variant['finetune']-by_variant['scratch']:.5f} |")
    lines += ["", "## Test-set exposure and independent group counts", "",
              "| Dataset | Pairs | Drugs | Scaffolds | Drugs in PCBA train | Drugs in PCBA validation |", "|---|---|---|---|---|---|"]
    for dataset in datasets:
        c = reviews[dataset]["partitions"]["test"]
        e = reviews[dataset]["pcba_exposure"]["test"]
        lines.append(f"| {dataset} | {c['pairs']} | {c['drugs']} | {c['scaffolds']} | {e['pcba']['train']['drugs']} | {e['pcba']['validation']['drugs']} |")
    if overlap:
        lines += ["", f"Across the two curated datasets: {overlap['shared_drugs']} shared drugs, {overlap['shared_exact_proteins']} shared exact proteins and {overlap['shared_exact_drug_protein_pairs']} shared drug-protein pairs. These benchmarks need not be wholly independent."]
    lines += ["", "## Primary paired comparison: fine-tuned minus scratch MSE", "",
              "Negative differences favor fine-tuning. These are conditional, pointwise 95% percentile intervals; not p-values or proof of causation.", ""]
    for dataset, bs in bootstraps.items():
        if not bs["records"]:
            lines.append(f"- {dataset}: only {bs['groups']} held-out {bs['group_column']} group; no informative cluster interval can be estimated.")
            continue
        r = next(r for r in bs["records"] if (r["metric"],r["treatment"],r["control"],r["seed"]) == ("mse","finetune","scratch","mean_fixed_seeds"))
        lo, hi = r["difference_percentile_95_interval"]
        lines.append(f"- {dataset}: difference {r['difference_treatment_minus_control']:.5f}; interval [{lo:.5f}, {hi:.5f}]; {bs['groups']} held-out {bs['group_column']} groups.")
        if bs["small_group_warning"]:
            lines.append("  Few groups (<20): treat the percentile interval cautiously; inspect group errors and additional fixed splits.")
        if lo <= 0 <= hi:
            lines.append("  The interval includes zero; these data do not clearly separate the models at this interval level.")
    lines += ["", "## Scope and safeguards", "",
              "- Predictions are aligned by pair IDs, with exact entity and saved-label checks across models/seeds.",
              "- Data and test-checkpoint fingerprints and saved errors are checked before analysis.",
              "- Errors remain pair-weighted; a drug with more pairs contributes more, as in the original benchmark metric.",
              "- Seeds remain fixed in the cluster bootstrap. Their sample SD is reported separately; this is not a combined seed-and-drug interval.",
              "- Scaffold reviews resample scaffolds rather than treating related drugs as independent.",
              "- Review data_and_exposure_review.json before claiming pretraining-unseen drugs or independent benchmarks.",
              "- Do not change hyperparameters based on these test reports.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return loaded
