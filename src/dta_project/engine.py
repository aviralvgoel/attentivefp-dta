"""Shared training/evaluation loops; never evaluate test data during fitting."""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .common import (
    RunDirectory,
    atomic_save,
    environment,
    jsonable,
    load_checkpoint,
    logger_for,
    restore_rng,
    rng_state,
    write_json,
)
from .dta import to_device
from .metrics import classification_metrics, masked_bce, regression_metrics


class Precision:
    def __init__(self, name, device):
        if name == "auto":
            name = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        if name not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be auto, fp32, bf16 or fp16")
        if name == "fp16" and device.type != "cuda":
            raise ValueError("Use fp32 for CPU; fp16 training is supported only on CUDA here")
        if name == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support BF16; choose fp32 or fp16")
        self.name, self.device = name, device
        self.scaler = torch.amp.GradScaler("cuda", enabled=name == "fp16")

    def context(self):
        if self.name == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16 if self.name == "bf16" else torch.float16
        )


def gradient_step(loss, model, optimizer, precision, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Nonfinite loss; training stopped before saving a corrupted checkpoint"
        )
    precision.scaler.scale(loss).backward()
    precision.scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    precision.scaler.step(optimizer)
    precision.scaler.update()
    return float(norm)


def train_epoch(
    model,
    loader,
    task,
    device,
    optimizer,
    precision,
    config,
    pos_weight=None,
    label_stats=None,
    logger=None,
):
    model.train()
    loss_sum, count, steps, norm_sum = 0.0, 0, 0, 0.0
    start = time.monotonic()
    for batch in tqdm(loader, desc="train", leave=False, disable=not config.get("progress", True)):
        optimizer.zero_grad(set_to_none=True)
        if task == "pcba":
            batch = batch.to(device, non_blocking=True)
            observed = int(batch.label_mask.sum().item())
            if observed == 0:
                continue  # No optimizer decay step from completely unobserved data.
            with precision.context():
                logits = model(batch)
                loss = masked_bce(logits, batch.y, batch.label_mask, pos_weight)
            weight = observed
        else:
            batch = to_device(batch, device)
            target = (batch["y"] - label_stats["mean"]) / label_stats["std"]
            with precision.context():
                prediction = model(batch)
                loss = F.mse_loss(prediction.float(), target.float())
            weight = len(target)
        norm_sum += gradient_step(loss, model, optimizer, precision, config["gradient_clip"])
        loss_sum += float(loss.detach()) * weight
        count += weight
        steps += 1
        if logger and steps % config.get("log_every", 100) == 0:
            logger.info(
                "batch=%s train_loss=%.6f elapsed_s=%.1f",
                steps,
                loss_sum / count,
                time.monotonic() - start,
            )
    if not steps:
        raise ValueError("No training steps with observed labels")
    return {
        "loss": loss_sum / count,
        "gradient_norm": norm_sum / steps,
        "steps": steps,
        "seconds": time.monotonic() - start,
    }


@torch.inference_mode()
def evaluate_pcba(model, loader, device, pos_weight=None):
    model.eval()
    ys, ps, masks, ids = [], [], [], []
    loss_sum, weighted_sum, observed_total = 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        n = int(batch.label_mask.sum().item())
        if n:
            loss_sum += float(masked_bce(logits, batch.y, batch.label_mask)) * n
            weighted_sum += float(masked_bce(logits, batch.y, batch.label_mask, pos_weight)) * n
        observed_total += n
        ys.append(batch.y.cpu().numpy())
        ps.append(logits.sigmoid().cpu().numpy())
        masks.append(batch.label_mask.cpu().numpy())
        ids.append(batch.molecule_id.cpu().numpy())
    if not observed_total:
        raise ValueError("Evaluation partition has no observed labels")
    y, pred, mask = np.concatenate(ys), np.concatenate(ps), np.concatenate(masks)
    metrics = classification_metrics(y, pred, mask)
    metrics.update(
        masked_bce=loss_sum / observed_total,
        weighted_bce=weighted_sum / observed_total,
        observed_labels=observed_total,
    )
    return metrics, {"molecule_id": np.concatenate(ids), "y": y, "probability": pred, "mask": mask}


@torch.inference_mode()
def evaluate_dta(model, loader, device, stats):
    model.eval()
    ys, ps, ids = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch).float() * stats["std"] + stats["mean"]
        ys.append(batch["y"].cpu().numpy())
        ps.append(pred.cpu().numpy())
        ids.append(batch["pair_id"].cpu().numpy())
    y, pred = np.concatenate(ys), np.concatenate(ps)
    metrics = regression_metrics(y, pred)
    metrics["mean_predictor_mse"] = float(
        np.mean((y - stats.get("train_mean", stats["mean"])) ** 2)
    )
    return metrics, {"pair_id": np.concatenate(ids), "y_true": y, "y_pred": pred}


def compatibility_config(config):
    # Device/loader speed may be changed during recovery; scientific settings may not.
    result = json.loads(json.dumps(config))
    result.pop("root", None)
    for key in ("device", "workers", "progress", "log_every", "stop_after_epoch"):
        result["training"].pop(key, None)
    return result


def fit(
    model,
    task,
    config,
    model_config,
    device,
    train_loader_factory,
    validation_loader,
    data_signature,
    schema_hash,
    *,
    pos_weight=None,
    label_stats=None,
    resume=None,
    protein_signature=None,
):
    tc = config["training"]
    if tc["epochs"] < 1 or tc["patience"] < 1 or tc["batch_size"] < 1 or tc["workers"] < 0:
        raise ValueError("Invalid training length, patience, batch size or workers")
    if task == "dta":
        transfer = config["transfer"]
        if transfer["mode"] not in ("scratch", "frozen", "finetune"):
            raise ValueError("transfer.mode must be scratch, frozen or finetune")
        if transfer["mode"] == "finetune" and not 0 <= transfer["warmup_epochs"] < tc["epochs"]:
            raise ValueError("Fine-tuning warm-up must be shorter than total epochs")
        encoder_params = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other = [p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": tc["encoder_learning_rate"]},
                {"params": other, "lr": tc["learning_rate"]},
            ],
            weight_decay=tc["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    precision = Precision(tc["precision"], device)
    checkpoint = load_checkpoint(resume) if resume else None
    if resume and Path(resume).resolve() != (Path(tc["output_dir"]) / "last.pt").resolve():
        raise ValueError("Resume the same run's last.pt; use transfer initialization for a new run")
    start_epoch, best_score, stale, selection = 1, -math.inf, 0, None
    if checkpoint:
        if checkpoint["task"] != task or checkpoint["data_signature"] != data_signature:
            raise ValueError("Cannot resume with a different task/dataset/split")
        if checkpoint.get("protein_signature") != protein_signature:
            raise ValueError("Cannot resume with different protein embeddings")
        if compatibility_config(checkpoint["config"]) != compatibility_config(config):
            raise ValueError(
                "Scientific configuration changed; do not resume into an incomparable experiment"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        precision.scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_score, stale, selection = (
            checkpoint["best_score"],
            checkpoint["stale_epochs"],
            checkpoint["selection_metric"],
        )
        restore_rng(checkpoint["rng_state"])
    with RunDirectory(tc["output_dir"], resume=bool(resume)) as out:
        logger = logger_for(out)
        write_json(out / "config.json", config)
        write_json(out / "environment.json", environment())
        write_json(out / "dataset_signature.json", data_signature)
        logger.info(
            "task=%s device=%s precision=%s params=%s",
            task,
            device,
            precision.name,
            sum(p.numel() for p in model.parameters()),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        best_path, last_path = out / "best.pt", out / "last.pt"
        history_path = out / "history.jsonl"
        for epoch in range(start_epoch, tc["epochs"] + 1):
            if task == "dta":
                freeze = transfer["mode"] == "frozen" or (
                    transfer["mode"] == "finetune" and epoch <= transfer["warmup_epochs"]
                )
                model.freeze_encoder(freeze)
                if transfer["mode"] == "finetune" and epoch == transfer["warmup_epochs"] + 1:
                    stale = 0
                logger.info("epoch=%s drug_encoder_frozen=%s", epoch, freeze)
            train = train_epoch(
                model,
                train_loader_factory(epoch),
                task,
                device,
                optimizer,
                precision,
                tc,
                pos_weight,
                label_stats,
                logger,
            )
            if task == "pcba":
                val, _ = evaluate_pcba(model, validation_loader, device, pos_weight)
                if selection is None:
                    selection = "macro_ap" if val["macro_ap"] is not None else "negative_masked_bce"
                    if selection != "macro_ap" and not tc["smoke_allow_bce_fallback"]:
                        raise ValueError(
                            "No valid validation assays. BCE fallback is only allowed for explicit smoke tests."
                        )
                if selection == "macro_ap" and val["macro_ap"] is None:
                    raise ValueError(
                        "Validation AP became undefined; do not compare incompatible score scales"
                    )
                score = val["macro_ap"] if selection == "macro_ap" else -val["masked_bce"]
            else:
                val, _ = evaluate_dta(model, validation_loader, device, label_stats)
                selection, score = "negative_mse", -val["mse"]
            if not math.isfinite(score):
                raise FloatingPointError("Validation selection score is nonfinite")
            scheduler.step()
            improved = score > best_score + tc.get("min_delta", 0)
            if improved:
                best_score, stale = score, 0
            else:
                stale += 1
            row = {
                "epoch": epoch,
                "train": train,
                "validation": val,
                "selection_metric": selection,
                "selection_score": score,
                "encoder_frozen": getattr(model, "encoder_frozen", False),
                "learning_rates_next_epoch": [g["lr"] for g in optimizer.param_groups],
            }
            with history_path.open("a") as stream:
                stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            payload = {
                "format_version": 1,
                "task": task,
                "epoch": epoch,
                "model_config": model_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": precision.scaler.state_dict(),
                "rng_state": rng_state(),
                "config": config,
                "data_signature": data_signature,
                "schema_hash": schema_hash,
                "protein_signature": protein_signature,
                "label_stats": label_stats,
                "pos_weight": pos_weight,
                "selection_metric": selection,
                "selection_score": score,
                "validation": val,
                "best_score": best_score,
                "stale_epochs": stale,
            }
            atomic_save(payload, last_path)
            if improved:
                atomic_save(payload, best_path)
                write_json(out / "validation_metrics.json", val)
            summary = {k: v for k, v in val.items() if k != "per_task"}
            logger.info(
                "epoch=%03d train_loss=%.6f val=%s selected=%s stale=%s/%s",
                epoch,
                train["loss"],
                json.dumps(jsonable(summary)),
                improved,
                stale,
                tc["patience"],
            )
            # Allow the fine-tuning phase to start before early stopping can terminate it.
            after_warmup = (
                task != "dta" or transfer["mode"] != "finetune" or epoch > transfer["warmup_epochs"]
            )
            if after_warmup and stale >= tc["patience"]:
                logger.info("Early stopping")
                break
            if tc.get("stop_after_epoch") is not None and epoch >= tc["stop_after_epoch"]:
                logger.info("Requested epoch-boundary pause; resume from last.pt")
                break
        if not best_path.exists():
            raise RuntimeError(
                "No best checkpoint. Resume requires the original best.pt as well as last.pt"
            )
        best = load_checkpoint(best_path)
        write_json(
            out / "metrics.json",
            {
                "task": task,
                "best_epoch": best["epoch"],
                "selection_metric": best["selection_metric"],
                "validation": best["validation"],
                "test_evaluated": False,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            },
        )
        logger.info(
            "Finished; best epoch=%s. Test set NOT evaluated. Checkpoint: %s",
            best["epoch"],
            best_path,
        )
        return best_path
