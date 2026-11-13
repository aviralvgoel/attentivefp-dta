"""Masked classification metrics and efficient O(N log N) concordance index."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def masked_bce(logits, y, mask, pos_weight=None):
    if logits.shape != y.shape or mask.shape != y.shape or logits.ndim != 2:
        raise ValueError(
            f"Expected equal [B, tasks] logits/targets/mask: {logits.shape}, {y.shape}, {mask.shape}"
        )
    mask = mask.bool()
    if not mask.any():
        return logits.sum() * 0.0
    if not torch.isfinite(y[mask]).all() or not ((y[mask] == 0) | (y[mask] == 1)).all():
        raise ValueError("Invalid observed binary labels")
    # Fill missing labels BEFORE BCE, not after: NaN * zero is still NaN.
    clean = y.masked_fill(~mask, 0).float()
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), clean, reduction="none", pos_weight=pos_weight
    )
    return loss[mask].mean()


def classification_metrics(y, probability, mask):
    scores = []
    for task in range(y.shape[1]):
        labels = y[mask[:, task], task]
        pred = probability[mask[:, task], task]
        valid = len(labels) > 0 and len(np.unique(labels)) == 2
        scores.append(
            {
                "task": task,
                "observed": len(labels),
                "positives": int((labels == 1).sum()),
                "average_precision": float(average_precision_score(labels, pred))
                if valid
                else None,
                "roc_auc": float(roc_auc_score(labels, pred)) if valid else None,
            }
        )
    valid = [row for row in scores if row["average_precision"] is not None]
    # AP is NOT trapezoidal PR-AUC. The exact definition is intentionally explicit.
    return {
        "macro_ap": float(np.mean([r["average_precision"] for r in valid])) if valid else None,
        "macro_roc_auc": float(np.mean([r["roc_auc"] for r in valid])) if valid else None,
        "valid_tasks": len(valid),
        "total_tasks": y.shape[1],
        "per_task": scores,
    }


def concordance_index(y, pred):
    """Higher prediction should correspond to higher true score; ties in y excluded.

    Fenwick tree counts lower/equal previous predictions. Equal prediction gets
    half credit. Process equal-y groups before adding them to avoid counting
    incomparable tied labels. Does not allocate an N-by-N pair matrix.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ranks = np.searchsorted(np.unique(pred), pred) + 1
    order = np.argsort(y, kind="stable")
    tree = np.zeros(int(ranks.max(initial=0)) + 1, dtype=np.int64)

    def query(i):
        total = 0
        while i > 0:
            total += int(tree[i])
            i -= i & -i
        return total

    numerator, denominator, seen, start = 0.0, 0, 0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and y[order[end]] == y[order[start]]:
            end += 1
        for idx in order[start:end]:
            rank = int(ranks[idx])
            lower = query(rank - 1)
            equal = query(rank) - lower
            numerator += lower + 0.5 * equal
            denominator += seen
        for idx in order[start:end]:
            rank = int(ranks[idx])
            while rank < len(tree):
                tree[rank] += 1
                rank += rank & -rank
        seen += end - start
        start = end
    return numerator / denominator if denominator else None


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if (
        not len(y)
        or y.shape != pred.shape
        or not np.isfinite(y).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError("Empty, mismatched or nonfinite regression predictions")
    mse = float(np.mean((y - pred) ** 2))
    varying = len(y) > 1 and np.std(y) > 0 and np.std(pred) > 0
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": float(np.mean(np.abs(y - pred))),
        "ci": concordance_index(y, pred),
        "pearson": float(pearsonr(y, pred).statistic) if varying else None,
        "spearman": float(spearmanr(y, pred).statistic) if varying else None,
        "pairs": len(y),
    }
