"""
Evaluation metrics for group emotion classification.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy. Both tensors must be 1-D, equal length."""
    if preds.shape != targets.shape:
        raise ValueError(f"shape mismatch: {preds.shape} vs {targets.shape}")
    return float((preds == targets).float().mean().item())


def per_class_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> List[float]:
    """Recall per class (a.k.a. per-class accuracy). NaN where support is 0."""
    out = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float((preds[mask] == c).float().mean().item()))
    return out


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """``[num_classes, num_classes]`` int matrix. Rows = true, cols = predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    p = preds.cpu().numpy().astype(int)
    t = targets.cpu().numpy().astype(int)
    for ti, pi in zip(t, p):
        cm[ti, pi] += 1
    return cm


def weighted_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Weighted-by-support F1. NaN classes contribute 0."""
    p = preds.cpu().numpy().astype(int)
    t = targets.cpu().numpy().astype(int)
    total = len(t)
    if total == 0:
        return float("nan")
    f1_sum = 0.0
    for c in range(num_classes):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        support = int((t == c).sum())
        if tp + fp == 0 or tp + fn == 0 or support == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        f1_sum += f1 * (support / total)
    return float(f1_sum)


def macro_f1(preds, targets, num_classes):
    """Unweighted mean of per-class F1 (a class with no support contributes 0)."""
    p = preds.cpu().numpy().astype(int)
    t = targets.cpu().numpy().astype(int)
    f1s = []
    for c in range(num_classes):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")


def compute_all_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    class_names: List[str] | None = None,
) -> Dict[str, float | List[float] | np.ndarray]:
    """One-shot eval summary. Returns a dict for easy JSON-ification."""
    pca = per_class_accuracy(preds, targets, num_classes)
    return {
        "accuracy": accuracy(preds, targets),
        "weighted_f1": weighted_f1(preds, targets, num_classes),
        "macro_f1": macro_f1(preds, targets, num_classes),
        "per_class_accuracy": pca,
        "per_class_names": class_names or [f"class_{i}" for i in range(num_classes)],
        "confusion_matrix": confusion_matrix(preds, targets, num_classes),
        "num_samples": int(len(targets)),
    }
