from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_score = probabilities[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "g_mean": float(np.sqrt(sensitivity * specificity)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, strategy: str) -> float:
    if strategy == "fixed_0.5":
        return 0.5
    y_score = probabilities[:, 1]
    candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    if len(candidates) == 0:
        return 0.5

    best_threshold = 0.5
    best_value = -1.0
    for threshold in candidates:
        metrics = classification_metrics(y_true, probabilities, float(threshold))
        if strategy == "maximize_f1":
            value = metrics["f1"]
        elif strategy == "maximize_g_mean":
            value = metrics["g_mean"]
        else:
            raise ValueError("threshold_strategy must be fixed_0.5, maximize_f1, or maximize_g_mean.")
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold
