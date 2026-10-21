"""Metrics for the insecure (positive) class, and threshold tuning on validation."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def tune_threshold(y, score):
    """Threshold that maximises F1 on the given (validation) data."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5


def evaluate(y, score, threshold):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(score) >= threshold
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(
        pr_auc=float(average_precision_score(y, score)) if y.any() else float("nan"),
        recall=float(recall), precision=float(precision),
        f1=float(2 * precision * recall / max(precision + recall, 1e-12)),
        accuracy=float((pred == y).mean()), base_rate=float(y.mean()), n=int(len(y)))
