"""Evaluation metrics and plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
RESULTS_PATH = REPORTS_DIR / "results.json"
FIGURES_DIR = REPORTS_DIR / "figures"


def detection_by_category(categories, y_pred) -> dict:
    """Share of flows flagged as attack for each attack_cat (for Normal this is the FPR)."""
    categories = np.asarray(categories)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        str(cat): {
            "n": int((categories == cat).sum()),
            "flagged": float(y_pred[categories == cat].mean()),
        }
        for cat in np.unique(categories)
    }


def evaluate_binary(y_true, scores, threshold: float, categories=None) -> dict:
    """Metrics for attack (1) vs normal (0); a flow is flagged when score >= threshold.

    With `categories` (attack_cat of each row) also adds the detection rate per category.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    both_classes = len(np.unique(y_true)) == 2

    metrics = {
        "n": len(y_true),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if both_classes else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if categories is not None:
        metrics["by_category"] = detection_by_category(categories, y_pred)
    return metrics


def choose_threshold(y_val, scores_val, mode: str = "fpr", max_fpr: float = 0.01) -> float:
    """Pick the threshold on validation data only.

    "fpr": highest recall with FPR <= max_fpr (depends only on how normal traffic scores).
    "f1": threshold with the best F1.
    """
    y_val = np.asarray(y_val)
    scores_val = np.asarray(scores_val, dtype=float)

    if mode == "fpr":
        fpr, tpr, thresholds = roc_curve(y_val, scores_val)
        ok = np.flatnonzero(fpr <= max_fpr)
        best = ok[np.argmax(tpr[ok])]
        threshold = thresholds[best]
        if np.isinf(threshold):
            # nothing can be flagged within the budget
            threshold = np.nextafter(scores_val.max(), np.inf)
        return float(threshold)

    if mode == "f1":
        precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
        p, r = precision[:-1], recall[:-1]
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        return float(thresholds[np.argmax(f1)])

    raise ValueError(f"mode must be 'fpr' or 'f1', got {mode!r}")


def evaluate_multiclass(y_true, y_pred, labels: list[str] | None = None) -> dict:
    """Accuracy, macro/weighted F1, per-class report and confusion matrix (rows = true class)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names: list = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])).tolist()

    # confusion_matrix silently drops rows with labels outside `names`
    unknown = sorted((set(np.unique(y_true)) | set(np.unique(y_pred))) - set(names))
    if unknown:
        raise ValueError(f"labels not in {names}: {unknown}")

    cm = confusion_matrix(y_true, y_pred, labels=names)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    # 0 when a class is never predicted / never present, same as zero_division=0 in sklearn
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)

    per_class = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for name, p, r, f, s in zip(names, precision, recall, f1, support)
    }
    return {
        "n": len(y_true),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
        "labels": names,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_test(evaluate, y_true, values, unseen, categories=None, **kwargs) -> dict:
    """Run `evaluate` on the whole test set and on the rows not seen in train.

    `values` are scores (binary) or predictions (multi-class), `unseen` is a boolean mask.
    `categories` is only for evaluate_binary and gets the same mask.
    """
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    unseen = np.asarray(unseen, dtype=bool)
    if categories is None:
        return {
            "test": evaluate(y_true, values, **kwargs),
            "test_unseen": evaluate(y_true[unseen], values[unseen], **kwargs),
        }

    categories = np.asarray(categories)
    return {
        "test": evaluate(y_true, values, categories=categories, **kwargs),
        "test_unseen": evaluate(y_true[unseen], values[unseen], categories=categories[unseen], **kwargs),
    }


def save_result(model: str, task: str, metrics: dict, params: dict | None = None,
                path: Path | str = RESULTS_PATH) -> list[dict]:
    """Add the entry to results.json, replacing an older one with the same model and task."""
    path = Path(path)
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    results = [r for r in results if (r["model"], r["task"]) != (model, task)]
    results.append({"model": model, "task": task, "params": params or {}, **metrics})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def results_table(task: str, subset: str = "test", metrics: list[str] | None = None,
                  path: Path | str = RESULTS_PATH) -> pd.DataFrame:
    """One row per model saved for `task`, with `metrics` taken from `subset` (val, test, test_unseen)."""
    if metrics is None:
        metrics = (["recall", "fpr", "precision", "f1", "roc_auc", "pr_auc"] if task == "binary"
                   else ["accuracy", "macro_f1", "weighted_f1"])
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {r["model"]: {m: r[subset][m] for m in metrics} for r in results if r["task"] == task}
    return pd.DataFrame.from_dict(rows, orient="index", columns=metrics)


def plot_roc_pr(y_true, scores, title: str = "", path: Path | str | None = None):
    """ROC and precision-recall curves side by side."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax_roc, plot_chance_level=True)
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax_pr, plot_chance_level=True)
    ax_roc.set_title("ROC")
    ax_pr.set_title("Precision-recall")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_confusion(cm, labels: list[str], title: str = "", normalize: bool = True,
                   path: Path | str | None = None):
    """Confusion matrix, rows normalized to recall by default."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    size = max(4.5, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f" if normalize else ".0f", colorbar=False,
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
