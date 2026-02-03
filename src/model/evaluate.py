from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import joblib

from src.data.paths import ROOT, DATA_PROCESSED


SPLIT_DIR = DATA_PROCESSED / "splits"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports" / "metrics"


def rank_of_positives(y_true, scores):
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)  # 1 = best
    return ranks[y_true == 1].tolist()


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return float("nan")
    k = min(int(k), len(y_true))
    order = np.argsort(-y_score)
    topk = order[:k]
    return float(y_true[topk].sum() / total_pos)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    k = min(int(k), len(y_true))
    order = np.argsort(-y_score)
    topk = order[:k]
    return float(y_true[topk].mean()) if k > 0 else float("nan")


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def load_threshold(model_name: str) -> float:
    meta_path = MODEL_DIR / f"{model_name}.meta.json"
    if not meta_path.exists():
        return 0.5
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return float(meta.get("best_threshold", 0.5))


def main(model_name: str = "logreg_baseline", split: str = "test"):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    pipe = joblib.load(model_path)
    thr = load_threshold(model_name)

    df = load_split(split)
    if "label" not in df.columns:
        raise KeyError("Expected a 'label' column in split parquet.")

    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"])

    scores = pipe.predict_proba(X)[:, 1]
    preds = (scores >= thr).astype(int)

    pr_auc = float(average_precision_score(y, scores))
    roc_auc = float(roc_auc_score(y, scores))
    cm = confusion_matrix(y, preds)

    out = {
        "model_name": model_name,
        "split": split,
        "n": int(len(df)),
        "pos_rate": float(np.mean(y)),
        "threshold_used": float(thr),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_10": recall_at_k(y, scores, 10),
        "recall_at_50": recall_at_k(y, scores, 50),
        "recall_at_100": recall_at_k(y, scores, 100),
        "recall_at_200": recall_at_k(y, scores, 200),
        "precision_at_10": precision_at_k(y, scores, 10),
        "precision_at_50": precision_at_k(y, scores, 50),
        "precision_at_100": precision_at_k(y, scores, 100),
        "precision_at_200": precision_at_k(y, scores, 200),
        "positive_rank": rank_of_positives(y, scores),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(y, preds, output_dict=True),
    }

    out_path = METRICS_DIR / f"{model_name}.{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== EVALUATION ===")
    print(f"Saved: {out_path}")
    print(f"{split} PR-AUC: {pr_auc:.4f}")
    print(f"{split} ROC-AUC: {roc_auc:.4f}")
    print(f"Threshold used: {thr:.4f}")
    print("Confusion matrix [ [tn fp], [fn tp] ]:")
    print(cm)
    print("\nRecall@50/100/200:",
          f"{out['recall_at_50']:.3f} / {out['recall_at_100']:.3f} / {out['recall_at_200']:.3f}")


if __name__ == "__main__":
    main()
