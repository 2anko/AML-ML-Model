from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from src.data.paths import ROOT, DATA_PROCESSED


SPLIT_DIR = DATA_PROCESSED / "splits"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports" / "metrics"


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@K: among all true positives, how many are in top K scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    total_pos = int(y_true.sum())
    if total_pos == 0:
        return float("nan")

    k = min(int(k), len(y_true))
    order = np.argsort(-y_score)  # descending
    topk = order[:k]
    hits = int(y_true[topk].sum())
    return hits / total_pos


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    k = min(int(k), len(y_true))
    order = np.argsort(-y_score)
    topk = order[:k]
    return float(y_true[topk].mean()) if k > 0 else float("nan")


def threshold_for_top_k(scores: np.ndarray, k: int) -> float:
    k = min(int(k), len(scores))
    # kth largest score
    return float(np.sort(scores)[-k])


@dataclass
class TrainRunSummary:
    model_name: str
    timestamp_utc: str
    n_train: int
    n_valid: int
    pos_rate_train: float
    pos_rate_valid: float
    n_features_input: int
    n_numeric_cols: int
    n_categorical_cols: int
    dropped_datetime_cols: list[str]
    valid_pr_auc: float
    valid_roc_auc: float
    best_threshold: float
    best_f1_valid: float
    recall_at_50: float
    recall_at_100: float
    recall_at_200: float


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Returns: (numeric_cols, categorical_cols, datetime_cols)
    """
    datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]

    numeric_cols = []
    categorical_cols = []
    for c in df.columns:
        if c in datetime_cols:
            continue
        if is_bool_dtype(df[c]) or is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols, datetime_cols


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main(model_name: str = "logreg_baseline", seed: int = 42):
    ensure_dirs()

    # Load your prebuilt splits
    train_df = load_split("train")
    valid_df = load_split("valid")

    if "label" not in train_df.columns:
        raise KeyError("Expected a 'label' column in train split.")
    if "label" not in valid_df.columns:
        raise KeyError("Expected a 'label' column in valid split.")

    y_train = train_df["label"].astype(int).to_numpy()
    y_valid = valid_df["label"].astype(int).to_numpy()

    X_train_df = train_df.drop(columns=["label"])
    X_valid_df = valid_df.drop(columns=["label"])

    # Drop datetime columns (baseline approach)
    numeric_cols, cat_cols, dt_cols = infer_column_types(X_train_df)
    if dt_cols:
        X_train_df = X_train_df.drop(columns=dt_cols)
        X_valid_df = X_valid_df.drop(columns=dt_cols)
        # re-infer after drop
        numeric_cols, cat_cols, _ = infer_column_types(X_train_df)

    # Preprocess
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
        random_state=seed,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    # Fit
    pipe.fit(X_train_df, y_train)

    # Validate
    valid_scores = pipe.predict_proba(X_valid_df)[:, 1]
    pr_auc = float(average_precision_score(y_valid, valid_scores))
    roc_auc = float(roc_auc_score(y_valid, valid_scores))

    K = 10  # try 10 and 20
    thr = threshold_for_top_k(valid_scores, K)
    best_f1 = float("nan")

    summary = TrainRunSummary(
        model_name=model_name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_train=int(len(train_df)),
        n_valid=int(len(valid_df)),
        pos_rate_train=float(np.mean(y_train)),
        pos_rate_valid=float(np.mean(y_valid)),
        n_features_input=int(X_train_df.shape[1]),
        n_numeric_cols=int(len(numeric_cols)),
        n_categorical_cols=int(len(cat_cols)),
        dropped_datetime_cols=list(dt_cols),
        valid_pr_auc=pr_auc,
        valid_roc_auc=roc_auc,
        best_threshold=float(thr),
        best_f1_valid=float(best_f1),
        recall_at_50=float(recall_at_k(y_valid, valid_scores, 50)),
        recall_at_100=float(recall_at_k(y_valid, valid_scores, 100)),
        recall_at_200=float(recall_at_k(y_valid, valid_scores, 200)),
    )

    # Save model + metadata
    model_path = MODEL_DIR / f"{model_name}.joblib"
    meta_path = MODEL_DIR / f"{model_name}.meta.json"
    metrics_path = METRICS_DIR / f"{model_name}.valid.json"

    joblib.dump(pipe, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print("\n=== TRAINED BASELINE ===")
    print(f"Model saved:   {model_path}")
    print(f"Metadata saved:{meta_path}")
    print(f"Valid metrics: {metrics_path}")
    print(f"Valid PR-AUC:  {summary.valid_pr_auc:.4f}")
    print(f"Valid ROC-AUC: {summary.valid_roc_auc:.4f}")
    print(f"Best thr (F1): {summary.best_threshold:.4f}  (F1={summary.best_f1_valid:.4f})")
    print(f"Recall@50/100/200: {summary.recall_at_50:.3f} / {summary.recall_at_100:.3f} / {summary.recall_at_200:.3f}")


if __name__ == "__main__":
    main()
