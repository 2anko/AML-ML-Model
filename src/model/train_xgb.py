"""
XGBoost + SMOTE training script.

Key differences from the logistic regression baseline:
  - SMOTE oversamples the minority (fraud) class in the training set before fitting,
    giving the model more examples to learn from.
  - XGBoost captures non-linear feature interactions (e.g. wire transfers AND student
    occupation) that logistic regression cannot represent.
  - scale_pos_weight provides a second layer of imbalance handling on top of SMOTE.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data.paths import DATA_PROCESSED, ROOT
from src.model.train_baseline import (
    infer_column_types,
    load_split,
    recall_at_k,
    precision_at_k,
    threshold_for_top_k,
)

SPLIT_DIR = DATA_PROCESSED / "splits"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports" / "metrics"


@dataclass
class XGBRunSummary:
    model_name: str
    timestamp_utc: str
    n_train: int
    n_valid: int
    n_train_after_smote: int
    pos_rate_train: float
    pos_rate_valid: float
    n_features_input: int
    n_numeric_cols: int
    n_categorical_cols: int
    dropped_datetime_cols: list[str]
    valid_pr_auc: float
    valid_roc_auc: float
    best_threshold: float
    recall_at_50: float
    recall_at_100: float
    recall_at_200: float


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main(model_name: str = "xgb_smote", seed: int = 42):
    ensure_dirs()

    train_df = load_split("train")
    valid_df = load_split("valid")

    y_train = train_df["label"].astype(int).to_numpy()
    y_valid = valid_df["label"].astype(int).to_numpy()

    X_train_df = train_df.drop(columns=["label"])
    X_valid_df = valid_df.drop(columns=["label"])

    # Drop datetime columns (not usable directly by XGBoost)
    numeric_cols, cat_cols, dt_cols = infer_column_types(X_train_df)
    if dt_cols:
        X_train_df = X_train_df.drop(columns=dt_cols)
        X_valid_df = X_valid_df.drop(columns=dt_cols)
        numeric_cols, cat_cols, _ = infer_column_types(X_train_df)

    print(f"Input features:  {X_train_df.shape[1]}  ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"Training labels: {y_train.sum()} fraud / {len(y_train) - y_train.sum()} non-fraud")

    # Preprocessing: numeric imputation only (no scaling needed for trees),
    # categorical one-hot encoding with unknown handling.
    numeric_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    # scale_pos_weight = ratio of negatives to positives in the original training set.
    # Combined with SMOTE this gives two complementary layers of imbalance handling.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    # SMOTE k_neighbors must be < number of minority samples
    k_neighbors = min(5, n_pos - 1)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    # imblearn Pipeline supports resamplers between preprocessing and the model.
    # SMOTE is applied only during fit() — predict() bypasses it automatically.
    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=seed, k_neighbors=k_neighbors)),
        ("model", clf),
    ])

    pipe.fit(X_train_df, y_train)

    # How many samples after SMOTE?
    # SMOTE targets 1:1 class balance by default
    n_after_smote = n_neg * 2
    print(f"Training samples after SMOTE: ~{n_after_smote}  ({n_neg} non-fraud + {n_neg} synthetic fraud)")

    # Feature importances
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop 20 features by XGBoost importance (gain):")
    for i in top_idx:
        if importances[i] > 0:
            print(f"  {feature_names[i]:45s}  {importances[i]:.4f}")

    # Evaluate on validation set
    valid_scores = pipe.predict_proba(X_valid_df)[:, 1]
    pr_auc = float(average_precision_score(y_valid, valid_scores))
    roc_auc = float(roc_auc_score(y_valid, valid_scores))

    K = 10
    thr = threshold_for_top_k(valid_scores, K)

    summary = XGBRunSummary(
        model_name=model_name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_train=int(len(train_df)),
        n_valid=int(len(valid_df)),
        n_train_after_smote=n_after_smote,
        pos_rate_train=float(np.mean(y_train)),
        pos_rate_valid=float(np.mean(y_valid)),
        n_features_input=int(X_train_df.shape[1]),
        n_numeric_cols=int(len(numeric_cols)),
        n_categorical_cols=int(len(cat_cols)),
        dropped_datetime_cols=list(dt_cols),
        valid_pr_auc=pr_auc,
        valid_roc_auc=roc_auc,
        best_threshold=float(thr),
        recall_at_50=float(recall_at_k(y_valid, valid_scores, 50)),
        recall_at_100=float(recall_at_k(y_valid, valid_scores, 100)),
        recall_at_200=float(recall_at_k(y_valid, valid_scores, 200)),
    )

    model_path = MODEL_DIR / f"{model_name}.joblib"
    meta_path = MODEL_DIR / f"{model_name}.meta.json"
    metrics_path = METRICS_DIR / f"{model_name}.valid.json"

    joblib.dump(pipe, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\n=== TRAINED XGB+SMOTE ===")
    print(f"Model saved:    {model_path}")
    print(f"Valid PR-AUC:   {summary.valid_pr_auc:.4f}")
    print(f"Valid ROC-AUC:  {summary.valid_roc_auc:.4f}")
    print(f"Recall@50/100/200: {summary.recall_at_50:.3f} / {summary.recall_at_100:.3f} / {summary.recall_at_200:.3f}")


if __name__ == "__main__":
    main()
