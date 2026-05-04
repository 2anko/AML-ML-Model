"""
Diagnostic script — answers three questions before investing in model work:

  Q1. Do any features separate fraud from non-fraud? (univariate signal)
  Q2. Is the join working? (do fraud rows have transaction data?)
  Q3. Is data volume the limiting factor? (label counts end-to-end)

Usage:
  python -m src.diagnostics
"""

import textwrap
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from src.data.paths import RAW_LABELS, OUT_JOINED, DATA_PROCESSED

warnings.filterwarnings("ignore")

SPLIT_DIR = DATA_PROCESSED / "splits"

SEP = "=" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── Q3 helpers ────────────────────────────────────────────────────────────────

def audit_label_counts(train_table: pd.DataFrame):
    section("Q3 · DATA VOLUME — label counts end-to-end")

    raw_labels = pd.read_csv(RAW_LABELS)
    n_raw = len(raw_labels)
    n_raw_pos = int(raw_labels["label"].sum())
    n_raw_neg = n_raw - n_raw_pos

    n_joined = len(train_table)
    n_joined_pos = int(train_table["label"].sum())
    n_joined_neg = n_joined - n_joined_pos

    print(f"{'':30s} {'Total':>8}  {'Fraud':>8}  {'Non-fraud':>10}  {'Fraud %':>8}")
    print("-" * 70)
    print(f"{'Raw labels.csv':30s} {n_raw:>8}  {n_raw_pos:>8}  {n_raw_neg:>10}  {n_raw_pos/n_raw*100:>7.2f}%")
    print(f"{'After join (train_table)':30s} {n_joined:>8}  {n_joined_pos:>8}  {n_joined_neg:>10}  {n_joined_pos/n_joined*100:>7.2f}%")

    dropped = n_raw - n_joined
    if dropped:
        print(f"\n  ⚠  {dropped} customers dropped during the join.")
    else:
        print("\n  ✓  No customers dropped during the join.")

    if n_joined_pos < 20:
        print(f"\n  ⚠  Only {n_joined_pos} fraud cases total — this is very few for ML.")
        print("     Consider: getting more labeled data, or synthetic oversampling (SMOTE).")

    for split_name in ("train", "valid", "test"):
        path = SPLIT_DIR / f"{split_name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            pos = int(df["label"].sum())
            print(f"  {split_name:6s} split: {len(df):4d} customers, {pos} fraud cases ({pos/len(df)*100:.2f}%)")


# ── Q2 helpers ────────────────────────────────────────────────────────────────

def audit_fraud_rows(train_table: pd.DataFrame):
    section("Q2 · JOIN QUALITY — do fraud rows have transaction data?")

    fraud = train_table[train_table["label"] == 1]
    legit = train_table[train_table["label"] == 0]

    print(f"  Fraud rows: {len(fraud)}    Non-fraud rows: {len(legit)}\n")

    tx_count_cols = [c for c in train_table.columns if c.endswith("_count")]
    kyc_cols = ["income", "age_years", "employee_count", "sales"]
    kyc_cols = [c for c in kyc_cols if c in train_table.columns]

    # Check: how many fraud rows have ALL-zero transaction counts
    if tx_count_cols:
        fraud_all_zero_tx = (fraud[tx_count_cols] == 0).all(axis=1).sum()
        legit_all_zero_tx = (legit[tx_count_cols] == 0).all(axis=1).sum()
        print(f"  Customers with zero transactions across ALL channels:")
        print(f"    Fraud:     {fraud_all_zero_tx} / {len(fraud)}  ({fraud_all_zero_tx/len(fraud)*100:.0f}%)")
        print(f"    Non-fraud: {legit_all_zero_tx} / {len(legit)}  ({legit_all_zero_tx/len(legit)*100:.0f}%)")

        if fraud_all_zero_tx > len(fraud) * 0.5:
            print("\n  ⚠  More than half of fraud rows have no transaction data.")
            print("     Likely a customer_id mismatch between labels and transaction files.")
        else:
            print("\n  ✓  Fraud rows have transaction data (join looks correct).")

    # KYC coverage
    if kyc_cols:
        print(f"\n  KYC field null rates (fraud vs non-fraud):")
        for col in kyc_cols:
            f_null = fraud[col].isna().mean() if col in fraud else float("nan")
            l_null = legit[col].isna().mean() if col in legit else float("nan")
            print(f"    {col:20s}  fraud null: {f_null*100:5.1f}%   legit null: {l_null*100:5.1f}%")

    # Per-channel transaction totals for fraud customers
    print(f"\n  Mean transaction counts per channel (fraud vs non-fraud):")
    channels = ["eft", "emt", "wire", "wu", "cheque", "abm", "card"]
    print(f"  {'Channel':10s}  {'Fraud mean':>12}  {'Non-fraud mean':>14}  {'Ratio':>8}")
    print("  " + "-" * 52)
    for ch in channels:
        col = f"{ch}_count"
        if col not in train_table.columns:
            continue
        f_mean = fraud[col].mean()
        l_mean = legit[col].mean()
        ratio = f_mean / l_mean if l_mean > 0 else float("nan")
        flag = "  ◀ signal" if abs(ratio - 1) > 0.5 and not np.isnan(ratio) else ""
        print(f"  {ch:10s}  {f_mean:>12.2f}  {l_mean:>14.2f}  {ratio:>8.2f}{flag}")


# ── Q1 helpers ────────────────────────────────────────────────────────────────

def univariate_signal(train_table: pd.DataFrame, top_n: int = 20):
    section("Q1 · FEATURE SIGNAL — univariate AUC per feature")

    y = train_table["label"].astype(int).to_numpy()
    feature_df = train_table.drop(columns=["label"])

    results = []

    for col in feature_df.columns:
        series = feature_df[col]

        if is_datetime64_any_dtype(series):
            continue

        if is_numeric_dtype(series):
            vals = series.fillna(0).to_numpy().astype(float)
        else:
            # encode categorical as frequency of each category
            freq = series.map(series.value_counts(normalize=True))
            vals = freq.fillna(0).to_numpy().astype(float)

        if vals.std() == 0:
            results.append((col, 0.5, "constant"))
            continue

        try:
            auc = roc_auc_score(y, vals)
            # flip if inverted signal (AUC < 0.5 means negative correlation)
            direction = "pos" if auc >= 0.5 else "neg"
            auc = max(auc, 1 - auc)
            results.append((col, auc, direction))
        except Exception:
            results.append((col, float("nan"), "error"))

    results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)

    print(f"  {'Feature':40s}  {'AUC':>6}  Direction")
    print("  " + "-" * 58)

    has_signal = [r for r in results if r[1] > 0.55]
    no_signal  = [r for r in results if r[1] <= 0.55]

    if has_signal:
        print(f"\n  Features WITH signal (AUC > 0.55):  {len(has_signal)} found\n")
        for col, auc, direction in has_signal[:top_n]:
            bar = "█" * int((auc - 0.5) * 40)
            print(f"  {col:40s}  {auc:.3f}  {direction:3s}  {bar}")
    else:
        print("\n  ⚠  No features with AUC > 0.55 found.")
        print("     This suggests the current features carry little signal for fraud detection.")

    print(f"\n  Features with NO signal (AUC ≤ 0.55): {len(no_signal)}")

    # Mean comparison table for numeric features with signal
    if has_signal:
        section("Q1b · MEAN COMPARISON — fraud vs non-fraud for top signal features")
        fraud = train_table[train_table["label"] == 1]
        legit = train_table[train_table["label"] == 0]

        sig_numeric = [
            (col, auc, d) for col, auc, d in has_signal
            if is_numeric_dtype(train_table[col])
        ]

        if sig_numeric:
            print(f"\n  {'Feature':40s}  {'Fraud mean':>12}  {'Non-fraud mean':>14}  {'AUC':>6}")
            print("  " + "-" * 78)
            for col, auc, _ in sig_numeric[:top_n]:
                f_mean = fraud[col].mean()
                l_mean = legit[col].mean()
                print(f"  {col:40s}  {f_mean:>12.2f}  {l_mean:>14.2f}  {auc:.3f}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(train_table: pd.DataFrame):
    section("SUMMARY & RECOMMENDED NEXT STEPS")

    y = train_table["label"].astype(int)
    n_pos = int(y.sum())
    n_total = len(train_table)

    feature_df = train_table.drop(columns=["label"])
    results = []
    for col in feature_df.columns:
        series = feature_df[col]
        if is_datetime64_any_dtype(series):
            continue
        if is_numeric_dtype(series):
            vals = series.fillna(0).to_numpy().astype(float)
        else:
            freq = series.map(series.value_counts(normalize=True))
            vals = freq.fillna(0).to_numpy().astype(float)
        if vals.std() == 0:
            continue
        try:
            auc = roc_auc_score(y, vals)
            results.append(max(auc, 1 - auc))
        except Exception:
            pass

    n_signal = sum(1 for a in results if a > 0.55)
    max_auc = max(results) if results else 0.5

    print(f"\n  Dataset:  {n_total} customers,  {n_pos} fraud ({n_pos/n_total*100:.2f}%)")
    print(f"  Best single-feature AUC: {max_auc:.3f}")
    print(f"  Features with AUC > 0.55: {n_signal}")

    print("\n  Diagnosis:")
    if n_pos < 20:
        print("  [!] Very few fraud cases — model metrics will be unreliable.")
        print("      → Get more labeled data, or apply SMOTE before training.")
    if n_signal == 0:
        print("  [!] No feature shows univariate signal.")
        print("      → Check the join (Q2), revisit feature definitions, or check label quality.")
    elif n_signal < 5:
        print("  [~] Weak signal — a few features discriminate but most don't.")
        print("      → Focus feature engineering on the top-signal channels.")
    else:
        print("  [✓] Signal exists. Model underperformance is likely due to data volume or tuning.")
        print("      → Try SMOTE, XGBoost, and threshold optimization.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    table_path = OUT_JOINED / "train_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(
            f"train_table.parquet not found at {table_path}\n"
            "Run `python -m src.train --stages build` first."
        )

    print(f"\nLoading {table_path} ...")
    train_table = pd.read_parquet(table_path)

    audit_label_counts(train_table)
    audit_fraud_rows(train_table)
    univariate_signal(train_table)
    print_summary(train_table)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
