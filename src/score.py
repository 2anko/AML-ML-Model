"""
Risk ranking script — scores all customers and outputs a ranked list for investigator review.

Loads the full train_table (all 1000 customers), runs them through the trained model,
and writes a ranked CSV and a summary report.

Usage:
  python -m src.score                                 # use default model (logreg_baseline)
  python -m src.score --model-name xgb_smote          # use a different trained model
  python -m src.score --top 50                        # show top 50 in console summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.paths import ROOT, DATA_PROCESSED, OUT_JOINED

MODEL_DIR   = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SCORES_DIR  = REPORTS_DIR / "scores"


# ── Risk flag rules ────────────────────────────────────────────────────────────
# Each flag is a human-readable note attached to a customer row if the condition
# is met. Investigators see these alongside the score.

def _build_flags(df: pd.DataFrame) -> pd.Series:
    flags = pd.Series([[] for _ in range(len(df))], index=df.index)

    def add(mask: pd.Series, note: str):
        for idx in df.index[mask.fillna(False)]:
            flags[idx].append(note)

    if "income_occupation_flag" in df.columns:
        add(df["income_occupation_flag"] == 1, "high income for occupation")

    if "occupation_code" in df.columns:
        add(df["occupation_code"].astype(str).str.upper() == "UNEMPLOYED", "unemployed")
        add(df["occupation_code"].astype(str).str.upper() == "STUDENT",    "student")

    if "account_tenure_days" in df.columns:
        add(df["account_tenure_days"] < 365, "account < 1 year old")

    if "days_established_to_onboard" in df.columns:
        add(
            (df["days_established_to_onboard"].notna()) &
            (df["days_established_to_onboard"] < 30),
            "onboarded <30 days after business founding",
        )

    if "business_age_days" in df.columns:
        add(
            (df["is_individual"] == 0) & (df["business_age_days"] < 180),
            "business < 6 months old",
        )

    if "revenue_per_employee" in df.columns:
        add(
            (df["is_individual"] == 0) & (df["revenue_per_employee"] == 0) & (df["employee_count"] > 0),
            "employees but zero revenue",
        )
        # Implausibly high revenue per employee is also suspicious (e.g. $1M with 1 staff).
        # Use the 95th percentile of the business population as the threshold.
        biz_mask = df["is_individual"] == 0
        if biz_mask.sum() > 0:
            rev_p95 = df.loc[biz_mask, "revenue_per_employee"].quantile(0.95)
            add(
                biz_mask & (df["revenue_per_employee"] > rev_p95) & (df["employee_count"] > 0),
                f"implausibly high revenue/employee (>${rev_p95:,.0f})",
            )

    if "wire_count" in df.columns:
        add(df["wire_count"] > 0, "has wire transfers")

    if "emt_amt_sum" in df.columns:
        emt_p90 = df["emt_amt_sum"].quantile(0.90)
        add(df["emt_amt_sum"] > emt_p90, "high EMT volume (top 10%)")

    if "eft_debit_sum" in df.columns:
        eft_p90 = df["eft_debit_sum"].quantile(0.90)
        add(df["eft_debit_sum"] > eft_p90, "high EFT outflows (top 10%)")

    if "card_foreign_rate" in df.columns:
        add(df["card_foreign_rate"] > 0.3, "30%+ foreign card transactions")

    if "cheque_count" in df.columns:
        cheque_p90 = df["cheque_count"].quantile(0.90)
        add(df["cheque_count"] > cheque_p90, "high cheque activity (top 10%)")

    return flags.apply(lambda x: "; ".join(x) if x else "")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(model_name: str = "logreg_baseline", top_n: int = 20):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `python -m src.train --stages train` first."
        )

    table_path = OUT_JOINED / "train_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {table_path}\n"
            "Run `python -m src.train --stages build` first."
        )

    print(f"Loading model: {model_name}")
    pipe = joblib.load(model_path)

    print(f"Loading feature table: {table_path}")
    table = pd.read_parquet(table_path)

    known_labels = table["label"].astype(int) if "label" in table.columns else None
    X = table.drop(columns=["label"], errors="ignore")

    print(f"Scoring {len(X)} customers ...")
    scores = pipe.predict_proba(X)[:, 1]

    # Build ranked output
    result = pd.DataFrame({"customer_id": X.index, "risk_score": scores})

    # Attach readable context columns
    context_cols = [
        "customer_type", "country", "province", "city",
        "gender", "marital_status", "occupation_code", "occupation_title",
        "income", "age_years", "account_tenure_days",
        "industry_code", "industry", "employee_count", "sales",
        "business_age_days", "days_established_to_onboard", "revenue_per_employee",
        "income_occupation_flag",
        "is_individual",
        "emt_amt_sum", "wire_count", "wire_amt_sum",
        "eft_debit_sum", "cheque_count", "card_foreign_rate",
    ]
    for col in context_cols:
        if col in table.columns:
            result[col] = table[col].values

    # Attach flags
    result["risk_flags"] = _build_flags(table).values

    # Attach true label if available
    if known_labels is not None:
        result["known_fraud"] = known_labels.values

    # Sort by score descending
    result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    # Save full ranked list
    out_csv  = SCORES_DIR / f"{model_name}.ranked_customers.csv"
    out_json = SCORES_DIR / f"{model_name}.score_summary.json"
    result.to_csv(out_csv, index=False)
    print(f"\nFull ranked list saved: {out_csv}")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RISK RANKING SUMMARY — {model_name}  ({len(result)} customers)")
    print(f"{'='*70}")

    top = result.head(top_n)
    display_cols = ["rank", "customer_id", "risk_score", "customer_type"]
    if "known_fraud" in result.columns:
        display_cols.append("known_fraud")
    display_cols.append("risk_flags")

    # Format for readability
    display = top[display_cols].copy()
    display["risk_score"] = display["risk_score"].map("{:.4f}".format)
    if "known_fraud" in display.columns:
        display["known_fraud"] = display["known_fraud"].map({1: "YES ◀", 0: ""})

    print(display.to_string(index=False))

    # ── Ranking quality (if labels available) ─────────────────────────────────
    if known_labels is not None:
        y = result["known_fraud"].to_numpy()
        s = result["risk_score"].to_numpy()
        n_fraud = int(y.sum())

        print(f"\n  Known fraud cases: {n_fraud}")
        print(f"  {'Review top N':20s}  {'Fraud found':>12}  {'Recall':>8}")
        print("  " + "-" * 44)
        for k in [10, 20, 50, 100, 200]:
            if k > len(result):
                continue
            found = int(y[:k].sum())
            recall = found / n_fraud if n_fraud else 0
            bar = "█" * found
            print(f"  {'Top ' + str(k):20s}  {found:>5} / {n_fraud:<5}   {recall:>6.1%}  {bar}")

        print(f"\n  Ranks of known fraud cases:")
        fraud_rows = result[result["known_fraud"] == 1][["rank", "customer_id", "risk_score", "risk_flags"]]
        print(fraud_rows.to_string(index=False))

        # Save summary JSON
        summary = {
            "model_name": model_name,
            "n_customers": len(result),
            "n_known_fraud": n_fraud,
            "recall_at_10":  int(y[:10].sum())  / n_fraud if n_fraud else None,
            "recall_at_20":  int(y[:20].sum())  / n_fraud if n_fraud else None,
            "recall_at_50":  int(y[:50].sum())  / n_fraud if n_fraud else None,
            "recall_at_100": int(y[:100].sum()) / n_fraud if n_fraud else None,
            "fraud_ranks": fraud_rows["rank"].tolist(),
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Score summary saved: {out_json}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all customers and output a risk ranking")
    parser.add_argument(
        "--model-name",
        default="logreg_baseline",
        help="Name of the trained model artifact to use (default: logreg_baseline)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top customers to show in the console summary (default: 20)",
    )
    args = parser.parse_args()
    main(model_name=args.model_name, top_n=args.top)
