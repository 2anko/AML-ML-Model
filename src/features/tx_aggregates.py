import pandas as pd
import numpy as np


def agg_basic_channel_csv(path, customer_set, prefix, chunksize=750_000):
    usecols = ["customer_id", "amount_cad", "debit_credit", "transaction_datetime"]
    parts = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["customer_id"].isin(customer_set)]
        if chunk.empty:
            continue

        chunk["transaction_datetime"] = pd.to_datetime(chunk["transaction_datetime"], errors="coerce")
        chunk["tx_date"] = chunk["transaction_datetime"].dt.date

        g = chunk.groupby("customer_id").agg(
            count=("amount_cad", "size"),
            amt_sum=("amount_cad", "sum"),
            amt_mean=("amount_cad", "mean"),
            amt_max=("amount_cad", "max"),
            active_days=("tx_date", pd.Series.nunique),
        )

        credit = chunk.loc[chunk["debit_credit"] == "C"].groupby("customer_id")["amount_cad"].sum()
        debit = chunk.loc[chunk["debit_credit"] == "D"].groupby("customer_id")["amount_cad"].sum()

        g["credit_sum"] = credit
        g["debit_sum"] = debit
        g = g.fillna(0)

        parts.append(g)

    if not parts:
        out = pd.DataFrame(index=pd.Index([], name="customer_id"))
    else:
        out = pd.concat(parts).groupby(level=0).sum(numeric_only=True)

    out.columns = [f"{prefix}_{c}" for c in out.columns]
    if not out.empty:
        out[f"{prefix}_net_flow"] = out[f"{prefix}_credit_sum"] - out[f"{prefix}_debit_sum"]
        out[f"{prefix}_avg_per_day"] = out[f"{prefix}_count"] / out[f"{prefix}_active_days"].replace(0, np.nan)
        out = out.fillna(0)

    return out


def agg_abm_csv(path, customer_set, chunksize=750_000):
    usecols = ["customer_id", "amount_cad", "debit_credit", "transaction_datetime",
               "cash_indicator", "province", "city", "country"]
    parts = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["customer_id"].isin(customer_set)]
        if chunk.empty:
            continue

        chunk["cash_indicator"] = chunk["cash_indicator"].fillna(0).astype(int)

        base = agg_basic_channel_df(chunk, "abm")
        extra = chunk.groupby("customer_id").agg(
            abm_cash_count=("cash_indicator", "sum"),
            abm_count_for_rate=("cash_indicator", "size"),
            abm_unique_province=("province", pd.Series.nunique),
            abm_unique_city=("city", pd.Series.nunique),
        )
        parts.append(base.join(extra, how="left"))

    if not parts:
        return pd.DataFrame(index=pd.Index([], name="customer_id"))

    out = pd.concat(parts).groupby(level=0).sum(numeric_only=True)
    out["abm_cash_rate"] = out["abm_cash_count"] / out["abm_count_for_rate"].replace(0, np.nan)
    out = out.drop(columns=["abm_count_for_rate"]).fillna(0)
    return out


def agg_card_csv(path, customer_set, chunksize=750_000):
    usecols = ["customer_id", "amount_cad", "debit_credit", "transaction_datetime",
               "merchant_category", "ecommerce_ind", "country", "province", "city"]
    parts = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["customer_id"].isin(customer_set)]
        if chunk.empty:
            continue

        chunk["ecommerce_ind"] = chunk["ecommerce_ind"].fillna(0).astype(int)
        chunk["is_foreign"] = (chunk["country"] != "CA").astype(int)

        base = agg_basic_channel_df(chunk, "card")
        extra = chunk.groupby("customer_id").agg(
            card_ecom_count=("ecommerce_ind", "sum"),
            card_count_for_rate=("ecommerce_ind", "size"),
            card_foreign_count=("is_foreign", "sum"),
            card_unique_mcc=("merchant_category", pd.Series.nunique),
            card_unique_province=("province", pd.Series.nunique),
            card_unique_city=("city", pd.Series.nunique),
        )
        parts.append(base.join(extra, how="left"))

    if not parts:
        return pd.DataFrame(index=pd.Index([], name="customer_id"))

    out = pd.concat(parts).groupby(level=0).sum(numeric_only=True)
    out["card_ecom_rate"] = out["card_ecom_count"] / out["card_count_for_rate"].replace(0, np.nan)
    out["card_foreign_rate"] = out["card_foreign_count"] / out["card_count_for_rate"].replace(0, np.nan)
    out = out.drop(columns=["card_count_for_rate"]).fillna(0)
    return out


def agg_basic_channel_df(df, prefix):
    df = df.copy()
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
    df["tx_date"] = df["transaction_datetime"].dt.date

    g = df.groupby("customer_id").agg(
        count=("amount_cad", "size"),
        amt_sum=("amount_cad", "sum"),
        amt_mean=("amount_cad", "mean"),
        amt_max=("amount_cad", "max"),
        active_days=("tx_date", pd.Series.nunique),
    )

    credit = df.loc[df["debit_credit"] == "C"].groupby("customer_id")["amount_cad"].sum()
    debit = df.loc[df["debit_credit"] == "D"].groupby("customer_id")["amount_cad"].sum()
    g["credit_sum"] = credit
    g["debit_sum"] = debit
    g = g.fillna(0)

    g.columns = [f"{prefix}_{c}" for c in g.columns]
    g[f"{prefix}_net_flow"] = g[f"{prefix}_credit_sum"] - g[f"{prefix}_debit_sum"]
    g[f"{prefix}_avg_per_day"] = g[f"{prefix}_count"] / g[f"{prefix}_active_days"].replace(0, np.nan)
    return g.fillna(0)
