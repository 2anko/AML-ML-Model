import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype
from src.data.paths import (
    RAW_LABELS, RAW_KYC_INDIV, RAW_KYC_SB,
    RAW_EFT, RAW_EMT, RAW_WIRE, RAW_WU, RAW_CHQ,
    RAW_ABM, RAW_CARD,
    REF_OCC, REF_IND,
    OUT_JOINED, OUT_CLEANED, OUT_FEATURES
)
from src.features.kyc_features import build_kyc
from src.features.tx_aggregates import (
    agg_basic_channel_csv, agg_abm_csv, agg_card_csv
)


def main():
    OUT_JOINED.mkdir(parents=True, exist_ok=True)
    OUT_CLEANED.mkdir(parents=True, exist_ok=True)
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(RAW_LABELS).set_index("customer_id")  # label anchor
    customer_set = set(labels.index)

    kyc = build_kyc(RAW_KYC_INDIV, RAW_KYC_SB, REF_OCC, REF_IND)
    kyc.to_parquet(OUT_CLEANED / "kyc.parquet")

    # Per-channel aggregates
    eft = agg_basic_channel_csv(RAW_EFT, customer_set, "eft")
    emt = agg_basic_channel_csv(RAW_EMT, customer_set, "emt")
    wire = agg_basic_channel_csv(RAW_WIRE, customer_set, "wire")
    wu = agg_basic_channel_csv(RAW_WU, customer_set, "wu")
    chq = agg_basic_channel_csv(RAW_CHQ, customer_set, "cheque")
    abm = agg_abm_csv(RAW_ABM, customer_set)
    card = agg_card_csv(RAW_CARD, customer_set)

    # Save intermediates (optional but helpful)
    eft.to_parquet(OUT_FEATURES / "tx_agg_eft.parquet")
    emt.to_parquet(OUT_FEATURES / "tx_agg_emt.parquet")
    wire.to_parquet(OUT_FEATURES / "tx_agg_wire.parquet")
    wu.to_parquet(OUT_FEATURES / "tx_agg_wu.parquet")
    chq.to_parquet(OUT_FEATURES / "tx_agg_cheque.parquet")
    abm.to_parquet(OUT_FEATURES / "tx_agg_abm.parquet")
    card.to_parquet(OUT_FEATURES / "tx_agg_card.parquet")

    # Join into final train table
    table = labels.join(kyc, how="left")
    for df in [eft, emt, wire, wu, chq, abm, card]:
        table = table.join(df, how="left")

    # Fill missing: no transactions in a channel => 0
    for code_col in ["occupation_code", "industry_code"]:
        if code_col in table.columns:
            table[code_col] = table[code_col].astype("string")

    for col in table.columns:
        if is_numeric_dtype(table[col]) or is_bool_dtype(table[col]):
            table[col] = table[col].fillna(0)
        elif is_datetime64_any_dtype(table[col]):
            # keep NaT (don’t fill dates with 0)
            pass
        else:
            # categorical/text columns
            table[col] = table[col].astype("string").fillna("unknown")

    out_path = OUT_JOINED / "train_table.parquet"
    table.to_parquet(out_path)
    print("Saved:", out_path.resolve())
    print("Shape:", table.shape)


if __name__ == "__main__":
    main()
