import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.paths import OUT_JOINED, DATA_PROCESSED

SPLIT_DIR = DATA_PROCESSED / "splits"

def main(seed=42):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OUT_JOINED / "train_table.parquet")  # index = customer_id
    y = df["label"].astype(int)

    ids = df.index.to_series()

    train_ids, temp_ids = train_test_split(
        ids, test_size=0.30, random_state=seed, stratify=y
    )
    y_temp = y.loc[temp_ids]

    valid_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=seed, stratify=y_temp
    )

    train_ids.to_csv(SPLIT_DIR / "train_ids.csv", index=False, header=["customer_id"])
    valid_ids.to_csv(SPLIT_DIR / "valid_ids.csv", index=False, header=["customer_id"])
    test_ids.to_csv(SPLIT_DIR / "test_ids.csv", index=False, header=["customer_id"])

    df.loc[train_ids].to_parquet(SPLIT_DIR / "train.parquet")
    df.loc[valid_ids].to_parquet(SPLIT_DIR / "valid.parquet")
    df.loc[test_ids].to_parquet(SPLIT_DIR / "test.parquet")

    print("Saved splits to:", SPLIT_DIR)

if __name__ == "__main__":
    main()
