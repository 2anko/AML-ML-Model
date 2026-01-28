from pathlib import Path

# Project root = folder that contains this file's grandparent (src/ -> root)
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "Data" / "raw"
DATA_PROCESSED = ROOT / "Data" / "Processed"

RAW_LABELS = DATA_RAW / "labels" / "labels.csv"

RAW_KYC_INDIV = DATA_RAW / "kyc" / "kyc_individual.csv"
RAW_KYC_SB    = DATA_RAW / "kyc" / "kyc_smallbusiness.csv"

RAW_TX_DIR = DATA_RAW / "transactions"
RAW_CARD = RAW_TX_DIR / "card.csv"
RAW_ABM  = RAW_TX_DIR / "abm.csv"
RAW_EFT  = RAW_TX_DIR / "eft.csv"
RAW_EMT  = RAW_TX_DIR / "emt.csv"
RAW_WIRE = RAW_TX_DIR / "wire.csv"
RAW_WU   = RAW_TX_DIR / "westernunion.csv"
RAW_CHQ  = RAW_TX_DIR / "cheque.csv"

REF_DIR = DATA_RAW / "reference"
REF_OCC = REF_DIR / "kyc_occupation_codes.csv"
REF_IND = REF_DIR / "kyc_industry_codes.csv"

OUT_JOINED = DATA_PROCESSED / "joined"
OUT_FEATURES = DATA_PROCESSED / "features"
OUT_CLEANED = DATA_PROCESSED / "cleaned"