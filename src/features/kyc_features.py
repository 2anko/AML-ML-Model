import numpy as np
import pandas as pd

TODAY = pd.Timestamp("today").normalize()

# Flag suspicious income for low-employment occupations.
# $50k+ while unemployed or a student warrants scrutiny.
_HIGH_INCOME_THRESHOLD = 50_000
_LOW_EMPLOYMENT_OCCUPATIONS = {"UNEMPLOYED", "STUDENT"}

# Shell company indicator: business onboarded within this many days of founding.
_NEW_BUSINESS_ONBOARD_DAYS = 90


def build_kyc(kyc_indiv_path, kyc_sb_path, occ_codes_path=None, ind_codes_path=None):
    ind = pd.read_csv(kyc_indiv_path)
    sb = pd.read_csv(kyc_sb_path)

    ind["customer_type"] = "individual"
    sb["customer_type"] = "smallbusiness"

    # ── Individual features ───────────────────────────────────────────────────
    ind["birth_date"] = pd.to_datetime(ind["birth_date"], errors="coerce")
    ind["onboard_date"] = pd.to_datetime(ind["onboard_date"], errors="coerce")
    ind["age_years"] = (TODAY - ind["birth_date"]).dt.days / 365.25
    ind["account_tenure_days"] = (TODAY - ind["onboard_date"]).dt.days
    ind["is_individual"] = 1

    # Suspicious if unemployed/student but reports high income
    occ_upper = ind["occupation_code"].astype("string").str.upper().str.strip()
    ind["income_occupation_flag"] = (
        occ_upper.isin(_LOW_EMPLOYMENT_OCCUPATIONS) & (ind["income"] > _HIGH_INCOME_THRESHOLD)
    ).astype(int)

    # ── Small business features ───────────────────────────────────────────────
    sb["established_date"] = pd.to_datetime(sb["established_date"], errors="coerce")
    sb["onboard_date"] = pd.to_datetime(sb["onboard_date"], errors="coerce")
    sb["account_tenure_days"] = (TODAY - sb["onboard_date"]).dt.days
    sb["is_individual"] = 0

    # How old was the business when it opened its account (shell company signal)
    sb["days_established_to_onboard"] = (
        sb["onboard_date"] - sb["established_date"]
    ).dt.days
    sb["business_age_days"] = (TODAY - sb["established_date"]).dt.days

    # Revenue efficiency — extremely high or zero revenue per headcount is suspicious
    sb["revenue_per_employee"] = (
        sb["sales"] / sb["employee_count"].replace(0, np.nan)
    ).fillna(0)

    # ── Reference code lookups ────────────────────────────────────────────────
    if occ_codes_path:
        occ = pd.read_csv(occ_codes_path, dtype={"occupation_code": "string"})
        ind["occupation_code"] = ind["occupation_code"].astype("string").str.strip()
        occ["occupation_code"] = occ["occupation_code"].astype("string").str.strip()
        ind = ind.merge(occ, on="occupation_code", how="left")

    if ind_codes_path:
        indcodes = pd.read_csv(ind_codes_path, dtype={"industry_code": "string"})
        sb["industry_code"] = sb["industry_code"].astype("string").str.strip()
        indcodes["industry_code"] = indcodes["industry_code"].astype("string").str.strip()
        sb = sb.merge(indcodes, on="industry_code", how="left")

    kyc = pd.concat([ind, sb], ignore_index=True, sort=False).set_index("customer_id")
    return kyc
