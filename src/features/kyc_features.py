import pandas as pd


def build_kyc(kyc_indiv_path, kyc_sb_path, occ_codes_path=None, ind_codes_path=None):
    ind = pd.read_csv(kyc_indiv_path)
    sb = pd.read_csv(kyc_sb_path)

    ind["customer_type"] = "individual"
    sb["customer_type"] = "smallbusiness"

    ind["birth_date"] = pd.to_datetime(ind["birth_date"], errors="coerce")
    ind["onboard_date"] = pd.to_datetime(ind["onboard_date"], errors="coerce")
    ind["age_years"] = (pd.Timestamp("today") - ind["birth_date"]).dt.days / 365.25

    sb["established_date"] = pd.to_datetime(sb["established_date"], errors="coerce")
    sb["onboard_date"] = pd.to_datetime(sb["onboard_date"], errors="coerce")

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
