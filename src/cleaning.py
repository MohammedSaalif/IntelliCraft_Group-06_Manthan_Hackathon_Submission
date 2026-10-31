"""
Phase 1: Data quality, cleaning, and trust.
Reads raw CSV, fixes known issues, logs all changes, and saves clean parquet.
"""

import pandas as pd
import numpy as np
import yaml
import re
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load config.yaml from project root."""
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def clean_raw_data(raw_path=None, config=None):
    """
    Master cleaning function.
    Returns (clean_df, cleaning_log: list[dict]).
    """
    if config is None:
        config = load_config()
    if raw_path is None:
        raw_path = ROOT / config["data"]["raw_file"]

    log = []

    def note(action, detail, rows_affected=None):
        log.append({"action": action, "detail": detail, "rows_affected": rows_affected})

    # --- Load ---
    df = pd.read_csv(raw_path)
    note("load", f"Loaded {len(df)} rows, {len(df.columns)} columns from {raw_path.name}")

    n_start = len(df)

    # --- 1. Strip whitespace from all string columns ---
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    note("strip_whitespace", f"Stripped whitespace from {len(str_cols)} object columns")

    # --- 2. Clean target: Time_taken(min) → integer ---
    # Format: "(min) 24"
    df["Time_taken(min)"] = (
        df["Time_taken(min)"]
        .str.replace(r"\(min\)\s*", "", regex=True)
        .astype(float)
    )
    note("clean_target", "Extracted numeric value from Time_taken(min)")

    # --- 3. Clean Weatherconditions: strip 'conditions ' prefix ---
    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        r"^conditions\s*", "", regex=True
    )
    note("clean_weather", "Removed 'conditions ' prefix from Weatherconditions")

    # --- 4. Convert 'NaN' strings to actual NaN ---
    for col in df.columns:
        mask = df[col].astype(str).str.strip().isin(["NaN", "nan", ""])
        n_nan = mask.sum()
        if n_nan > 0:
            df.loc[mask, col] = np.nan
            note("nan_strings", f"Converted {n_nan} 'NaN' strings to NaN in {col}", n_nan)

    # --- 5. Numeric type conversions ---
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    df["Vehicle_condition"] = pd.to_numeric(df["Vehicle_condition"], errors="coerce")
    note("type_cast", "Cast Age, Ratings, multiple_deliveries, Vehicle_condition to numeric")

    # --- 6. Parse Order_Date ---
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    note("parse_date", "Parsed Order_Date to datetime (DD-MM-YYYY)")

    # --- 7. Parse Time_Orderd (note the typo in the column name) ---
    time_col = "Time_Orderd" if "Time_Orderd" in df.columns else "Time_Ordered"
    df["order_hour"] = pd.to_datetime(df[time_col], format="%H:%M:%S", errors="coerce").dt.hour
    df["order_minute"] = pd.to_datetime(df[time_col], format="%H:%M:%S", errors="coerce").dt.minute
    note("parse_time", f"Extracted order_hour and order_minute from {time_col}")

    # --- 8. Compute haversine distance ---
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )
    note("distance", "Computed haversine distance_km from store to customer")

    # --- 9. Date features ---
    df["day_of_week"] = df["Order_Date"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["Order_Date"].dt.month
    note("date_features", "Added day_of_week, is_weekend, month")

    # --- 10. Peak hour flags ---
    peak = config.get("peak_hours", {})
    lunch = peak.get("lunch", [12, 14])
    dinner = peak.get("dinner", [19, 22])
    df["is_lunch_peak"] = df["order_hour"].between(lunch[0], lunch[1]).astype(int)
    df["is_dinner_peak"] = df["order_hour"].between(dinner[0], dinner[1]).astype(int)
    df["is_peak"] = (df["is_lunch_peak"] | df["is_dinner_peak"]).astype(int)
    note("peak_flags", f"Added is_lunch_peak, is_dinner_peak, is_peak (lunch={lunch}, dinner={dinner})")

    # --- 11. Remove impossible coordinates (lat/lon = 0) ---
    coord_cols = [
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
    ]
    zero_mask = (df[coord_cols] == 0).any(axis=1)
    n_zero = zero_mask.sum()
    if n_zero > 0:
        df = df[~zero_mask].copy()
        note("drop_zero_coords", f"Dropped {n_zero} rows with zero lat/lon", n_zero)

    # --- 12. Remove extreme distances (> 100 km likely erroneous) ---
    extreme_dist = df["distance_km"] > 100
    n_extreme = extreme_dist.sum()
    if n_extreme > 0:
        df = df[~extreme_dist].copy()
        note("drop_extreme_distance", f"Dropped {n_extreme} rows with distance > 100 km", n_extreme)

    # --- 13. Remove extreme target values ---
    target_mask = (df["Time_taken(min)"] < 1) | (df["Time_taken(min)"] > 120)
    n_target_outlier = target_mask.sum()
    if n_target_outlier > 0:
        df = df[~target_mask].copy()
        note("drop_extreme_target", f"Dropped {n_target_outlier} rows with target < 1 or > 120 min", n_target_outlier)

    # --- 14. Drop Age outliers ---
    age_mask = (df["Delivery_person_Age"] < 16) | (df["Delivery_person_Age"] > 65)
    n_age = age_mask.sum()
    if n_age > 0:
        df = df[~age_mask].copy()
        note("drop_age_outlier", f"Dropped {n_age} rows with age < 16 or > 65", n_age)

    # --- 15. Fill remaining NaN in numeric columns with median ---
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        n_na = df[col].isna().sum()
        if n_na > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            note("fill_median", f"Filled {n_na} NaN in {col} with median={median_val:.2f}", n_na)

    # --- 16. Fill remaining NaN in categorical columns with mode ---
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        n_na = df[col].isna().sum()
        if n_na > 0:
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
            df[col] = df[col].fillna(mode_val)
            note("fill_mode", f"Filled {n_na} NaN in {col} with mode='{mode_val}'", n_na)

    # --- 17. Drop leakage columns ---
    leakage_cols = ["Time_Order_picked"]
    existing_leakage = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=existing_leakage)
    note("drop_leakage", f"Dropped leakage columns: {existing_leakage}")

    # --- 18. Drop original time/date string columns (replaced by features) ---
    drop_raw = [c for c in [time_col, "Order_Date"] if c in df.columns]
    df = df.drop(columns=drop_raw)
    note("drop_raw_temporal", f"Dropped raw temporal columns: {drop_raw}")

    # --- 19. Late flag ---
    sla = config["sla"]["late_threshold_min"]
    df["is_late"] = (df["Time_taken(min)"] > sla).astype(int)
    note("late_flag", f"Created is_late flag (SLA = {sla} min). Late rate = {df['is_late'].mean():.2%}")

    # --- 20. Reset index ---
    df = df.reset_index(drop=True)

    n_end = len(df)
    note("summary", f"Final: {n_end} rows ({n_start - n_end} removed, {(n_start - n_end) / n_start:.1%})")

    return df, log


def save_clean_data(df, log, config=None):
    """Save clean data and cleaning log."""
    if config is None:
        config = load_config()

    out_path = ROOT / config["data"]["processed_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    log_df = pd.DataFrame(log)
    log_df.to_csv(ROOT / "results" / "cleaning_log.csv", index=False)

    return out_path


if __name__ == "__main__":
    print("Running data cleaning pipeline...")
    df, log = clean_raw_data()
    out = save_clean_data(df, log)
    print(f"Clean data saved to {out} ({len(df)} rows)")
    print("\nCleaning log:")
    for entry in log:
        print(f"  [{entry['action']}] {entry['detail']}")
