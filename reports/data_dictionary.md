# Data Dictionary — Quick-Commerce Delivery Intelligence

## 1. Overview
- **Raw source:** `data/raw/train.csv` (45,593 rows × 20 columns)
- **Cleaned dataset:** `data/processed/clean.parquet` (41,497 rows × 24 feature columns + target & labels)
- **Target Variable (Regression):** `Time_taken(min)` (Continuous, range: 10–54 minutes)
- **Target Variable (Classification):** `is_late` (Binary: 1 if `Time_taken(min)` > 30 min SLA, else 0)

---

## 2. Feature Policy & Data Leakage Prevention
To guarantee that the models are deployable in production at **order placement time**, strict data governance rules were applied:
- **Strictly Allowed:** Information known at or before order confirmation (store location, customer location, order timestamp, scheduled delivery items, vehicle type, weather/traffic conditions at placement).
- **Strictly Forbidden (Leakage Eliminated):**
  - `Time_Order_picked`: Known only after rider arrives at merchant and picks up goods. Using this would create target leakage by incorporating pickup delay directly.
  - Target-derived metrics computed across the global dataset (e.g., rider mean historical delay without temporal partitioning).
  - Raw arbitrary identifiers (`ID`, `Delivery_person_ID`) directly passed as predictive features.

---

## 3. Complete Field Definitions

| Column Name | Raw Data Type | Clean Data Type | Description | Values / Range | Preprocessing / Imputation |
|---|---|---|---|---|---|
| `ID` | String | String | Unique identifier for delivery record | Hex strings (e.g. `0x4607`) | Whitespace stripped. Kept for indexing only. |
| `Delivery_person_ID` | String | String | Identifier for delivery rider | String (e.g. `INDORES13DEL02`) | Whitespace stripped. Used for clustering & OOF evaluation. |
| `Delivery_person_Age` | String/Object | Float64 | Age of the delivery partner | 18–60 years | Stripped whitespace, cast to float, filled missing with median (30.0 yrs). Outliers (<16 or >65) removed. |
| `Delivery_person_Ratings` | String/Object | Float64 | Partner customer rating | 1.0–5.0 | Stripped whitespace, cast to float, filled missing with median (4.70). |
| `Restaurant_latitude` | Float64 | Float64 | Merchant location latitude | ~11.0°N to ~30.9°N | Lat/Lon 0.0 missing sentinels (3,640 rows) dropped. |
| `Restaurant_longitude` | Float64 | Float64 | Merchant location longitude | ~72.8°E to ~88.4°E | Scaled via StandardScaler in pipeline. |
| `Delivery_location_latitude` | Float64 | Float64 | Customer drop location latitude | ~11.0°N to ~31.0°N | Co-located with restaurant coordinates. |
| `Delivery_location_longitude` | Float64 | Float64 | Customer drop location longitude | ~72.8°E to ~88.5°E | Scaled via StandardScaler in pipeline. |
| `distance_km` | *Engineered* | Float64 | Great-circle Haversine distance from merchant to customer | 0.05–21.0 km | Computed via vectorised spherical Haversine formula. Outliers (>100 km) removed. |
| `order_hour` | *Engineered* | Int64 | Hour of the day the order was placed | 0–23 | Extracted from `Time_Orderd`. Missing imputed with mode (19). |
| `order_minute` | *Engineered* | Int64 | Minute of the hour | 0–59 | Extracted from `Time_Orderd`. Missing imputed with median (30). |
| `day_of_week` | *Engineered* | Int64 | Day of the week | 0 (Monday) – 6 (Sunday) | Extracted from `Order_Date`. |
| `is_weekend` | *Engineered* | Int64 | Binary flag for Saturday or Sunday | 0 or 1 | 1 if day_of_week in [5, 6]. |
| `month` | *Engineered* | Int64 | Month of order | 1–12 | Extracted from `Order_Date`. |
| `is_lunch_peak` | *Engineered* | Int64 | Binary flag for lunch window | 0 or 1 | 1 if order_hour between 12 and 14. |
| `is_dinner_peak` | *Engineered* | Int64 | Binary flag for dinner window | 0 or 1 | 1 if order_hour between 19 and 22. |
| `is_peak` | *Engineered* | Int64 | Combined peak hour flag | 0 or 1 | 1 if lunch or dinner peak. |
| `Weatherconditions` | String | String / Categorical | Atmospheric conditions at placement | Sunny, Stormy, Sandstorms, Cloudy, Fog, Windy | Removed `'conditions '` prefix, converted `'NaN'` to mode (`Fog`). One-hot encoded. |
| `Road_traffic_density` | String | String / Categorical | Traffic congestion level | Low, Medium, High, Jam | Stripped spaces, converted `'NaN'` to mode (`Low`). One-hot encoded. |
| `Vehicle_condition` | Int64 | Float64 | Operational condition rating of vehicle | 0 (poor) to 3 (excellent) | Scaled via StandardScaler in pipeline. |
| `Type_of_order` | String | String / Categorical | Category of meal ordered | Snack, Meal, Drinks, Buffet | Stripped trailing spaces. One-hot encoded. |
| `Type_of_vehicle` | String | String / Categorical | Delivery vehicle type | motorcycle, scooter, electric_scooter, bicycle | Stripped trailing spaces. One-hot encoded. |
| `multiple_deliveries` | Float64 | Float64 | Count of concurrent deliveries rider is handling | 0, 1, 2, 3 | Imputed with median (1.0). Scaled via StandardScaler. |
| `Festival` | String | String / Categorical | Indicates active cultural festival/holiday | No, Yes | Converted `'NaN'` to mode (`No`). One-hot encoded. |
| `City` | String | String / Categorical | Urban zone classification | Metropolitian, Urban, Semi-Urban | Converted `'NaN'` to mode (`Metropolitian`). One-hot encoded. |
| `Time_taken(min)` | String | Float64 | Total delivery duration (Target) | 10–54 minutes | Extracted numeric from `(min) XX`. |
| `is_late` | *Engineered* | Int64 | Target flag: late delivery indicator | 0 (On-time), 1 (Late) | Evaluated as `Time_taken(min) > 30` (SLA threshold). |

---

## 4. Split Strategy
- **Holdout Test Set:** 20% (8,300 rows) — Held strictly untouched during tuning, evaluated only once for final benchmarks.
- **Validation Set:** 20% within train (6,640 rows) — Used for model selection, threshold optimization, and probability calibration.
- **Train Set:** 64% of total (26,557 rows) — Used for pipeline training.
- **Random Seed:** 42 (fixed single source of truth across all scripts).
