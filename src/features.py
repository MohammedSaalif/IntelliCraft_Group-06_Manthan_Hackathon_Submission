"""
Phase 2: Feature engineering and preprocessing pipeline.
Provides shared feature columns and sklearn preprocessor for all models.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# === Feature policy ===
# Only features known at ORDER PLACEMENT time.
# Forbidden: Time_Order_picked, anything derived from it, the target, raw IDs.

NUMERIC_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "distance_km",
    "order_hour",
    "order_minute",
    "day_of_week",
    "is_weekend",
    "month",
    "is_lunch_peak",
    "is_dinner_peak",
    "is_peak",
]

CATEGORICAL_FEATURES = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

TARGET = "Time_taken(min)"

LEAKAGE_COLUMNS = [
    "Time_Order_picked",
    "is_late",  # derived from target; only for classifier label, never a feature
]

ID_COLUMNS = ["ID", "Delivery_person_ID"]


def get_feature_columns():
    """Return the ordered list of feature columns (numeric + categorical)."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_numeric_features():
    return NUMERIC_FEATURES.copy()


def get_categorical_features():
    return CATEGORICAL_FEATURES.copy()


def build_preprocessor():
    """
    Build a ColumnTransformer that:
      - Scales numeric features
      - One-hot encodes categorical features
    Returns (preprocessor, feature_names_func) where feature_names_func
    is a callable that returns transformed feature names after fitting.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_transformed_feature_names(preprocessor):
    """Get feature names after fitting the preprocessor."""
    num_names = NUMERIC_FEATURES.copy()
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names


def make_splits(df, config=None):
    """
    Create train/val/test split.
    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                             train_idx, val_idx, test_idx
    """
    if config is None:
        config = load_config()

    seed = config["random_state"]
    test_size = config["split"]["test_size"]
    val_size = config["split"]["val_size_within_train"]

    features = get_feature_columns()
    X = df[features].copy()
    y = df[TARGET].copy()

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=seed
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval, y_trainval, idx_trainval, test_size=val_size, random_state=seed
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "train_idx": idx_train, "val_idx": idx_val, "test_idx": idx_test,
    }


def leakage_check(feature_list):
    """Verify no leakage columns are in the feature list."""
    forbidden = set(LEAKAGE_COLUMNS + ID_COLUMNS + [TARGET])
    overlap = set(feature_list) & forbidden
    if overlap:
        raise ValueError(f"LEAKAGE DETECTED: {overlap} found in feature list!")
    return True
