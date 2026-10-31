"""
Phase 4: Regression models — baseline, Ridge, Random Forest, XGBoost.
Predicts delivery time with ETA uncertainty band.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    # Verify dynamic library loading works
    _test_xgb = XGBRegressor(n_estimators=1)
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.features import (
    build_preprocessor,
    get_transformed_feature_names,
    get_feature_columns,
    leakage_check,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.cleaning import load_config

ROOT = Path(__file__).resolve().parent.parent


def train_regression_models(splits, config=None):
    """
    Train baseline + 3 regression models.
    Returns dict of {name: (pipeline, metrics_dict)}.
    """
    if config is None:
        config = load_config()

    seed = config["random_state"]

    # Leakage check
    leakage_check(get_feature_columns())

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    models = {}

    # --- Baseline: predict mean ---
    baseline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", DummyRegressor(strategy="mean")),
    ])
    baseline.fit(X_train, y_train)
    models["baseline_mean"] = baseline

    # --- Ridge Regression ---
    ridge = Pipeline([
        ("prep", build_preprocessor()),
        ("model", Ridge(alpha=1.0, random_state=seed)),
    ])
    ridge.fit(X_train, y_train)
    models["ridge"] = ridge

    # --- Random Forest ---
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # --- Gradient Boosting (XGBoost or HistGradientBoosting) ---
    if HAS_XGB:
        xgb = Pipeline([
            ("prep", build_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
            )),
        ])
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
    else:
        gb = Pipeline([
            ("prep", build_preprocessor()),
            ("model", HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=8,
                learning_rate=0.1,
                random_state=seed,
            )),
        ])
        gb.fit(X_train, y_train)
        models["gradient_boosting"] = gb

    # Evaluate all
    results = {}
    for name, model in models.items():
        metrics = {}
        for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            preds = model.predict(X)
            metrics[f"{split_name}_mae"] = round(mean_absolute_error(y, preds), 4)
            metrics[f"{split_name}_rmse"] = round(np.sqrt(mean_squared_error(y, preds)), 4)
            metrics[f"{split_name}_r2"] = round(r2_score(y, preds), 4)
        results[name] = {"model": model, "metrics": metrics}

    return results


def select_best_model(results, metric="val_mae"):
    """Select the best model by validation MAE (lower is better)."""
    best_name = min(results, key=lambda k: results[k]["metrics"][metric])
    return best_name, results[best_name]


def compute_eta_band(model, X, y, coverage=0.80):
    """
    Compute the ETA uncertainty band that covers `coverage` fraction of orders.
    Uses residuals to compute quantile-based prediction interval.
    Returns (lower_q, upper_q, actual_coverage).
    """
    preds = model.predict(X)
    residuals = y.values - preds  # actual - predicted

    alpha = (1 - coverage) / 2
    lower_q = np.quantile(residuals, alpha)
    upper_q = np.quantile(residuals, 1 - alpha)

    # Check actual coverage
    within = ((y.values - preds >= lower_q) & (y.values - preds <= upper_q)).mean()

    return {
        "lower_residual_q": round(float(lower_q), 2),
        "upper_residual_q": round(float(upper_q), 2),
        "coverage_target": coverage,
        "coverage_actual": round(float(within), 4),
    }


def generate_oof_predictions(splits, best_model, df):
    """
    Generate out-of-fold predictions for the best model.
    Returns a DataFrame with columns: ID, Delivery_person_ID, actual, predicted.
    """
    # Use val set as OOF proxy (for single model)
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    val_idx = splits["val_idx"]

    preds = best_model.predict(X_val)

    oof_df = pd.DataFrame({
        "ID": df.loc[val_idx, "ID"].values,
        "Delivery_person_ID": df.loc[val_idx, "Delivery_person_ID"].values,
        "actual": y_val.values,
        "predicted": np.round(preds, 2),
    })
    return oof_df


def save_regression_outputs(results, best_name, best_result, eta_band, oof_df, config=None):
    """Save all regression outputs."""
    if config is None:
        config = load_config()

    # 1. Save comparison table
    rows = []
    for name, r in results.items():
        row = {"model": name, **r["metrics"]}
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(ROOT / "results" / "regression_comparison.csv", index=False)

    # 2. Save best model
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result["model"], ROOT / "models" / "regressor.joblib")

    # 3. Save metadata
    metadata = {
        "best_model": best_name,
        "metrics": best_result["metrics"],
        "eta_band": eta_band,
        "features": get_feature_columns(),
        "random_state": config["random_state"],
    }
    with open(ROOT / "models" / "metadata_regression.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Save OOF predictions
    oof_df.to_csv(ROOT / "results" / "oof_regression.csv", index=False)

    return comp_df
