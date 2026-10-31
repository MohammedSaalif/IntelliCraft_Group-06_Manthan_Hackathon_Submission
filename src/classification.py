"""
Phase 5: Classification models — late order prediction with cost-based threshold.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    _test_xgb = XGBClassifier(n_estimators=1)
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.features import (
    build_preprocessor,
    get_feature_columns,
    leakage_check,
)
from src.cleaning import load_config

ROOT = Path(__file__).resolve().parent.parent


def train_classification_models(splits, df, config=None):
    """
    Train baseline + 3 classifiers for late-order prediction.
    Target: is_late (derived from SLA in config.yaml).
    Returns dict of {name: {model, metrics, probas}}.
    """
    if config is None:
        config = load_config()

    seed = config["random_state"]
    sla = config["sla"]["late_threshold_min"]

    # Leakage check
    leakage_check(get_feature_columns())

    X_train, y_train_reg = splits["X_train"], splits["y_train"]
    X_val, y_val_reg = splits["X_val"], splits["y_val"]
    X_test, y_test_reg = splits["X_test"], splits["y_test"]

    # Create binary labels from the continuous target
    y_train = (y_train_reg > sla).astype(int)
    y_val = (y_val_reg > sla).astype(int)
    y_test = (y_test_reg > sla).astype(int)

    late_rate = y_train.mean()
    print(f"Late rate (SLA={sla}min): train={y_train.mean():.2%}, val={y_val.mean():.2%}, test={y_test.mean():.2%}")

    models = {}

    # --- Baseline ---
    baseline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", DummyClassifier(strategy="stratified", random_state=seed)),
    ])
    baseline.fit(X_train, y_train)
    models["baseline_stratified"] = baseline

    # --- Logistic Regression ---
    lr = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # --- Random Forest ---
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # --- Gradient Boosting (XGBoost or HistGradientBoosting) ---
    scale_pos_weight = (1 - late_rate) / max(late_rate, 0.01)
    if HAS_XGB:
        xgb = Pipeline([
            ("prep", build_preprocessor()),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
            )),
        ])
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
    else:
        gb = Pipeline([
            ("prep", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=8,
                learning_rate=0.1,
                class_weight="balanced",
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
            try:
                probas = model.predict_proba(X)[:, 1]
                metrics[f"{split_name}_roc_auc"] = round(roc_auc_score(y, probas), 4)
                metrics[f"{split_name}_pr_auc"] = round(average_precision_score(y, probas), 4)
            except Exception:
                probas = None
                metrics[f"{split_name}_roc_auc"] = None
                metrics[f"{split_name}_pr_auc"] = None

            metrics[f"{split_name}_f1"] = round(f1_score(y, preds), 4)
            metrics[f"{split_name}_precision"] = round(precision_score(y, preds, zero_division=0), 4)
            metrics[f"{split_name}_recall"] = round(recall_score(y, preds, zero_division=0), 4)

        results[name] = {
            "model": model,
            "metrics": metrics,
        }

    return results, y_train, y_val, y_test


def find_cost_optimal_threshold(model, X_val, y_val, cost_config):
    """
    Find the threshold that minimises expected cost.
    cost = missed_late_cost * FN + intervention_cost * FP
    Returns (best_threshold, cost_at_threshold, threshold_analysis_df).
    """
    probas = model.predict_proba(X_val)[:, 1]

    missed_cost = cost_config["missed_late_cost"]
    interv_cost = cost_config["intervention_cost"]

    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        preds = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        total_cost = missed_cost * fn + interv_cost * fp
        rows.append({
            "threshold": round(t, 2),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "expected_cost": total_cost,
        })

    analysis_df = pd.DataFrame(rows)
    best_row = analysis_df.loc[analysis_df["expected_cost"].idxmin()]
    best_threshold = best_row["threshold"]

    return best_threshold, best_row["expected_cost"], analysis_df


def sensitivity_analysis(model, X_val, y_val, cost_config):
    """Run threshold optimisation for different cost ratios."""
    results = []
    for ratio in cost_config["sensitivity_ratios"]:
        modified_config = cost_config.copy()
        modified_config["missed_late_cost"] = ratio
        modified_config["intervention_cost"] = 1
        best_t, best_cost, _ = find_cost_optimal_threshold(model, X_val, y_val, modified_config)
        results.append({
            "cost_ratio_missed_to_intervention": ratio,
            "optimal_threshold": best_t,
            "expected_cost": best_cost,
        })
    return pd.DataFrame(results)


def calibrate_model(model, X_val, y_val, method="isotonic"):
    """
    Calibrate the classifier probabilities.
    Returns calibrated model.
    """
    # Extract the base estimator for calibration
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def build_surge_windows(df, config=None):
    """
    Identify surge-staffing windows: hour × weekday cells where late rate
    exceeds network average by min_excess_late_pp percentage points.
    """
    if config is None:
        config = load_config()

    surge = config["surge_rule"]
    min_orders = surge["min_window_orders"]
    min_excess_pp = surge["min_excess_late_pp"]

    network_late_rate = df["is_late"].mean()

    # Hour × weekday table
    hw = df.groupby(["order_hour", "day_of_week"]).agg(
        n_orders=("is_late", "count"),
        late_rate=("is_late", "mean"),
    ).reset_index()

    hw["excess_pp"] = (hw["late_rate"] - network_late_rate) * 100

    surge_windows = hw[
        (hw["n_orders"] >= min_orders) &
        (hw["excess_pp"] >= min_excess_pp)
    ].sort_values("excess_pp", ascending=False)

    return hw, surge_windows


def save_classification_outputs(results, best_name, best_result, threshold, eta_band_info,
                                 sensitivity_df, surge_windows, hw_table, config=None):
    """Save all classification outputs."""
    if config is None:
        config = load_config()

    # 1. Comparison table
    rows = []
    for name, r in results.items():
        row = {"model": name, **r["metrics"]}
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(ROOT / "results" / "classification_comparison.csv", index=False)

    # 2. Save best model
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result["model"], ROOT / "models" / "classifier.joblib")

    # 3. Metadata
    metadata = {
        "best_model": best_name,
        "metrics": best_result["metrics"],
        "threshold": float(threshold),
        "features": get_feature_columns(),
        "sla_min": config["sla"]["late_threshold_min"],
        "random_state": config["random_state"],
    }
    with open(ROOT / "models" / "metadata_classifier.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Sensitivity
    sensitivity_df.to_csv(ROOT / "results" / "threshold_sensitivity.csv", index=False)

    # 5. Surge windows
    surge_windows.to_csv(ROOT / "results" / "surge_windows.csv", index=False)
    hw_table.to_csv(ROOT / "results" / "hour_weekday_late_rate.csv", index=False)

    return comp_df
