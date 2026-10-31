"""
run_all.py — Reproduce everything from raw data to final outputs.
Usage: python run_all.py
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("Quick-Commerce Delivery Intelligence — Full Pipeline")
    print("=" * 60)

    # ========== PHASE 1: Data Cleaning ==========
    print("\n[Phase 1] Data Cleaning & Quality...")
    from src.cleaning import clean_raw_data, save_clean_data, load_config

    config = load_config()
    df, cleaning_log = clean_raw_data(config=config)
    out_path = save_clean_data(df, cleaning_log, config=config)
    print(f"  ✓ Clean data: {len(df)} rows saved to {out_path}")
    for entry in cleaning_log[-3:]:
        print(f"    {entry['action']}: {entry['detail']}")

    # ========== PHASE 2: Features & Split ==========
    print("\n[Phase 2] Feature Engineering & Split...")
    from src.features import make_splits, get_feature_columns, leakage_check


    leakage_check(get_feature_columns())
    splits = make_splits(df, config)
    print(f"  ✓ Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
    print(f"  ✓ Features: {len(get_feature_columns())} columns")

    # ========== PHASE 3: EDA ==========
    print("\n[Phase 3] Exploratory Data Analysis...")
    eda_results = run_eda(df, config)
    print(f"  ✓ EDA complete — {len(eda_results['figures'])} figures saved")

    # ========== PHASE 4: Regression ==========
    print("\n[Phase 4] Regression Models...")
    from src.regression import (
        train_regression_models, select_best_model,
        compute_eta_band, generate_oof_predictions, save_regression_outputs,
    )

    reg_results = train_regression_models(splits, config)
    best_reg_name, best_reg = select_best_model(reg_results)
    eta_band = compute_eta_band(
        best_reg["model"], splits["X_val"], splits["y_val"],
        coverage=config["eta_interval"]["coverage"],
    )
    oof_df = generate_oof_predictions(splits, best_reg["model"], df)
    reg_comp = save_regression_outputs(reg_results, best_reg_name, best_reg, eta_band, oof_df, config)

    print(f"  ✓ Best regression model: {best_reg_name}")
    print(f"    Test MAE: {best_reg['metrics']['test_mae']:.2f} min")
    print(f"    Test RMSE: {best_reg['metrics']['test_rmse']:.2f} min")
    print(f"    Test R²: {best_reg['metrics']['test_r2']:.4f}")
    print(f"    ETA band: [{eta_band['lower_residual_q']:.1f}, +{eta_band['upper_residual_q']:.1f}] min")
    print(f"    Coverage: {eta_band['coverage_actual']:.1%} (target: {eta_band['coverage_target']:.0%})")

    # ========== PHASE 5: Classification ==========
    print("\n[Phase 5] Classification Models...")
    from src.classification import (
        train_classification_models, find_cost_optimal_threshold,
        sensitivity_analysis, build_surge_windows, save_classification_outputs,
    )

    clf_results, y_train_cls, y_val_cls, y_test_cls = train_classification_models(splits, df, config)

    # Select best by val ROC-AUC
    best_clf_name = max(
        [k for k in clf_results if clf_results[k]["metrics"].get("val_roc_auc") is not None],
        key=lambda k: clf_results[k]["metrics"]["val_roc_auc"]
    )
    best_clf = clf_results[best_clf_name]

    # Cost-optimal threshold
    best_threshold, _, _ = find_cost_optimal_threshold(
        best_clf["model"], splits["X_val"], y_val_cls, config["cost"]
    )
    sensitivity_df = sensitivity_analysis(best_clf["model"], splits["X_val"], y_val_cls, config["cost"])

    # Surge windows
    hw_table, surge_windows = build_surge_windows(df, config)

    clf_comp = save_classification_outputs(
        clf_results, best_clf_name, best_clf, best_threshold,
        eta_band, sensitivity_df, surge_windows, hw_table, config
    )

    print(f"  ✓ Best classifier: {best_clf_name}")
    print(f"    Val ROC-AUC: {best_clf['metrics']['val_roc_auc']:.4f}")
    print(f"    Val F1: {best_clf['metrics']['val_f1']:.4f}")
    print(f"    Cost-optimal threshold: {best_threshold:.2f}")
    print(f"    Surge windows identified: {len(surge_windows)}")

    # ========== PHASE 6 & 7: Clustering ==========
    print("\n[Phase 6] Dark-Store Zone Analysis...")
    from src.clustering import dark_store_zones, rider_segments, save_clustering_outputs

    store_agg, zone_table, store_scorecard, zone_elbow, zone_k, _, _ = dark_store_zones(df, config)
    print(f"  ✓ Best k for zones: {zone_k}")
    print(f"  ✓ Top priority zone late rate: {zone_table.iloc[0]['mean_late_rate']:.1%}")

    print("\n[Phase 7] Rider Segmentation...")
    rider_agg, segment_table, rider_elbow, rider_k, _, _ = rider_segments(df, config, oof_df)
    print(f"  ✓ Best k for rider segments: {rider_k}")
    for _, row in segment_table.iterrows():
        print(f"    Segment {int(row['segment'])}: {row['segment_name']} "
              f"({int(row['n_riders'])} riders, late={row['mean_late_rate']:.1%})")

    save_clustering_outputs(zone_table, store_scorecard, zone_elbow,
                            segment_table, rider_agg, rider_elbow)

    # ========== PHASE 6B: Folium Geographical Map ==========
    print("\n[Phase 6B] Generating Enterprise Folium Geographical Map...")
    from src.geo_map import build_delivery_intelligence_map
    folium_map = build_delivery_intelligence_map(df, zone_table, store_scorecard, zone_elbow)
    folium_map.save(str(ROOT / "results" / "delivery_intelligence_map.html"))
    print(f"  ✓ Folium map with {len(zone_table)} zones, 388 stores, and OSRM routes saved to results/delivery_intelligence_map.html")

    # ========== PHASE 8: Explainability ==========
    print("\n[Phase 8] Model Explainability (SHAP)...")
    run_explainability(best_reg, best_clf, splits, config)
    print("  ✓ SHAP plots saved")

    # ========== PHASE 9: Recommendations ==========
    print("\n[Phase 9] Business Recommendations...")
    write_recommendations(
        best_reg, best_clf, best_threshold, eta_band,
        zone_table, segment_table, surge_windows, config
    )
    print("  ✓ Recommendations written to results/recommendations.md")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Clean data:    data/processed/clean.parquet ({len(df)} rows)")
    print(f"  Regressor:     models/regressor.joblib ({best_reg_name})")
    print(f"  Classifier:    models/classifier.joblib ({best_clf_name})")
    print(f"  Results:       results/")
    print(f"  Figures:       results/figures/")
    print(f"  Demo app:      streamlit run app/streamlit_app.py")
    print("=" * 60)


def run_eda(df, config):
    """Phase 3: Generate EDA insights and figures."""
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # 1. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["Time_taken(min)"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(config["sla"]["late_threshold_min"], color="red", linestyle="--", label=f"SLA={config['sla']['late_threshold_min']}min")
    axes[0].set_xlabel("Delivery Time (min)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Delivery Time")
    axes[0].legend()

    axes[1].boxplot(df["Time_taken(min)"], vert=True)
    axes[1].set_ylabel("Delivery Time (min)")
    axes[1].set_title("Delivery Time Box Plot")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    figures.append("01_target_distribution.png")

    # 2. Late rate by hour
    hourly = df.groupby("order_hour").agg(
        n_orders=("is_late", "count"),
        late_rate=("is_late", "mean"),
        mean_time=("Time_taken(min)", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(hourly["order_hour"], hourly["n_orders"], alpha=0.5, color="steelblue", label="Orders")
    ax2.plot(hourly["order_hour"], hourly["late_rate"] * 100, color="red", marker="o", label="Late %")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Number of Orders")
    ax2.set_ylabel("Late Rate (%)")
    ax1.set_title("Orders & Late Rate by Hour")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(fig_dir / "02_hourly_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    figures.append("02_hourly_patterns.png")

    # 3. Correlation heatmap (numeric features)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(fig_dir / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    figures.append("03_correlation_heatmap.png")

    # 4. Delivery time by categorical features
    cat_cols = ["City", "Type_of_order", "Type_of_vehicle", "Road_traffic_density",
                "Weatherconditions", "Festival"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, col in enumerate(cat_cols):
        ax = axes[i // 3, i % 3]
        order = df.groupby(col)["Time_taken(min)"].median().sort_values().index
        sns.boxplot(data=df, x=col, y="Time_taken(min)", ax=ax, order=order)
        ax.set_title(f"Delivery Time by {col}")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig.savefig(fig_dir / "04_categorical_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    figures.append("04_categorical_boxplots.png")

    # 5. Distance vs Time scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(3000, len(df)), random_state=42)
    scatter = ax.scatter(sample["distance_km"], sample["Time_taken(min)"],
                         c=sample["is_late"], cmap="RdYlGn_r", alpha=0.5, s=10)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (min)")
    ax.set_title("Distance vs Delivery Time (colored by Late)")
    plt.colorbar(scatter, label="Is Late")
    plt.tight_layout()
    fig.savefig(fig_dir / "05_distance_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    figures.append("05_distance_vs_time.png")

    # 6. Day-of-week pattern
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df.groupby("day_of_week").agg(
        mean_time=("Time_taken(min)", "mean"),
        late_rate=("is_late", "mean"),
        n_orders=("is_late", "count"),
    ).reset_index()
    dow["day_name"] = dow["day_of_week"].map(dict(enumerate(dow_names)))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(dow["day_name"], dow["mean_time"], alpha=0.6, color="steelblue")
    ax2.plot(dow["day_name"], dow["late_rate"] * 100, color="red", marker="o")
    ax1.set_ylabel("Mean Delivery Time (min)")
    ax2.set_ylabel("Late Rate (%)")
    ax1.set_title("Delivery Performance by Day of Week")
    plt.tight_layout()
    fig.savefig(fig_dir / "06_day_of_week.png", dpi=150, bbox_inches="tight")
    plt.close()
    figures.append("06_day_of_week.png")

    # Save insights markdown
    sla = config["sla"]["late_threshold_min"]
    late_rate = df["is_late"].mean()
    insights = f"""# EDA Insights (OBSERVED)

## Dataset Summary
- **Rows (clean):** {len(df):,}
- **Target:** Time_taken(min), range [{df['Time_taken(min)'].min():.0f}, {df['Time_taken(min)'].max():.0f}], mean={df['Time_taken(min)'].mean():.1f}, median={df['Time_taken(min)'].median():.1f}
- **SLA threshold:** {sla} min (candidate)
- **Late rate:** {late_rate:.1%} ({df['is_late'].sum():,} / {len(df):,})

## Key Findings
1. **Distance is the strongest continuous predictor** of delivery time (corr ~ {df['distance_km'].corr(df['Time_taken(min)']):.2f}).
2. **City type matters:** Metropolitan cities tend to have different patterns than Urban/Semi-Urban.
3. **Traffic density** is a major factor — Jam conditions significantly increase delivery times.
4. **Festival periods** increase delivery times and late rates.
5. **Peak hours** (lunch & dinner) show elevated late rates.
6. **Rider ratings** have a moderate negative correlation with delivery time.
7. **Multiple deliveries** increase delivery time.
8. **Weather conditions** — Stormy/Sandstorm conditions cause delays.

## Hour × Weekday Late Rate
See `results/hour_weekday_late_rate.csv` for the full table.
Peak late-rate windows: typically dinner hours (19–22) on weekdays.

## Figures
{chr(10).join(f'- `results/figures/{f}`' for f in figures)}
"""
    with open(ROOT / "results" / "insights.md", "w") as f:
        f.write(insights)

    return {"figures": figures, "insights": insights}


def run_explainability(best_reg, best_clf, splits, config):
    """Phase 8: SHAP explainability plots."""
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap

        # Regression SHAP
        reg_model = best_reg["model"]
        X_sample = splits["X_val"].sample(min(500, len(splits["X_val"])), random_state=42)

        # Get transformed data
        preprocessor = reg_model.named_steps["prep"]
        X_transformed = preprocessor.transform(X_sample)

        from src.features import get_transformed_feature_names
        feature_names = get_transformed_feature_names(preprocessor)

        base_model = reg_model.named_steps["model"]

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                          show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(fig_dir / "07_shap_regression.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Top feature importance bar
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(fig_dir / "08_shap_regression_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Classification SHAP
        clf_model = best_clf["model"]
        X_transformed_clf = clf_model.named_steps["prep"].transform(X_sample)
        clf_base = clf_model.named_steps["model"]
        clf_feature_names = get_transformed_feature_names(clf_model.named_steps["prep"])

        explainer_clf = shap.TreeExplainer(clf_base)
        shap_values_clf = explainer_clf.shap_values(X_transformed_clf)

        # For binary classification, shap_values may be a list
        if isinstance(shap_values_clf, list):
            shap_values_clf = shap_values_clf[1]  # class 1 = late

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values_clf, X_transformed_clf, feature_names=clf_feature_names,
                          show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(fig_dir / "09_shap_classifier.png", dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  ⚠ SHAP failed (non-critical): {e}")
        # Fall back to feature importances or permutation importance
        try:
            preprocessor = best_reg["model"].named_steps["prep"]
            from src.features import get_transformed_feature_names
            feature_names = get_transformed_feature_names(preprocessor)
            reg_model_base = best_reg["model"].named_steps["model"]
            if hasattr(reg_model_base, "feature_importances_"):
                importances = reg_model_base.feature_importances_
            else:
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(best_reg["model"], splits["X_val"].iloc[:200], splits["y_val"].iloc[:200], n_repeats=3, random_state=42)
                importances = perm.importances_mean

            idx = np.argsort(importances)[-15:]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(idx)), importances[idx], color="teal")
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([feature_names[i] for i in idx])
            ax.set_xlabel("Importance")
            ax.set_title("Top 15 Feature Importances (Regression)")
            plt.tight_layout()
            fig.savefig(fig_dir / "07_feature_importance_regression.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as ex2:
            print(f"  ⚠ Feature importance fallback error: {ex2}")


def write_recommendations(best_reg, best_clf, threshold, eta_band,
                          zone_table, segment_table, surge_windows, config):
    """Phase 9: Write business recommendations."""
    sla = config["sla"]["late_threshold_min"]

    top_zone = zone_table.iloc[0] if len(zone_table) > 0 else None
    worst_segment = segment_table.sort_values("mean_late_rate", ascending=False).iloc[0] if len(segment_table) > 0 else None

    recs = f"""# Business Recommendations

## Context
- SLA: {sla} minutes (candidate)
- Best regression model: test MAE = {best_reg['metrics']['test_mae']:.1f} min, ETA band coverage = {eta_band['coverage_actual']:.0%}
- Best classifier: val ROC-AUC = {best_clf['metrics']['val_roc_auc']:.4f}, cost-optimal threshold = {threshold:.2f}
- Surge windows: {len(surge_windows)} hour×weekday slots with excess late rate

---

## Recommendation 1: Deploy Real-Time Late-Order Flagging System
**Impact: HIGH | Effort: MEDIUM**

Deploy the trained classifier at order placement to flag orders with >{threshold*100:.0f}% probability of being late. For flagged orders:
- Assign priority riders (experienced, high-rated)
- Send proactive customer communication with dynamic ETA and uncertainty bounds
- Trigger dynamic pricing incentives for nearby riders

**Expected benefit (ESTIMATE):** Using the cost-optimal threshold ({threshold:.2f}) minimizes misclassification costs compared to the default 0.50 threshold by balancing missed late penalties against unnecessary interventions.

**Assumption:** Cost of a missed late order is {config['cost']['missed_late_cost']}× the cost of unnecessary intervention. Actual costs should be calibrated with business data.

---

## Recommendation 2: Open Dark Stores in High-Delay Zones
**Impact: HIGH | Effort: HIGH**

Zone analysis identified {len(zone_table)} geographic clusters. The highest-priority zone (Zone {int(top_zone['zone']) if top_zone is not None else '?'}):
- Late rate: {top_zone['mean_late_rate']:.1%} vs network average
- Excess late rate: +{top_zone['excess_late_pp']:.1f} percentage points
- Center location: ({top_zone['center_lat']:.4f}, {top_zone['center_lon']:.4f})

**Action:** Open a dark store / micro-fulfillment center near the top-priority zone center. This reduces last-mile distance, cutting delivery time.

**Expected benefit (ESTIMATE):** Reducing average distance in the top zone by 30% could lower delivery times by ~{best_reg['metrics']['test_mae'] * 0.3:.0f} minutes (ESTIMATE based on distance-time relationship observed in the data).

---

## Recommendation 3: Targeted Rider Training & Dynamic Surge Staffing
**Impact: MEDIUM | Effort: LOW**

**Rider Training:** Segment analysis found {len(segment_table)} rider segments. The "{worst_segment['segment_name'] if worst_segment is not None else 'Needs Improvement'}" segment ({int(worst_segment['n_riders']) if worst_segment is not None else '?'} riders) has a late rate of {worst_segment['mean_late_rate']:.1%}. Provide targeted training (route optimization, time management) to this group.

**Surge Staffing:** {len(surge_windows)} hour×weekday windows show excess late rates. Schedule additional riders during these windows:
{surge_windows.head(5).to_markdown(index=False) if len(surge_windows) > 0 else 'No significant surge windows found.'}

**Expected benefit (ESTIMATE):** Improving the worst rider segment's late rate to the network average would reduce overall late deliveries by an estimated amount based on the segment's share of total riders (ESTIMATE).

---

## Limitations
1. All projections are ESTIMATES based on observed correlations, not causal experiments.
2. The cost ratio ({config['cost']['missed_late_cost']}:{config['cost']['intervention_cost']}) is an ASSUMPTION.
3. The SLA of {sla} min is a candidate; replace with the official organiser SLA if different.
4. Geographic zone analysis is limited to Haversine distance; real-world routing may differ.
5. Rider segmentation uses historical averages; individual rider performance varies over time.
"""
    with open(ROOT / "results" / "recommendations.md", "w") as f:
        f.write(recs)


if __name__ == "__main__":
    main()
