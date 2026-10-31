"""
Phase 6 & 7: Dark-store zone clustering and rider segmentation.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.cleaning import load_config

ROOT = Path(__file__).resolve().parent.parent


def find_best_k(X_scaled, k_range, seeds, metric="silhouette"):
    """
    Run KMeans over k_range and multiple seeds.
    Returns (results_df, best_k).
    """
    rows = []
    for k in k_range:
        sil_scores = []
        inertias = []
        for seed in seeds:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
            sil_scores.append(sil)
            inertias.append(km.inertia_)

        rows.append({
            "k": k,
            "silhouette_mean": round(np.mean(sil_scores), 4),
            "silhouette_std": round(np.std(sil_scores), 4),
            "inertia_mean": round(np.mean(inertias), 2),
        })

    results_df = pd.DataFrame(rows)
    best_k = results_df.loc[results_df["silhouette_mean"].idxmax(), "k"]
    return results_df, int(best_k)


def dark_store_zones(df, config=None):
    """
    Phase 6: Cluster stores by location and delivery performance.
    Returns (zone_labels, zone_table, store_scorecard, elbow_results, best_k).
    """
    if config is None:
        config = load_config()

    seed = config["random_state"]
    k_range = range(*config["clustering"]["k_range"])
    seeds = config["clustering"]["seeds_for_stability"]

    # Features for zone clustering: store location + performance
    store_agg = df.groupby(["Restaurant_latitude", "Restaurant_longitude"]).agg(
        n_orders=("Time_taken(min)", "count"),
        mean_time=("Time_taken(min)", "mean"),
        late_rate=("is_late", "mean"),
        mean_distance=("distance_km", "mean"),
    ).reset_index()

    zone_features = ["Restaurant_latitude", "Restaurant_longitude", "mean_time", "late_rate", "mean_distance"]
    X_raw = store_agg[zone_features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Find best K
    elbow_df, best_k = find_best_k(X_scaled, k_range, seeds)

    # Final clustering with best K
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    store_agg["zone"] = km.fit_predict(X_scaled)

    # Network-level benchmark
    network_late_rate = df["is_late"].mean()
    network_mean_time = df["Time_taken(min)"].mean()

    # Zone table: aggregate per zone
    zone_table = store_agg.groupby("zone").agg(
        n_stores=("n_orders", "count"),
        total_orders=("n_orders", "sum"),
        mean_delivery_time=("mean_time", "mean"),
        mean_late_rate=("late_rate", "mean"),
        mean_distance_km=("mean_distance", "mean"),
        center_lat=("Restaurant_latitude", "mean"),
        center_lon=("Restaurant_longitude", "mean"),
    ).reset_index()

    zone_table["excess_late_pp"] = (zone_table["mean_late_rate"] - network_late_rate) * 100
    zone_table["excess_time_min"] = zone_table["mean_delivery_time"] - network_mean_time

    # Priority index: higher = worse performance = more need for dark store
    zone_table["priority_index"] = (
        zone_table["excess_late_pp"].clip(lower=0) * 0.5 +
        zone_table["excess_time_min"].clip(lower=0) * 0.3 +
        (1 / zone_table["mean_distance_km"].clip(lower=0.1)) * 0.2  # shorter distance but still late = problem
    )
    zone_table = zone_table.sort_values("priority_index", ascending=False)

    # Store scorecard
    store_scorecard = store_agg.copy()
    store_scorecard["vs_network_late"] = (store_scorecard["late_rate"] - network_late_rate) * 100
    store_scorecard["vs_network_time"] = store_scorecard["mean_time"] - network_mean_time

    return store_agg, zone_table, store_scorecard, elbow_df, best_k, km, scaler


def rider_segments(df, config=None, oof_df=None):
    """
    Phase 7: Segment riders by behavior.
    Returns (rider_agg, segment_table, elbow_results, best_k).
    """
    if config is None:
        config = load_config()

    seed = config["random_state"]
    k_range = range(*config["clustering"]["k_range"])
    seeds = config["clustering"]["seeds_for_stability"]
    min_orders = config["clustering"]["min_orders_per_rider"]

    # Rider-level aggregation
    rider_agg = df.groupby("Delivery_person_ID").agg(
        n_orders=("Time_taken(min)", "count"),
        mean_time=("Time_taken(min)", "mean"),
        std_time=("Time_taken(min)", "std"),
        late_rate=("is_late", "mean"),
        mean_rating=("Delivery_person_Ratings", "mean"),
        mean_distance=("distance_km", "mean"),
        age=("Delivery_person_Age", "first"),
        multi_delivery_rate=("multiple_deliveries", "mean"),
    ).reset_index()

    # Fill NaN std (riders with 1 order)
    rider_agg["std_time"] = rider_agg["std_time"].fillna(0)

    # Filter to riders with enough orders
    rider_agg = rider_agg[rider_agg["n_orders"] >= min_orders].copy()

    # Add route-adjusted delay if OOF predictions available
    if oof_df is not None:
        rider_delay = oof_df.groupby("Delivery_person_ID").agg(
            mean_residual=("actual", lambda x: (x - oof_df.loc[x.index, "predicted"]).mean()),
        ).reset_index()
        rider_agg = rider_agg.merge(rider_delay, on="Delivery_person_ID", how="left")
        rider_agg["mean_residual"] = rider_agg["mean_residual"].fillna(0)

    # Features for segmentation
    seg_features = ["mean_time", "std_time", "late_rate", "mean_rating", "mean_distance", "multi_delivery_rate"]
    if "mean_residual" in rider_agg.columns:
        seg_features.append("mean_residual")

    X_raw = rider_agg[seg_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Find best K
    elbow_df, best_k = find_best_k(X_scaled, k_range, seeds)

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    rider_agg["segment"] = km.fit_predict(X_scaled)

    # Segment table
    segment_table = rider_agg.groupby("segment").agg(
        n_riders=("Delivery_person_ID", "count"),
        mean_delivery_time=("mean_time", "mean"),
        mean_late_rate=("late_rate", "mean"),
        mean_rating=("mean_rating", "mean"),
        mean_orders=("n_orders", "mean"),
        mean_distance=("mean_distance", "mean"),
    ).reset_index()

    # Auto-label segments
    labels = []
    for _, row in segment_table.iterrows():
        if row["mean_late_rate"] < 0.15 and row["mean_rating"] > 4.5:
            labels.append("Star Performer")
        elif row["mean_late_rate"] > 0.4:
            labels.append("Needs Improvement")
        elif row["mean_distance"] > segment_table["mean_distance"].median():
            labels.append("Long-Distance Specialist")
        elif row["mean_orders"] > segment_table["mean_orders"].median():
            labels.append("High-Volume Rider")
        else:
            labels.append("Average Rider")
    segment_table["segment_name"] = labels

    return rider_agg, segment_table, elbow_df, best_k, km, scaler


def save_clustering_outputs(zone_table, store_scorecard, zone_elbow,
                             segment_table, rider_agg, rider_elbow):
    """Save all clustering outputs."""

    (ROOT / "results").mkdir(parents=True, exist_ok=True)

    zone_table.to_csv(ROOT / "results" / "zones_table.csv", index=False)
    store_scorecard.to_csv(ROOT / "results" / "store_scorecard.csv", index=False)
    zone_elbow.to_csv(ROOT / "results" / "zone_elbow.csv", index=False)

    segment_table.to_csv(ROOT / "results" / "rider_segments.csv", index=False)
    rider_agg.to_csv(ROOT / "results" / "rider_agg.csv", index=False)
    rider_elbow.to_csv(ROOT / "results" / "rider_elbow.csv", index=False)
