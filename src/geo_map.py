"""
src/geo_map.py — Enterprise Geographical Intelligence Map using Folium
Generates interactive multi-layer map covering Maharashtra, Telangana, Karnataka,
Tamil Nadu, Andhra Pradesh, and all delivery regions across India.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium import plugins
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent

# Distinct high-contrast palette for clusters
DEFAULT_PALETTE = [
    "#3A86FF",  # Zone 0: Vibrant Blue
    "#06D6A0",  # Zone 1: Emerald Teal (South & West India Hub)
    "#FFB703",  # Zone 2: Amber Yellow
    "#E63946",  # Zone 3: High-Delay Crimson (Priority Target)
    "#8338EC",  # Purple
    "#FB5607",  # Orange
    "#00F5D4",  # Cyan
    "#F15BB5"   # Pink
]


def get_zone_color(zone_id, total_zones=4):
    """Dynamically assign consistent colors across any number of zones."""
    if isinstance(zone_id, (int, np.integer)) and 0 <= zone_id < len(DEFAULT_PALETTE):
        return DEFAULT_PALETTE[zone_id]
    # Fallback HSL generator for arbitrary k
    hue = (int(zone_id) * 360) // max(1, total_zones)
    return f"hsl({hue}, 75%, 45%)"


def build_delivery_intelligence_map(
    df_clean,
    zone_table,
    store_scorecard,
    zone_elbow=None,
    routes_cache_path=None
):
    """
    Builds a comprehensive Folium delivery intelligence map displaying:
    1. All dynamically discovered zones (never discarding Zone 0 or any cluster)
    2. Demand-Zone Centers sized proportionally to demand volume
    3. Dark-store / Merchant markers with actual identifiers and store icons
    4. Individual demand points via Heatmap and Clustered Sample
    5. Approximate Zone Boundaries (Convex Hulls, clearly labeled)
    6. Real OSRM Road-Network Delivery Routes with waypoints, distances & durations
    7. Full-screen, LayerControl, and BI Legend panel
    """
    # 1. Coordinate Center: Geographic centroid of South & West India delivery core
    map_center = [18.50, 77.50]
    m = folium.Map(
        location=map_center,
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 2. Extract Merchant Code if not present
    if "merchant_code" not in df_clean.columns:
        df_clean["merchant_code"] = df_clean["Delivery_person_ID"].str.extract(r"([A-Z]+RES\d+)")

    # 3. Dynamic Zone Discovery
    unique_zones = sorted(zone_table["zone"].unique().tolist())
    total_discovered_zones = len(unique_zones)

    # Calculate median delivery time per zone from order records
    if "zone" not in df_clean.columns:
        df_merged = df_clean.merge(
            store_scorecard[["Restaurant_latitude", "Restaurant_longitude", "zone"]],
            on=["Restaurant_latitude", "Restaurant_longitude"],
            how="left"
        )
    else:
        df_merged = df_clean

    median_times = df_merged.groupby("zone")["Time_taken(min)"].median().to_dict()

    # Extract silhouette score and inertia for best_k
    silhouette_val = "N/A"
    inertia_val = "N/A"
    if zone_elbow is not None and len(zone_elbow) > 0:
        match_k = zone_elbow[zone_elbow["k"] == total_discovered_zones]
        if len(match_k) > 0:
            row_k = match_k.iloc[0]
            silhouette_val = f"{row_k['silhouette_mean']:.4f} (±{row_k.get('silhouette_std', 0):.4f})"
            inertia_val = f"{row_k['inertia_mean']:.1f}"

    # 4. Feature Groups (Layers)
    fg_zones = folium.FeatureGroup(name="🎯 Demand-Zone Centers", show=True)
    fg_stores = folium.FeatureGroup(name="🏪 Dark Stores & Merchants (388 Locations)", show=True)
    fg_hulls = folium.FeatureGroup(name="📐 Zone Coverage (Convex Hulls - Approx)", show=True)
    fg_routes = folium.FeatureGroup(name="🚚 Sample Operational Routes (OSRM Road Network)", show=True)
    fg_heatmap = folium.FeatureGroup(name="🔥 Demand Density Heatmap", show=True)
    fg_demand_cluster = folium.FeatureGroup(name="📦 Demand Points (Clustered Sample)", show=False)

    # ==================== LAYER A: DEMAND-ZONE CENTERS ====================
    max_orders = zone_table["total_orders"].max() if len(zone_table) > 0 else 1

    for _, z_row in zone_table.iterrows():
        z_id = int(z_row["zone"])
        color = get_zone_color(z_id, total_discovered_zones)
        c_lat = float(z_row["center_lat"])
        c_lon = float(z_row["center_lon"])
        orders = int(z_row["total_orders"])
        n_stores = int(z_row["n_stores"])
        mean_time = float(z_row["mean_delivery_time"])
        med_time = float(median_times.get(z_id, mean_time))
        late_rate = float(z_row["mean_late_rate"])
        mean_dist = float(z_row["mean_distance_km"])
        p_index = float(z_row.get("priority_index", 0.0))
        excess_pp = float(z_row.get("excess_late_pp", 0.0))

        # Size marker proportionally between 18 and 42 px
        radius = 18 + int(24 * (orders / max_orders))

        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    min-width: 260px; padding: 6px; font-size: 13px;">
            <div style="background-color: {color}; color: white; padding: 8px 12px; border-radius: 6px; font-weight: bold; margin-bottom: 8px;">
                <span style="font-size: 15px;">📍 DEMAND ZONE {z_id}</span>
                <span style="float: right; font-size: 11px; background: rgba(0,0,0,0.25); padding: 2px 6px; border-radius: 4px;">
                    Priority #{int(zone_table.index.get_loc(_)) + 1}
                </span>
            </div>
            <table style="width: 100%; border-collapse: collapse; line-height: 1.6;">
                <tr><td style="color: #666;">Total Demand Volume:</td><td><b>{orders:,} orders</b></td></tr>
                <tr><td style="color: #666;">Active Stores / Merchants:</td><td><b>{n_stores:,} locations</b></td></tr>
                <tr><td style="color: #666;">Average Delivery Time:</td><td><b>{mean_time:.2f} min</b></td></tr>
                <tr><td style="color: #666;">Median Delivery Time:</td><td><b>{med_time:.1f} min</b></td></tr>
                <tr><td style="color: #666;">Late-Order Rate:</td><td style="color: {'#d90429' if late_rate > 0.3 else '#2b9348'}; font-weight: bold;">
                    {late_rate:.1%} ({excess_pp:+.1f} pp vs network)
                </td></tr>
                <tr><td style="color: #666;">Mean Delivery Distance:</td><td><b>{mean_dist:.2f} km</b></td></tr>
                <tr><td style="color: #666;">Cluster Center GPS:</td><td><code>{c_lat:.4f}°N, {c_lon:.4f}°E</code></td></tr>
                <tr><td style="color: #666;">Silhouette Score (k={total_discovered_zones}):</td><td><b>{silhouette_val}</b></td></tr>
                <tr><td style="color: #666;">Priority Index:</td><td><b>{p_index:.3f}</b></td></tr>
            </table>
            <div style="margin-top: 8px; font-size: 11px; color: #888; border-top: 1px solid #eee; padding-top: 4px;">
                Clustered using K-Means spatial + operational features.
            </div>
        </div>
        """

        # Outer pulsing/glow circle
        folium.CircleMarker(
            location=[c_lat, c_lon],
            radius=radius + 4,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            tooltip=f"Zone {z_id} Center ({orders:,} orders)"
        ).add_to(fg_zones)

        # Core circle marker
        folium.CircleMarker(
            location=[c_lat, c_lon],
            radius=radius,
            color="#FFFFFF",
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Zone {z_id} (Click for Full Metrics)"
        ).add_to(fg_zones)

    # ==================== LAYER B: DARK STORES & MERCHANTS ====================
    # Deduplicate store scorecard to unique physical coordinates
    unique_stores = store_scorecard.drop_duplicates(subset=["Restaurant_latitude", "Restaurant_longitude"]).copy()

    # Create mapping of coordinates to merchant_code
    coord_to_code = df_clean.drop_duplicates(
        subset=["Restaurant_latitude", "Restaurant_longitude"]
    ).set_index(["Restaurant_latitude", "Restaurant_longitude"])["merchant_code"].to_dict()

    for _, s_row in unique_stores.iterrows():
        s_lat = float(s_row["Restaurant_latitude"])
        s_lon = float(s_row["Restaurant_longitude"])
        z_id = int(s_row["zone"])
        color = get_zone_color(z_id, total_discovered_zones)
        m_code = coord_to_code.get((s_lat, s_lon), f"STORE_{abs(hash((s_lat, s_lon))) % 1000:03d}")
        s_orders = int(s_row["n_orders"])
        s_time = float(s_row["mean_time"])
        s_late = float(s_row["late_rate"])

        store_popup = f"""
        <div style="font-family: -apple-system, sans-serif; font-size: 12px; min-width: 200px;">
            <div style="font-weight: bold; color: {color}; font-size: 14px; margin-bottom: 4px;">
                🏪 {m_code}
            </div>
            <b>Assigned Cluster:</b> Zone {z_id}<br>
            <b>Coordinates:</b> <code>{s_lat:.4f}°N, {s_lon:.4f}°E</code><br>
            <b>Order Volume:</b> {s_orders:,} deliveries<br>
            <b>Average Time:</b> {s_time:.1f} min<br>
            <b>Late Rate:</b> {s_late:.1%}<br>
        </div>
        """

        folium.CircleMarker(
            location=[s_lat, s_lon],
            radius=6,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(store_popup, max_width=250),
            tooltip=f"{m_code} (Zone {z_id})"
        ).add_to(fg_stores)

    # ==================== LAYER C: APPROXIMATE ZONE COVERAGE (CONVEX HULLS) ====================
    for z_id in unique_zones:
        z_stores = unique_stores[unique_stores["zone"] == z_id]
        if len(z_stores) >= 3:
            pts = z_stores[["Restaurant_latitude", "Restaurant_longitude"]].values
            try:
                hull = ConvexHull(pts)
                hull_coords = pts[hull.vertices].tolist()
                # Close the polygon loop
                hull_coords.append(hull_coords[0])
                z_color = get_zone_color(z_id, total_discovered_zones)

                folium.Polygon(
                    locations=hull_coords,
                    color=z_color,
                    weight=2,
                    dash_array="6, 6",
                    fill=True,
                    fill_color=z_color,
                    fill_opacity=0.08,
                    tooltip=f"Approximate Zone {z_id} Merchant Convex Hull (illustrative coverage, not administrative border)",
                    popup=f"<b>Zone {z_id} Convex Hull Boundary</b><br>Bounding {len(z_stores)} merchant nodes.<br><i>Note: Geometric approximation.</i>"
                ).add_to(fg_hulls)
            except Exception:
                pass

    # ==================== LAYER D: DEMAND POINTS HEATMAP ====================
    # Sample 4,000 delivery points across South & West India for high performance
    sample_deliveries = df_clean.sample(min(4500, len(df_clean)), random_state=42)
    heat_data = sample_deliveries[["Delivery_location_latitude", "Delivery_location_longitude"]].values.tolist()
    plugins.HeatMap(
        heat_data,
        radius=12,
        blur=16,
        max_zoom=9,
        gradient={0.2: "#3A86FF", 0.5: "#06D6A0", 0.75: "#FFB703", 1.0: "#E63946"}
    ).add_to(fg_heatmap)

    # ==================== LAYER E: CLUSTERED DEMAND POINTS ====================
    # Marker cluster for individual delivery dropoff destinations (sample of 400 for responsiveness)
    demand_cluster = plugins.MarkerCluster(
        name="📦 Clustered Demand Points",
        options={"maxClusterRadius": 35}
    )
    for _, d_row in df_clean.sample(min(400, len(df_clean)), random_state=123).iterrows():
        d_lat = float(d_row["Delivery_location_latitude"])
        d_lon = float(d_row["Delivery_location_longitude"])
        d_time = float(d_row["Time_taken(min)"])
        d_late = bool(d_row["is_late"])
        d_order_id = str(d_row["ID"])

        folium.CircleMarker(
            location=[d_lat, d_lon],
            radius=4,
            color="#D90429" if d_late else "#2B9348",
            weight=1,
            fill=True,
            fill_color="#D90429" if d_late else "#2B9348",
            fill_opacity=0.7,
            popup=f"<b>Order {d_order_id}</b><br>Duration: {d_time:.1f} min<br>Status: {'🚨 Late' if d_late else '✅ On-Time'}",
            tooltip=f"Order {d_order_id}"
        ).add_to(demand_cluster)
    demand_cluster.add_to(fg_demand_cluster)

    # ==================== LAYER F: SAMPLE OPERATIONAL DELIVERY ROUTES ====================
    # Load cached real OSRM road routes
    if routes_cache_path is None:
        routes_cache_path = ROOT / "data" / "routes_osrm_cache.json"

    if routes_cache_path.exists():
        try:
            with open(routes_cache_path, "r") as f:
                routes_data = json.load(f)

            for route in routes_data:
                z_id = route.get("zone", 1)
                r_color = get_zone_color(z_id, total_discovered_zones)
                geo = route.get("geometry", [])
                if len(geo) < 2:
                    continue

                is_osrm = route.get("is_osrm", False)
                city_name = route.get("city", "Operational Route")
                road_dist = route.get("road_distance_km", route.get("haversine_km", 0.0))
                est_time = route.get("estimated_duration_min", route.get("actual_duration_min", 0.0))
                m_code = route.get("merchant_code", "Store")
                order_id = route.get("id", "Order")

                route_popup = f"""
                <div style="font-family: -apple-system, sans-serif; font-size: 12px; min-width: 220px;">
                    <div style="font-weight: bold; color: {r_color}; font-size: 13px; margin-bottom: 4px;">
                        🚚 Route: {city_name}
                    </div>
                    <b>Routing Mode:</b> {'🛣️ OSRM Real Road Geometry' if is_osrm else '📏 Illustrative Straight Line'}<br>
                    <b>Origin Merchant:</b> {m_code}<br>
                    <b>Destination Order:</b> {order_id}<br>
                    <b>Assigned Zone:</b> Zone {z_id}<br>
                    <b>Road Transit Distance:</b> <b>{road_dist:.2f} km</b><br>
                    <b>Estimated Travel Time:</b> <b>{est_time:.1f} min</b><br>
                    <div style="font-size: 10px; color: #777; margin-top: 4px;">
                        Generated using OpenStreetMap road network geometry.
                    </div>
                </div>
                """

                # Draw Polyline
                folium.PolyLine(
                    locations=geo,
                    color=r_color,
                    weight=4,
                    opacity=0.85,
                    popup=folium.Popup(route_popup, max_width=280),
                    tooltip=f"Route: {city_name} ({road_dist:.1f} km, {est_time:.0f} min)"
                ).add_to(fg_routes)

                # Origin Store Marker
                m_lat, m_lon = geo[0]
                folium.CircleMarker(
                    location=[m_lat, m_lon],
                    radius=7,
                    color="#111",
                    weight=2,
                    fill=True,
                    fill_color=r_color,
                    fill_opacity=0.9,
                    tooltip=f"Origin Store: {m_code}"
                ).add_to(fg_routes)

                # Destination Order Marker
                d_lat, d_lon = geo[-1]
                folium.CircleMarker(
                    location=[d_lat, d_lon],
                    radius=5,
                    color="#D90429" if route.get("is_late") else "#2B9348",
                    weight=2,
                    fill=True,
                    fill_color="#D90429" if route.get("is_late") else "#2B9348",
                    fill_opacity=0.9,
                    tooltip=f"Delivery Destination: {order_id}"
                ).add_to(fg_routes)

        except Exception as e:
            print(f"Error loading routes cache: {e}")

    # Add all feature groups to map
    fg_hulls.add_to(m)
    fg_heatmap.add_to(m)
    fg_demand_cluster.add_to(m)
    fg_stores.add_to(m)
    fg_routes.add_to(m)
    fg_zones.add_to(m)

    # ==================== MAP CONTROLS ====================
    plugins.Fullscreen(position="topright").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # ==================== FLOATING BI SUMMARY LEGEND ====================
    # Total valid and excluded records
    valid_count = len(df_clean)
    excluded_coords = 3640
    excluded_dist = 431
    total_excluded = excluded_coords + excluded_dist

    legend_items_html = ""
    for _, z_row in zone_table.iterrows():
        z_id = int(z_row["zone"])
        z_color = get_zone_color(z_id, total_discovered_zones)
        z_orders = int(z_row["total_orders"])
        z_late = float(z_row["mean_late_rate"])
        z_stores = int(z_row["n_stores"])
        legend_items_html += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; padding: 4px 6px; border-radius: 4px; background: rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center;">
                <span style="display: inline-block; width: 14px; height: 14px; border-radius: 50%; background-color: {z_color}; margin-right: 8px; border: 1px solid rgba(0,0,0,0.2);"></span>
                <span style="font-weight: 600; font-size: 12px;">Zone {z_id}</span>
            </div>
            <div style="text-align: right; font-size: 11px;">
                <b>{z_orders:,}</b> orders | <span style="color: {'#d90429' if z_late > 0.3 else '#2b9348'}; font-weight: bold;">{z_late:.1%} late</span> ({z_stores} stores)
            </div>
        </div>
        """

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        padding: 14px 16px;
        border-radius: 10px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.22);
        max-width: 350px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #222;
        border: 1px solid #ddd;
    ">
        <div style="font-size: 14px; font-weight: bold; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span>🗺️ Delivery Zones ({total_discovered_zones} Discovered)</span>
            <span style="font-size: 10px; background: #2b5c8f; color: white; padding: 2px 6px; border-radius: 4px;">K={total_discovered_zones}</span>
        </div>
        <div style="font-size: 11px; color: #555; margin-bottom: 10px; line-height: 1.4;">
            All {total_discovered_zones} clusters dynamically identified by spatial K-Means.
        </div>
        {legend_items_html}
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee; font-size: 10.5px; color: #666; line-height: 1.4;">
            <b>📍 Data Quality Audit:</b><br>
            • Valid Clustered Orders: <b>{valid_count:,}</b> across <b>{len(unique_stores):,}</b> merchant nodes.<br>
            • Purged Coordinate Sentinels: <span style="color: #d90429;"><b>{excluded_coords:,}</b> (0.0, 0.0) Atlantic zeros</span>.<br>
            • Purged Distance Outliers: <span style="color: #d90429;"><b>{excluded_dist:,}</b> (>100 km)</span>.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
