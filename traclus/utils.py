import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as tslearn_dtw
import similaritymeasures
from streamlit_folium import folium_static
from sklearn.metrics import silhouette_score
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, time, timedelta

EARTH_RADIUS_KM = 6371.0

# --- Data Loading & Initial Cleaning ---

# @st.cache_data # Caching initial load might be okay, but requires careful keying if file content changes
def upload_data(uploaded_file_content):
    """
    Loads CSV data, handles specific formats like leading unnamed columns,
    maps common column names, converts types, and performs basic validation.
    """
    try:
        df = pd.read_csv(uploaded_file_content)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # --- Format Handling ---
    # 1. Handle leading unnamed index column (common in pandas to_csv)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
        st.write("Detected and removed leading unnamed index column.")

    # 2. Flexible Column Mapping (lowercase for robustness)
    column_mapping = {
        # DateTime variations
        "timestamp": "DateTime", "date and time": "DateTime", "datetime": "DateTime",
        # TaxiID variations
        "taxiid": "TaxiID", "driveid": "TaxiID", "vehicleid": "TaxiID", "driverno": "TaxiID", "driveno": "TaxiID",
        # Longitude variations
        "longitude": "Longitude", "lon": "Longitude",
        # Latitude variations
        "latitude": "Latitude", "lat": "Latitude",
    }
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.lower().map(lambda x: column_mapping.get(x, x.lower())) # Keep original name if no map
    mapped_columns = df.columns.tolist()
    if original_columns != mapped_columns:
         st.write(f"Column names mapped (showing final names): {mapped_columns}")

    # 3. Check for Required Columns AFTER mapping
    required_cols = ["DateTime", "TaxiID", "Latitude", "Longitude"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns after mapping: {', '.join(missing_cols)}. Cannot proceed.")
        return None

    # --- Type Conversion & Cleaning ---
    initial_rows = len(df)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['TaxiID'] = pd.to_numeric(df['TaxiID'], errors='coerce').fillna(-1).astype(int)

    # Handle Timezones (Convert to UTC if present)
    if pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        if df["DateTime"].dt.tz is not None:
            df["DateTime"] = df["DateTime"].dt.tz_convert('UTC')
            # st.info("Timezone detected and converted to UTC.") # Reduce messages
        # else: st.info("No timezone info found.")
    else: st.warning("Could not parse DateTime column.")

    # Drop rows with conversion errors or invalid TaxiID
    df = df.dropna(subset=["DateTime", "Latitude", "Longitude"])
    df = df[df['TaxiID'] != -1]
    rows_after_na = len(df)
    if rows_after_na < initial_rows:
         st.write(f"Removed {initial_rows - rows_after_na} rows with missing/invalid critical data.")

    # --- Validation ---
    if df.empty:
        st.warning("No valid data remaining after cleaning.")
        return None

    # Optional: Check Lat/Lon ranges (can add warnings if needed)
    # lat_invalid = df[(df['Latitude'] < -90) | (df['Latitude'] > 90)]
    # lon_invalid = df[(df['Longitude'] < -180) | (df['Longitude'] > 180)]
    # if not lat_invalid.empty or not lon_invalid.empty:
    #     st.warning("Some Lat/Lon values are outside standard ranges.")

    st.success(f"Loaded {len(df)} valid data points.")
    return df


# --- Filtering Functions (Not Cached - Rerun on filter changes) ---

def filter_data_by_date(df, selected_dates):
    """Filters DataFrame by selected list of dates."""
    if df is None or df.empty: return pd.DataFrame()
    if not selected_dates: return df.copy()
    if "DateTime" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        return df.copy() # Skip if column invalid
    dates_to_filter = [pd.Timestamp(d).date() for d in selected_dates]
    return df[df['DateTime'].dt.date.isin(dates_to_filter)].copy()

def filter_data_by_time_mode(df, mode="All Day (0-24)", custom_hour_range=(0, 24)):
    """Filters DataFrame by time of day."""
    if df is None or df.empty: return pd.DataFrame()
    if "DateTime" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        return df.copy() # Skip if column invalid

    dt = df['DateTime'].dt
    if mode == "Daytime (6-18)":
        return df[(dt.hour >= 6) & (dt.hour < 18)].copy()
    elif mode == "Nighttime (<6 | >=18)":
        return df[(dt.hour < 6) | (dt.hour >= 18)].copy()
    elif mode == "Custom Range...":
        start, end = custom_hour_range
        if start == 0 and end == 24: return df.copy() # No filter needed
        if end == 24: return df[dt.hour >= start].copy() # Handle end=24
        return df[(dt.hour >= start) & (dt.hour < end)].copy()
    else: # All Day
        return df.copy()


# --- Feature Calculation & Anomaly Filtering (Cached) ---

@st.cache_data
def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two points using Haversine."""
    if pd.isna(lon1) or pd.isna(lat1) or pd.isna(lon2) or pd.isna(lat2): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * EARTH_RADIUS_KM * 1000

@st.cache_data
def calculate_trajectory_features(_df):
    """Adds TimeDiff_s, DistJump_m, Speed_kmh to the DataFrame."""
    if _df is None or _df.empty: return pd.DataFrame()
    if not all(col in _df.columns for col in ["TaxiID", "DateTime", "Latitude", "Longitude"]):
        st.error("Cannot calculate features: Missing required columns.")
        return _df.copy() # Return original if columns missing

    df = _df.sort_values(['TaxiID', 'DateTime']).copy()
    gb = df.groupby('TaxiID')
    df['TimeDiff_s'] = gb['DateTime'].diff().dt.total_seconds()
    df['PrevLat'] = gb['Latitude'].shift(1)
    df['PrevLon'] = gb['Longitude'].shift(1)

    valid_prev = df['PrevLat'].notna()
    df['DistJump_m'] = np.nan
    # Apply haversine only on valid rows (avoids errors, slightly slower than full vectorization if possible)
    df.loc[valid_prev, 'DistJump_m'] = df[valid_prev].apply(
        lambda r: haversine(r['PrevLon'], r['PrevLat'], r['Longitude'], r['Latitude']), axis=1)

    df['Speed_kmh'] = np.nan
    valid_speed = valid_prev & (df['TimeDiff_s'] > 1e-6) & df['DistJump_m'].notna()
    df.loc[valid_speed, 'Speed_kmh'] = (df.loc[valid_speed, 'DistJump_m'] / df.loc[valid_speed, 'TimeDiff_s']) * 3.6

    return df.drop(columns=['PrevLat', 'PrevLon'])

@st.cache_data
def filter_invalid_moves(_df_with_features, max_speed_kmh=150, max_dist_jump_m=5000):
    """Filters points based on speed and distance jump thresholds."""
    if _df_with_features is None or _df_with_features.empty: return pd.DataFrame()
    if 'Speed_kmh' not in _df_with_features.columns or 'DistJump_m' not in _df_with_features.columns:
        st.warning("Features missing, cannot filter invalid moves.")
        return _df_with_features.copy()

    invalid = (
        (_df_with_features['Speed_kmh'] > max_speed_kmh) |
        (_df_with_features['DistJump_m'] > max_dist_jump_m)
    ).fillna(False) # Treat NaN as valid (not invalid)

    valid_df = _df_with_features[~invalid].copy()
    num_removed = len(_df_with_features) - len(valid_df)
    if num_removed > 0: st.write(f"Filtered {num_removed} points by speed/distance.")
    return valid_df


# --- Data Preprocessing for Clustering (Cached) ---

@st.cache_data
def preprocess_data(_filtered_df):
    """
    Filters trajectories to have >= 2 points, sorts, and extracts
    data into list of numpy arrays and a corresponding processed DataFrame.
    """
    if _filtered_df is None or _filtered_df.empty: return [], pd.DataFrame(), []
    required = ["DateTime", "TaxiID", "Latitude", "Longitude"]
    if not all(col in _filtered_df.columns for col in required): return [], pd.DataFrame(), []

    # Filter groups (trajectories) with at least 2 points
    grouped = _filtered_df.groupby('TaxiID')
    processed_df = grouped.filter(lambda x: len(x) >= 2)
    if processed_df.empty: return [], pd.DataFrame(), []

    # Sort within groups and reset index
    processed_df = processed_df.groupby('TaxiID', group_keys=False)\
                               .apply(lambda x: x.sort_values('DateTime'))\
                               .reset_index(drop=True)

    # Extract trajectory data ([lat, lon] arrays) and IDs
    traj_data = processed_df.groupby('TaxiID')[['Latitude', 'Longitude']]\
                            .apply(lambda x: x.values).tolist()
    traj_ids = processed_df['TaxiID'].unique().tolist() # Order matches traj_data

    return traj_data, processed_df, traj_ids


# --- Distance Matrix Calculation (Cached) ---

@st.cache_data(show_spinner=False)
def calculate_distance_matrix(_trajectory_data, metric='dtw'):
    """Calculates pairwise distance matrix between trajectories."""
    n = len(_trajectory_data)
    if n < 2: return np.array([])
    dist_matrix = np.full((n, n), np.inf) # Initialize with infinity
    np.fill_diagonal(dist_matrix, 0)

    # Pre-convert trajectories and check validity
    trajectories_np = []
    valid_indices = []
    for i, t in enumerate(_trajectory_data):
        arr = np.array(t, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
            trajectories_np.append(arr)
            valid_indices.append(i)
        else:
            trajectories_np.append(None) # Placeholder for invalid traj

    num_invalid = n - len(valid_indices)
    if num_invalid > 0: st.warning(f"Skipping {num_invalid} invalid trajectories in distance calculation.")

    # Calculate upper triangle using valid trajectories only
    for i_idx, orig_i in enumerate(valid_indices):
        for j_idx in range(i_idx + 1, len(valid_indices)):
            orig_j = valid_indices[j_idx]
            traj_i = trajectories_np[orig_i]
            traj_j = trajectories_np[orig_j]
            dist = np.inf
            try:
                if metric == 'dtw': dist = tslearn_dtw(traj_i, traj_j)
                elif metric == 'frechet': dist = similaritymeasures.frechet_dist(traj_i, traj_j)
                elif metric == 'hausdorff': dist = similaritymeasures.hausdorff_dist(traj_i, traj_j)
                else: dist = np.inf # Unknown metric
                dist = dist if np.isfinite(dist) else np.inf # Ensure finite or inf
            except Exception: dist = np.inf # Error during calc -> inf
            dist_matrix[orig_i, orig_j] = dist_matrix[orig_j, orig_i] = dist

    # Handle infinities (replace with 1.1 * max finite or large number)
    finite_vals = dist_matrix[np.isfinite(dist_matrix)]
    if not np.all(np.isfinite(dist_matrix)):
        max_finite = np.max(finite_vals) if finite_vals.size > 0 else 0
        replace_val = max_finite * 1.1 if max_finite > 0 else 1e9
        dist_matrix[~np.isfinite(dist_matrix)] = replace_val
        # st.write("Replaced non-finite distances in matrix.") # Less verbose

    return dist_matrix


# --- Stop Detection (Cached) ---
@st.cache_data
def detect_stops(_df_single_taxi, speed_thresh_kmh=3.0, min_duration_s=120):
    """Identifies stops in a single taxi's trajectory (requires features)."""
    if _df_single_taxi is None or _df_single_taxi.empty: return pd.DataFrame()
    if 'Speed_kmh' not in _df_single_taxi.columns or 'TimeDiff_s' not in _df_single_taxi.columns:
        st.warning("Stop detection requires Speed_kmh and TimeDiff_s.")
        df_out = _df_single_taxi.copy(); df_out['IsStop'] = False
        return df_out

    df = _df_single_taxi.sort_values('DateTime').copy()
    df['IsStop'] = False
    start_idx, current_duration = None, 0.0

    for i in range(len(df)):
        speed = df['Speed_kmh'].iloc[i]
        time_diff = df['TimeDiff_s'].iloc[i] # Duration *since previous point*

        is_slow = pd.notna(speed) and speed < speed_thresh_kmh
        is_valid_interval = pd.notna(time_diff) and time_diff > 0

        if is_slow and is_valid_interval:
            if start_idx is None: start_idx = i # Mark start of potential stop
            current_duration += time_diff # Accumulate duration
        else:
            # End of potential stop sequence (or invalid interval)
            if start_idx is not None and current_duration >= min_duration_s:
                df.loc[df.index[start_idx], 'IsStop'] = True # Mark the START point of the stop
            start_idx, current_duration = None, 0.0 # Reset

    # Check if trajectory ends in a stop
    if start_idx is not None and current_duration >= min_duration_s:
        df.loc[df.index[start_idx], 'IsStop'] = True

    return df


# --- Clustering Analysis Helpers ---

# @st.cache_data # Caching might be okay if inputs rarely change for same data
def find_dbscan_prototypes(_traj_data, _labels, _dist_matrix):
    """Finds representative trajectory index for each DBSCAN cluster."""
    prototypes = {}
    unique_labels = sorted([lbl for lbl in set(_labels) if lbl != -1])
    if _dist_matrix is None or _dist_matrix.size == 0: return prototypes

    for label in unique_labels:
        cluster_indices = np.where(_labels == label)[0]
        if len(cluster_indices) == 0: continue
        if len(cluster_indices) == 1: prototypes[label] = cluster_indices[0]; continue

        try:
            cluster_dist_submatrix = _dist_matrix[np.ix_(cluster_indices, cluster_indices)]
            if not np.all(np.isfinite(cluster_dist_submatrix)): continue # Skip if internal distances invalid
            sum_dists = cluster_dist_submatrix.sum(axis=1)
            # Find index within the *submatrix* with min sum
            min_sum_sub_idx = np.argmin(sum_dists)
            # Map back to original index
            prototypes[label] = cluster_indices[min_sum_sub_idx]
        except Exception: pass # Ignore errors finding prototype for a cluster
    return prototypes

def calculate_silhouette(dist_matrix, labels):
    """Calculates silhouette score, returns None if invalid."""
    if labels is None or dist_matrix is None or dist_matrix.size == 0: return None
    n_labels = len(set(labels))
    n_samples = len(labels)
    # Score defined for 2 <= k <= n-1 clusters
    if 2 <= n_labels < n_samples:
        if dist_matrix.shape == (n_samples, n_samples) and np.all(np.isfinite(dist_matrix)):
            try: return silhouette_score(dist_matrix, labels, metric='precomputed')
            except ValueError: return None # Handle cases like all points in one cluster
    return None # Default return if conditions not met

@st.cache_data
def calculate_cluster_summary(_processed_df_labeled, _traj_data, _traj_ids, _labels):
    """Generates summary statistics for each valid cluster."""
    summary = []
    if _processed_df_labeled is None or _traj_data is None or _traj_ids is None or _labels is None or \
       len(_traj_ids) != len(_labels) or len(_traj_data) != len(_labels):
        return pd.DataFrame() # Return empty if inputs inconsistent

    id_to_idx = {tid: i for i, tid in enumerate(_traj_ids)}
    unique_labels = sorted([lbl for lbl in set(_labels) if lbl != -1])

    for label in unique_labels:
        cluster_df = _processed_df_labeled[_processed_df_labeled['ClusterLabel'] == label]
        if cluster_df.empty: continue
        cluster_traj_ids = cluster_df['TaxiID'].unique()
        cluster_indices = [id_to_idx[tid] for tid in cluster_traj_ids if tid in id_to_idx]
        if not cluster_indices: continue

        n_traj = len(cluster_indices)
        points = [len(_traj_data[i]) for i in cluster_indices if i < len(_traj_data)]
        avg_pts = np.mean(points) if points else 0
        earliest, latest = cluster_df['DateTime'].min(), cluster_df['DateTime'].max()

        durations = []
        for tid in cluster_traj_ids:
            traj_pts = cluster_df[cluster_df['TaxiID'] == tid]
            if len(traj_pts) >= 2: durations.append((traj_pts['DateTime'].max() - traj_pts['DateTime'].min()).total_seconds())
        avg_dur = np.mean(durations) if durations else np.nan

        avg_spd = cluster_df['Speed_kmh'].mean() if 'Speed_kmh' in cluster_df.columns else np.nan

        summary.append({
            "Cluster": label, "Num Trajectories": n_traj, "Avg Points/Traj": f"{avg_pts:.1f}",
            "Earliest Point": earliest, "Latest Point": latest,
            "Avg Duration (s)": f"{avg_dur:.0f}" if pd.notna(avg_dur) else "N/A",
            "Avg Speed (km/h)": f"{avg_spd:.1f}" if pd.notna(avg_spd) else "N/A",
        })

    if not summary: return pd.DataFrame()
    summary_df = pd.DataFrame(summary).set_index("Cluster")
    # Format datetime columns safely
    time_fmt = '%Y-%m-%d %H:%M' # Shorter format
    for col in ['Earliest Point', 'Latest Point']:
         if pd.api.types.is_datetime64_any_dtype(summary_df[col]):
             summary_df[col] = summary_df[col].dt.strftime(time_fmt).fillna("N/A")
         else: summary_df[col] = "N/A"
    return summary_df


# --- Visualization Functions ---

def display_stats(raw_len, processed_df, traj_data):
    """Displays key data statistics."""
    n_traj = len(traj_data) if traj_data else 0
    n_proc_pts = len(processed_df) if processed_df is not None else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Points", f"{raw_len:,}" if raw_len is not None else "N/A")
    c2.metric("Valid Trajectories", f"{n_traj:,}")
    c3.metric("Points Processed", f"{n_proc_pts:,}")

    if processed_df is not None and not processed_df.empty and n_traj > 0:
        pts = [len(t) for t in traj_data]
        st.write(f"**Pts/Traj (Min/Avg/Max):** {min(pts):,} / {np.mean(pts):.1f} / {max(pts):,}")
        if pd.api.types.is_datetime64_any_dtype(processed_df["DateTime"]):
             t_min, t_max = processed_df["DateTime"].min(), processed_df["DateTime"].max()
             fmt = '%Y-%m-%d %H:%M'
             st.write(f"**Time Range:** {t_min.strftime(fmt)} to {t_max.strftime(fmt)}")

def visualize_1(df_features): # Folium Map
    """Folium map showing individual trajectories."""
    st.subheader("Trajectories (Folium)")
    if df_features is None or df_features.empty: st.info("No data for Folium map."); return
    try: center = [df_features['Latitude'].mean(), df_features['Longitude'].mean()]
    except: center = [39.9, 116.4]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
    ids = df_features['TaxiID'].unique()
    cmap = plt.cm.get_cmap('tab20', len(ids))
    colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(ids))]
    id_colors = {tid: colors[i % len(colors)] for i, tid in enumerate(ids)}

    for tid, group in df_features.groupby('TaxiID'):
        g_sort = group.sort_values("DateTime")
        if len(g_sort) >= 2:
            route = list(zip(g_sort["Latitude"], g_sort["Longitude"]))
            popup = f"ID: {tid}<br>Pts: {len(route)}"
            if 'Speed_kmh' in g_sort:
                 avg_spd = g_sort['Speed_kmh'].mean()
                 if pd.notna(avg_spd): popup += f"<br>Avg Spd: {avg_spd:.1f} km/h"
            folium.PolyLine(route, color=id_colors.get(tid,'gray'), weight=2, opacity=0.7, popup=popup).add_to(m)
    folium_static(m, height=600)

def visualize_2(processed_df): # Plotly Map
    """Plotly map showing individual trajectories."""
    st.subheader("Trajectories (Plotly)")
    if processed_df is None or processed_df.empty or len(processed_df)<2: st.info("No data for Plotly map."); return
    df_sort = processed_df.sort_values(["TaxiID", "DateTime"])
    try:
        fig = px.line_mapbox(df_sort, lat="Latitude", lon="Longitude", color="TaxiID",
                             mapbox_style="open-street-map", zoom=10, height=600)
        fig.update_layout(margin=dict(l=5,r=5,t=5,b=5), showlegend=False,
                          mapbox_center_lat = df_sort['Latitude'].mean(),
                          mapbox_center_lon = df_sort['Longitude'].mean())
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Plotly map error: {e}")

def visualize_heatmap(df_features):
    """Density heatmap of points."""
    st.subheader("Density Heatmap")
    if df_features is None or df_features.empty: st.info("No data for heatmap."); return
    heat_data = df_features[['Latitude', 'Longitude']].dropna().values.tolist()
    if not heat_data: st.warning("No valid points for heatmap."); return
    try: center = [df_features['Latitude'].mean(), df_features['Longitude'].mean()]
    except: center = [39.9, 116.4]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
    HeatMap(heat_data, radius=10, blur=7).add_to(m)
    folium_static(m, height=500)

def visualize_clusters(proc_df, traj_data, labels, traj_ids, proto_indices=None):
    """Folium map visualizing clustered trajectories."""
    st.subheader("Clustered Trajectories")
    if not traj_data or labels is None or traj_ids is None or \
       len(traj_data) != len(labels) or len(traj_ids) != len(labels):
        st.warning("Invalid input for cluster visualization."); return
    try: center = [proc_df['Latitude'].mean(), proc_df['Longitude'].mean()] if proc_df is not None else [39.9, 116.4]
    except: center = [39.9, 116.4]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    u_labels = sorted([lbl for lbl in set(labels) if lbl != -1])
    n_clusters = len(u_labels)
    if n_clusters > 0:
        cmap = plt.cm.get_cmap('viridis', n_clusters)
        colors = {lbl: matplotlib.colors.rgb2hex(cmap(i / max(1, n_clusters -1))) for i, lbl in enumerate(u_labels)}
    else: colors = {}
    colors[-1] = '#808080' # Noise color

    layers = {}
    for lbl in u_labels + [-1]:
        name = f"Cluster {lbl}" if lbl != -1 else "Noise"
        layers[name] = folium.FeatureGroup(name=name, show=True).add_to(m)

    lbl_to_proto = {lbl: idx for lbl, idx in proto_indices.items()} if proto_indices else {}

    for i, traj in enumerate(traj_data):
        lbl = labels[i]; tid = traj_ids[i]
        name = f"Cluster {lbl}" if lbl != -1 else "Noise"
        color = colors.get(lbl, '#808080')
        is_proto = (lbl != -1 and lbl_to_proto.get(lbl) == i)
        folium_traj = np.array(traj).tolist()
        if len(folium_traj) >= 2:
            folium.PolyLine(locations=folium_traj, color=color,
                            weight=4 if is_proto else 2, opacity=1.0 if is_proto else 0.6,
                            popup=f"ID: {tid}<br>Cluster: {lbl}{' (Proto)' if is_proto else ''}"
                           ).add_to(layers[name])
    if layers: folium.LayerControl(collapsed=False).add_to(m)
    folium_static(m, height=600)

def plot_cluster_sizes(labels):
    """Bar chart of cluster sizes."""
    if labels is None or len(labels) == 0: return
    st.subheader("Cluster Sizes")
    counts = pd.Series(labels).value_counts().sort_index()
    counts.index = counts.index.map(lambda x: f"Cluster {x}" if x != -1 else "Noise")
    if not counts.empty:
        sort_key = lambda x: int(x.split()[-1]) if x != "Noise" else float('inf')
        sorted_cats = sorted(counts.index, key=sort_key)
        fig = px.bar(counts, x=counts.index, y=counts.values, text_auto=True,
                     labels={'x': 'Cluster', 'y': 'Count'}, title="Trajectories per Cluster")
        fig.update_layout(xaxis_title=None, yaxis_title="Count",
                          xaxis={'categoryorder': 'array', 'categoryarray': sorted_cats})
        st.plotly_chart(fig, use_container_width=True)

def visualize_single_trajectory_animation_plotly(df_single_taxi_with_stops, speed_multiplier=1.0):
    """Enhanced Plotly animation for a single taxi with speed control and info display."""
    if df_single_taxi_with_stops is None or df_single_taxi_with_stops.empty:
        st.warning("No data for selected taxi animation.")
        return None # Return None if no data
    required = ['DateTime', 'Latitude', 'Longitude', 'TaxiID', 'IsStop', 'Speed_kmh']
    if not all(col in df_single_taxi_with_stops.columns for col in required):
        st.error("Animation failed: Missing required columns."); return None

    df_anim = df_single_taxi_with_stops.copy()
    df_anim['DateTime'] = pd.to_datetime(df_anim['DateTime'], errors='coerce')
    df_anim = df_anim.dropna(subset=['DateTime', 'Latitude', 'Longitude']).sort_values('DateTime').reset_index(drop=True)

    if len(df_anim) < 2: st.warning("Need >= 2 points for animation."); return None

    taxi_id = df_anim['TaxiID'].iloc[0]
    # Subheader bây giờ sẽ được đặt trong app.py để tránh in nhiều lần khi rerun
    # st.subheader(f"Animation for TaxiID: {taxi_id}")
    df_anim['Frame'] = df_anim.index
    df_anim['Speed_kmh_display'] = df_anim['Speed_kmh'].fillna(0.0)

    lat_min, lat_max = df_anim['Latitude'].min(), df_anim['Latitude'].max()
    lon_min, lon_max = df_anim['Longitude'].min(), df_anim['Longitude'].max()
    lat_pad = (lat_max - lat_min) * 0.1 or 0.02
    lon_pad = (lon_max - lon_min) * 0.1 or 0.02
    bounds = {"west": lon_min-lon_pad, "east": lon_max+lon_pad, "south": lat_min-lat_pad, "north": lat_max+lat_pad}
    center = dict(lat=(lat_min+lat_max)/2, lon=(lon_min+lon_max)/2)
    base_frame_duration_ms = 150
    # Điều chỉnh frame_duration để tốc độ trực quan hơn
    # Tốc độ 1x -> duration base, 2x -> duration / 2, 0.5x -> duration * 2
    frame_duration = max(20, int(base_frame_duration_ms / speed_multiplier))
    # Giảm transition để chuyển động tức thời hơn
    transition_duration = 0 # max(10, int(frame_duration * 0.2)) # Set to 0 for immediate jumps

    # --- Create Plot ---
    # Tạo figure mới mỗi lần gọi hàm để đảm bảo không dùng lại state cũ
    fig = go.Figure()

    # Trace 0: Static Full Path Line (RED)
    fig.add_trace(go.Scattermapbox(lat=df_anim["Latitude"], lon=df_anim["Longitude"], mode='lines', line=dict(width=2, color='rgba(255, 0, 0, 0.7)'), hoverinfo='none', name="Full Path"))
    # Trace 1: Static Markers for Stops
    stops_df = df_anim[df_anim['IsStop']]
    fig.add_trace(go.Scattermapbox(lat=stops_df["Latitude"], lon=stops_df["Longitude"], mode='markers', marker=dict(size=8, color='purple', symbol='circle'), name='Detected Stops', hoverinfo='text', text=[f"Stop @ {dt.strftime('%H:%M:%S')}" for dt in stops_df['DateTime']]))
    # Trace 2: Start Marker
    fig.add_trace(go.Scattermapbox(lat=[df_anim["Latitude"].iloc[0]], lon=[df_anim["Longitude"].iloc[0]], mode='markers', marker=dict(size=14, color='green', symbol='circle-dot'), name='Start', hoverinfo='text', text=f"Start: {df_anim['DateTime'].iloc[0].strftime('%H:%M:%S')}"))
    # Trace 3: End Marker
    fig.add_trace(go.Scattermapbox(lat=[df_anim["Latitude"].iloc[-1]], lon=[df_anim["Longitude"].iloc[-1]], mode='markers', marker=dict(size=14, color='black', symbol='x-dot'), name='End', hoverinfo='text', text=f"End: {df_anim['DateTime'].iloc[-1].strftime('%H:%M:%S')}"))
    # Trace 4: THE MOVING MARKER (Initially at start)
    fig.add_trace(go.Scattermapbox(lat=[df_anim["Latitude"].iloc[0]], lon=[df_anim["Longitude"].iloc[0]], mode='markers', marker=dict(size=16, color='blue', symbol='arrow', allowoverlap=True), name='Current Position', customdata=df_anim[['DateTime', 'Speed_kmh_display']].iloc[[0]], hovertemplate="<b>Current:</b><br>Time: %{customdata[0]|%Y-%m-%d %H:%M:%S}<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>"))

    # --- Create Frames ---
    frames = []
    for k in range(len(df_anim)):
        point_data = df_anim.iloc[k]
        dt_str = point_data['DateTime'].strftime('%Y-%m-%d %H:%M:%S')
        speed_str = f"{point_data['Speed_kmh_display']:.1f}"
        info_text = f"Time: {dt_str}<br>Speed: {speed_str} km/h"

        # Dữ liệu cho trace 4 trong frame này
        frame_trace_data = go.Scattermapbox(
            lat=[point_data["Latitude"]], lon=[point_data["Longitude"]],
            mode='markers', marker=dict(size=16, color='blue', symbol='arrow', allowoverlap=True),
            name='Current Position',
            customdata=[[point_data['DateTime'], point_data['Speed_kmh_display']]],
            hovertemplate="<b>Current:</b><br>Time: %{customdata[0]|%Y-%m-%d %H:%M:%S}<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>"
        )

        frames.append(go.Frame(
            name=str(k),
            data=[frame_trace_data], # Chỉ chứa data cho trace cần update
            traces=[4],            # Chỉ định index của trace cần update (moving marker)
            layout=go.Layout(      # Layout chỉ chứa phần cần update (annotation)
                 annotations=[dict(
                    text=info_text, align='left', showarrow=False, xref='paper', yref='paper',
                    x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', borderpad=4
                )]
            )
        ))
    # Gán frames sau khi tạo xong figure ban đầu
    fig.frames = frames

    # --- Configure Layout & Animation Controls ---
    initial_point = df_anim.iloc[0]
    initial_dt_str = initial_point['DateTime'].strftime('%Y-%m-%d %H:%M:%S')
    initial_speed_str = f"{initial_point['Speed_kmh_display']:.1f}"
    initial_info_text = f"Time: {initial_dt_str}<br>Speed: {initial_speed_str} km/h"

    fig.update_layout(
        # title=f"Movement Simulation for Taxi {taxi_id}", # Tiêu đề đặt ở app.py
        height=700, mapbox_style="carto-positron",
        mapbox_bounds=bounds, mapbox_center=center, mapbox_zoom=12, margin=dict(r=5, t=10, l=5, b=5), # Giảm margin top
        showlegend=True, legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'),
        annotations=[dict(text=initial_info_text, name="info_annotation", align='left', showarrow=False, xref='paper', yref='paper', x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', borderpad=4)],
        # Sử dụng 'immediate' mode và transition duration = 0 để nhảy điểm tức thời
        updatemenus=[dict(
            type="buttons", direction="right", showactive=False, x=0.95, y=0.05, xanchor="right", yanchor="bottom",
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True, "transition": {"duration": transition_duration}, "mode": "immediate"}]), # Immediate mode
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )],
        sliders=[dict(
            active=0, currentvalue={"prefix": "Point Index: ", "font": {"size": 14}}, pad={"b": 10, "t": 10}, len=0.8, x=0.1, y=0.05, yanchor="bottom",
             # args trong step cũng dùng immediate mode và transition=0
            steps=[{"args": [[f.name], {"frame": {"duration": frame_duration, "redraw": True}, "mode": "immediate", "transition": {"duration": transition_duration}}],
                    "label": str(k), "method": "animate"} for k, f in enumerate(fig.frames)]
        )]
    )

    return fig # Return the figure object

def visualize_congestion(df_features, speed_thresh_kmh=10):
    """Visualizes congestion points using Folium MarkerCluster."""
    st.subheader(f"Congestion Hotspots (< {speed_thresh_kmh} km/h)")
    if df_features is None or df_features.empty or 'Speed_kmh' not in df_features.columns:
        st.info("No speed data for congestion analysis."); return

    congestion = df_features[(df_features['Speed_kmh'].notna()) & (df_features['Speed_kmh'] < speed_thresh_kmh)].copy()
    if congestion.empty: st.info(f"No points below {speed_thresh_kmh} km/h found."); return

    st.write(f"Found **{len(congestion):,}** potential congestion points.")
    try: center = [congestion['Latitude'].mean(), congestion['Longitude'].mean()]
    except: center = [39.9, 116.4]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # Use MarkerCluster for performance
    mc = MarkerCluster(name="Congestion Points").add_to(m)
    for idx, r in congestion.iterrows():
         popup = (f"Time: {r['DateTime'].strftime('%H:%M:%S')}<br>"
                  f"Speed: {r['Speed_kmh']:.1f} km/h<br>ID: {r['TaxiID']}")
         folium.CircleMarker(location=[r['Latitude'], r['Longitude']], radius=4,
                             color='orange', fill=True, fill_color='red', fill_opacity=0.6,
                             popup=popup).add_to(mc)
    # folium.LayerControl().add_to(m) # Optional layer control
    folium_static(m, height=500)