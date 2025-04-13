import streamlit as st
import pandas as pd
import numpy as np

from math import radians, cos, sin, asin, sqrt

EARTH_RADIUS_KM = 6371.0

# --- Data Loading & Initial Cleaning ---

def upload_data(uploaded_file_content):
    try:
        df = pd.read_csv(uploaded_file_content)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    df.columns = df.columns.str.strip()
    column_mapping = {
        "TimeStamp": "DateTime", 
        "Date and Time": "DateTime",
        "DriveNo": "TaxiID"
    }
    df.rename(columns=column_mapping, inplace=True)

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors='coerce')
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors='coerce')
    df["TaxiID"] = pd.to_numeric(df["TaxiID"], errors='coerce').fillna(-1).astype(int)

    if pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        df["DateTime"] = df["DateTime"].dt.tz_localize(None, ambiguous='NaT', nonexistent='NaT')
    else:
        st.warning("Could not parse DateTime column.")

    st.success(f"Loaded {len(df)} data points.")
    return df

# --- Filtering Functions (Not Cached - Rerun on filter changes) ---
# @st.cache_data
def filter_data_by_date(df, selected_dates, len_sad):
    """Filters DataFrame by selected list of dates."""
    if len(selected_dates) == len_sad: 
        return df.copy() 
    dates_to_filter = [pd.Timestamp(d).date() for d in selected_dates]
    return df[df['DateTime'].dt.date.isin(dates_to_filter)].copy()

# @st.cache_data
def filter_data_by_hours(df, custom_hour_range=(0, 24)):
    """Filters DataFrame by time of day."""
    dt = df['DateTime'].dt
    start, end = custom_hour_range
    if start == 0 and end == 24: return df.copy() # No filter needed
    if end == 24: return df[dt.hour >= start].copy() 
    return df[(dt.hour >= start) & (dt.hour < end)].copy()

# --- Feature Calculation & Anomaly Filtering (Cached) ---

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two points using Haversine."""
    if pd.isna(lon1) or pd.isna(lat1) or pd.isna(lon2) or pd.isna(lat2): return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * EARTH_RADIUS_KM * 1000

# @st.cache_data
def calculate_trajectory_features(_df):
    """Adds TimeDiff_s, DistJump_m, Speed_kmh to the DataFrame."""
    df = _df.sort_values(['TaxiID', 'DateTime']).copy()
    
    gb = df.groupby('TaxiID')
    df['TimeDiff_s'] = gb['DateTime'].diff().dt.total_seconds()
    df['PrevLat'] = gb['Latitude'].shift(1)
    df['PrevLon'] = gb['Longitude'].shift(1)

    valid_prev = df['PrevLat'].notna()
    df['DistJump_m'] = np.nan
    df.loc[valid_prev, 'DistJump_m'] = df[valid_prev].apply(
        lambda r: haversine(r['PrevLon'], r['PrevLat'], r['Longitude'], r['Latitude']), axis=1)

    df['Speed_kmh'] = np.nan
    valid_speed = valid_prev & (df['TimeDiff_s'] > 1e-6) & df['DistJump_m'].notna()
    df.loc[valid_speed, 'Speed_kmh'] = (df.loc[valid_speed, 'DistJump_m'] / df.loc[valid_speed, 'TimeDiff_s']) * 3.6

    return df.drop(columns=['PrevLat', 'PrevLon'])

# @st.cache_data
def filter_invalid_moves(_df_with_features, max_speed_kmh=150):
    """Filters points based on speed and distance jump thresholds."""
    if 'Speed_kmh' not in _df_with_features.columns:
        st.warning("Features missing, cannot filter invalid moves.")
        return _df_with_features.copy()

    invalid = (
        (_df_with_features['Speed_kmh'] > max_speed_kmh) 
    ).fillna(False) 

    valid_df = _df_with_features[~invalid].copy()
    num_removed = len(_df_with_features) - len(valid_df)
    if num_removed > 0: st.write(f"Filtered {num_removed} points by speed/distance.")
    return valid_df

# --- Data Preprocessing for Clustering (Cached) ---

# @st.cache_data
def preprocess_data(_filtered_df):
    """
    Filters trajectories to have >= 2 points, sorts, and extracts
    data into list of numpy arrays and a corresponding processed DataFrame.
    """
    # Filter groups (trajectories) with at least 2 points
    grouped = _filtered_df.groupby('TaxiID')
    processed_df = grouped.filter(lambda x: len(x) >= 2)

    # Sort within groups and reset index
    processed_df = processed_df.groupby('TaxiID', group_keys=False)\
                               .apply(lambda x: x.sort_values('DateTime'))\
                               .reset_index(drop=True)

    # Extract trajectory data ([lat, lon] arrays) and IDs
    traj_data = processed_df.groupby('TaxiID')[['Latitude', 'Longitude']]\
                            .apply(lambda x: x.values).tolist()
    traj_ids = processed_df['TaxiID'].unique().tolist() # Order matches traj_data
    return traj_data, processed_df, traj_ids

def calculate_summary(_processed_df_labeled, _traj_data, _traj_ids, _labels):
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
    st.write(processed_df)
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