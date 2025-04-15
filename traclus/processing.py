import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2

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
def filter_data_by_hours(df, custom_hour_range=(0, 23)):
    """Filters DataFrame by time of day."""
    dt = df['DateTime'].dt
    start, end = custom_hour_range
    if start == 0 and end == 23: return df.copy() 
    if end == 23: return df[dt.hour >= start].copy() 
    return df[(dt.hour >= start) & (dt.hour <= end)].copy()

# --- Feature Calculation & Anomaly Filtering (Cached) ---
def vincenty_distance(lon1, lat1, lon2, lat2):
    if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
        return np.nan

    a = 6378137.0  # Semi-major axis of Earth (WGS84) in meters
    f = 1/298.257223563  # Flattening of the ellipsoid (WGS84)
    b = (1 - f) * a  # Semi-minor axis

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    L = lon2 - lon1
    U1 = atan2((1 - f) * sin(lat1), cos(lat1))
    U2 = atan2((1 - f) * sin(lat2), cos(lat2))
    sinU1 = sin(U1)
    cosU1 = cos(U1)
    sinU2 = sin(U2)
    cosU2 = cos(U2)

    lambd = L
    iterLimit = 100
    while True:
        sinLambda = sin(lambd)
        cosLambda = cos(lambd)
        sinSigma = sqrt((cosU2 * sinLambda)**2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)**2)
        if sinSigma == 0:
            return 0.0  # Coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = atan2(sinSigma, cosSigma)
        sinAlphaSq = (cosU1 * cosU2 * sinLambda / sinSigma)**2
        cosSqAlpha = 1 - sinAlphaSq
        if cosSqAlpha == 0:
            cos2SigmaM = 0
        else:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        lambdaPrev = lambd
        lambd = L + (1 - C) * f * sinAlphaSq * (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM**2)))
        if abs(lambd - lambdaPrev) < 1e-12:
            break
        iterLimit -= 1
        if iterLimit == 0:
            return np.nan  # Formula failed to converge

    uSq = cosSqAlpha * (a**2 - b**2) / b**2
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM**2) - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2SigmaM**2)))

    distance = b * A * (sigma - deltaSigma)

    return distance

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
        lambda r: vincenty_distance(r['PrevLon'], r['PrevLat'], r['Longitude'], r['Latitude']), axis=1)

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
    traj_data = processed_df.groupby('TaxiID')[['Longitude', 'Latitude']]\
                            .apply(lambda x: x.values).tolist()
    traj_ids = processed_df['TaxiID'].unique().tolist() # Order matches traj_data
    return traj_data, processed_df, traj_ids
# --- Visualization Functions ---

def display_stats(processed_len, processed_df, traj_data):
    """Displays key data statistics."""
    st.write(processed_df)
    n_traj = len(traj_data) if traj_data else 0
    n_proc_pts = len(processed_df) if processed_df is not None else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Points", f"{processed_len:,}" if processed_len is not None else "N/A")
    c2.metric("Valid Trajectories", f"{n_traj:,}")
    c3.metric("Points Processed", f"{n_proc_pts:,}")

    if processed_df is not None and not processed_df.empty and n_traj > 0:
        pts = [len(t) for t in traj_data]
        st.write(f"**Pts/Traj (Min/Avg/Max):** {min(pts):,} / {np.mean(pts):.1f} / {max(pts):,}")
        if pd.api.types.is_datetime64_any_dtype(processed_df["DateTime"]):
             t_min, t_max = processed_df["DateTime"].min(), processed_df["DateTime"].max()
             fmt = '%Y-%m-%d %H:%M'
             st.write(f"**Time Range:** {t_min.strftime(fmt)} to {t_max.strftime(fmt)}")