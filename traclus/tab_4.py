#tab_4.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.cluster import DBSCAN
import folium
import plotly.express as px
import matplotlib.colors
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as tslearn_dtw
import similaritymeasures
from streamlit_folium import folium_static
from sklearn.metrics import silhouette_score

def point_to_line_distance(point, start, end):
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    if den == 0:
        return math.hypot(x0 - x1, y0 - y1)
    return num / den

def partition_trajectory(trajectory, threshold=0.0005):
    segments = []
    start_idx = 0
    n = len(trajectory)
    if n < 3:
        return [trajectory]
    for i in range(1, n - 1):
        d = point_to_line_distance(trajectory[i], trajectory[start_idx], trajectory[i + 1])
        if d > threshold:
            segments.append(trajectory[start_idx:i + 1])
            start_idx = i
    segments.append(trajectory[start_idx:])
    return segments

def extract_segment_features(segment):
    segment = np.array(segment)
    start = segment[0]
    end = segment[-1]
    midpoint = (start + end) / 2
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return np.array([midpoint[0], midpoint[1], angle])

def TRACLUS(trajectories, partition_threshold=0.0005, eps_cluster=0.001, min_samples=2):
    all_segments = []
    segment_features = []
    segment_origin = []
    for i, traj in enumerate(trajectories):
        segments = partition_trajectory(traj, threshold=partition_threshold)
        for seg in segments:
            feat = extract_segment_features(seg)
            all_segments.append(seg)
            segment_features.append(feat)
            segment_origin.append(i)
    segment_features = np.array(segment_features)
    clustering = DBSCAN(eps=eps_cluster, min_samples=min_samples)
    labels = clustering.fit_predict(segment_features)
    return labels, all_segments, segment_features, segment_origin

# @st.cache_data(show_spinner=False)
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

def calculate_clustering_summary(_processed_df_labeled, _traj_data, _traj_ids, _labels):
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

