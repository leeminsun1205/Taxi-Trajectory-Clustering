import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import matplotlib.colors
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as tslearn_dtw
import similaritymeasures
from streamlit_folium import folium_static
from sklearn.metrics import silhouette_score

# --- Distance Matrix Calculation (Cached) ---

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

