#tab_4.py
import math
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.colors
import similaritymeasures
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import plotly.graph_objects as go
from tslearn.metrics import cdist_dtw
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


@st.cache_data(show_spinner=False) 
def calculate_distance_matrix(_trajectory_data, metric='dtw'):

    n = len(_trajectory_data)
    if n < 2:
        return np.array([])
    
    dist_matrix = np.zeros((n, n))
    max_len = max(len(traj) for traj in _trajectory_data)
    padded_trajectories = np.array([
        np.pad(traj, ((0, max_len - len(traj)), (0, 0)), mode='edge') 
        for traj in _trajectory_data
    ])

    if metric == 'dtw':
        dist_matrix = cdist_dtw(padded_trajectories)
    elif metric == 'frechet':
        for i in range(n - 1):
            p = padded_trajectories[i]
            for j in range(i + 1, n):
                q = padded_trajectories[j]
                dist_matrix[i, j] = similaritymeasures.frechet_dist(p, q)
                dist_matrix[j, i] = dist_matrix[i, j]              

    return dist_matrix


def find_dbscan_prototypes(_labels, _dist_matrix):
    prototypes = {}
    unique_labels = sorted([lbl for lbl in set(_labels) if lbl != -1])
    if _dist_matrix is None or _dist_matrix.size == 0: return prototypes

    for label in unique_labels:
        cluster_indices = np.where(_labels == label)[0]
        if len(cluster_indices) == 0: continue
        if len(cluster_indices) == 1: prototypes[label] = cluster_indices[0]; continue

        try:
            cluster_dist_submatrix = _dist_matrix[np.ix_(cluster_indices, cluster_indices)]
            if not np.all(np.isfinite(cluster_dist_submatrix)): continue
            sum_dists = cluster_dist_submatrix.sum(axis=1)
            min_sum_sub_idx = np.argmin(sum_dists)
            prototypes[label] = cluster_indices[min_sum_sub_idx]
        except Exception: pass 
    return prototypes


def calculate_silhouette(dist_matrix, labels):

    n_labels = len(set(labels))
    n_samples = len(labels)

    if 2 <= n_labels < n_samples:
        if dist_matrix.shape == (n_samples, n_samples) and np.all(np.isfinite(dist_matrix)):
            return silhouette_score(dist_matrix, labels, metric='precomputed')
    return None 


def rdp_with_index(points, indices, epsilon):
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = point_to_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            dmax, index = d, i
    if dmax >= epsilon:
        first_points, first_indices = rdp_with_index(points[:index+1], indices[:index+1], epsilon)
        second_points, second_indices = rdp_with_index(points[index:], indices[index:], epsilon)
        results = first_points[:-1] + second_points
        results_indices = first_indices[:-1] + second_indices
    else:
        results, results_indices = [points[0], points[-1]], [indices[0], indices[-1]]
    return results, results_indices


def simplify_trajectories(traj_data, epsilon=0.0005):
    simplified_trajs = []
    simplified_indices = []

    for traj in traj_data:
        indices = list(range(len(traj)))
        simp_points, simp_indices = rdp_with_index(traj, indices, epsilon)
        simplified_trajs.append(np.array(simp_points))
        simplified_indices.append(simp_indices)
    
    return simplified_trajs, simplified_indices


def calculate_clustering_summary(_processed_df_labeled, _traj_data, _traj_ids, _labels):
    summary = []
    if _processed_df_labeled is None or _traj_data is None or _traj_ids is None or _labels is None or \
       len(_traj_ids) != len(_labels) or len(_traj_data) != len(_labels):
        return pd.DataFrame()

    id_to_idx = {tid: i for i, tid in enumerate(_traj_ids)}
    unique_labels = sorted([lbl for lbl in set(_labels) if lbl != -1])

    for label in unique_labels:
        cluster_df = _processed_df_labeled[_processed_df_labeled['ClusterLabel'] == label]
        if cluster_df.empty:
            continue

        cluster_traj_ids = cluster_df['TaxiID'].unique()
        cluster_indices = [id_to_idx[tid] for tid in cluster_traj_ids if tid in id_to_idx]
        if not cluster_indices:
            continue

        n_traj = len(cluster_indices)
        points = [len(_traj_data[i]) for i in cluster_indices if i < len(_traj_data)]
        avg_pts = np.mean(points) if points else 0

        earliest, latest = cluster_df['DateTime'].min(), cluster_df['DateTime'].max()

        traj_lengths = []
        for tid in cluster_traj_ids:
            traj_pts = cluster_df[cluster_df['TaxiID'] == tid]
            if 'DistJump_m' in traj_pts.columns and not traj_pts['DistJump_m'].isna().all():
                total_dist = traj_pts['DistJump_m'].sum()
                traj_lengths.append(total_dist)
        avg_dist_m = np.mean(traj_lengths) if traj_lengths else np.nan

        durations = []
        for tid in cluster_traj_ids:
            traj_pts = cluster_df[cluster_df['TaxiID'] == tid]
            if len(traj_pts) >= 2:
                dur = (traj_pts['DateTime'].max() - traj_pts['DateTime'].min()).total_seconds()
                durations.append(dur)
        avg_dur_s = np.mean(durations) if durations else np.nan

        def format_seconds(seconds):
            if pd.isna(seconds): return "N/A"
            h, m = divmod(int(seconds), 3600)
            m, s = divmod(m, 60)
            return f"{h:02}:{m:02}:{s:02}"

        avg_spd_all = cluster_df['Speed_kmh'].mean() if 'Speed_kmh' in cluster_df.columns else np.nan
        avg_spd_nonzero = cluster_df.loc[cluster_df['Speed_kmh'] > 0, 'Speed_kmh'].mean()

        summary.append({
            "Cluster": label,
            "Num Trajectories": n_traj,
            "Avg Points/Traj": f"{avg_pts:.1f}",
            "Earliest Point": earliest,
            "Latest Point": latest,
            "Avg Duration": format_seconds(avg_dur_s),
            "Avg Speed (km/h)": f"{avg_spd_all:.1f}" if pd.notna(avg_spd_all) else "N/A",
            "Avg Speed (non-zero km/h)": f"{avg_spd_nonzero:.1f}" if pd.notna(avg_spd_nonzero) else "N/A",
            "Avg Distance (km)": f"{avg_dist_m/1000:.1f}" if pd.notna(avg_dist_m) else "N/A",
        })


    if not summary:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary).set_index("Cluster")

    # Format datetime columns safely
    time_fmt = '%Y-%m-%d %H:%M'
    for col in ['Earliest Point', 'Latest Point']:
        if pd.api.types.is_datetime64_any_dtype(summary_df[col]):
            summary_df[col] = summary_df[col].dt.strftime(time_fmt).fillna("N/A")
        else:
            summary_df[col] = "N/A"

    return summary_df


def visualize_clusters(proc_df, traj_data, labels, traj_ids, proto_indices=None):
    """Folium map visualizing clustered trajectories."""
    st.subheader("Clustered Trajectories")
    if not traj_data or labels is None or traj_ids is None or \
       len(traj_data) != len(labels) or len(traj_ids) != len(labels):
        st.warning("Invalid input for cluster visualization."); return
    try: center = [proc_df['Latitude'].mean(), proc_df['Longitude'].mean()] if proc_df is not None else [39.9, 116.4]
    except: center = [39.9, 116.4]
    st.write(f"Map center: {center}")
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
        folium_traj = [[p[1], p[0]] for p in traj]

        if len(folium_traj) >= 2:
            folium.PolyLine(locations=folium_traj, color=color,
                            weight=4 if is_proto else 2, opacity=1.0 if is_proto else 0.6,
                            popup=f"ID: {tid}<br>Cluster: {lbl}{' (Proto)' if is_proto else ''}"
                           ).add_to(layers[name])
    if layers: folium.LayerControl(collapsed=False).add_to(m)
    folium_static(m, width=1200, height=600)


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


def plot_pca_projection(dist_mat, labels, title="PCA Projection of Clusters"):
    if dist_mat is None or len(dist_mat) < 2:
        st.info("Not enough data to project.")
        return
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(dist_mat)

        fig, ax = plt.subplots()
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", ax=ax)
        ax.set_title(title)
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate PCA projection: {e}")


def plot_selected_cluster(traj_data, traj_ids, labels, selected_id):
    if selected_id not in traj_ids:
        st.warning("TaxiID không tồn tại.")
        return

    idx = traj_ids.index(selected_id)
    selected_label = labels[idx]

    fig = go.Figure()

    for i, (traj, label) in enumerate(zip(traj_data, labels)):
        if label == selected_label and traj_ids[i] != selected_id:
            fig.add_trace(go.Scattermapbox(
                lon=traj[:, 0], lat=traj[:, 1],
                mode='lines',
                line=dict(width=2, color='cyan'),
                name=f"Taxi {traj_ids[i]}",
                hoverinfo='skip'
            ))

    traj = traj_data[idx]
    fig.add_trace(go.Scattermapbox(
        lon=traj[:, 0], lat=traj[:, 1],
        mode='lines+markers',
        line=dict(width=4, color='red'),
        marker=dict(size=6, color='red'),
        name=f"Selected Taxi {selected_id}"
    ))

    fig.add_trace(go.Scattermapbox(
        lon=[traj[0, 0]], lat=[traj[0, 1]],
        mode='markers+text',
        marker=dict(size=10, color='green'),
        text=["Start"], textposition="top center",
        name="Start"
    ))

    fig.add_trace(go.Scattermapbox(
        lon=[traj[-1, 0]], lat=[traj[-1, 1]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=["End"], textposition="bottom center",
        name="End"
    ))

    center_lat = traj[0, 1]
    center_lon = traj[0, 0]

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=12,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

