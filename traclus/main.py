# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import io
from sklearn_extra.cluster import KMedoids
from processing import *
from tab_1 import *
from tab_2 import *
from tab_3 import *
from tab_4 import *
# --- Page Config ---
st.set_page_config(layout="wide", page_title="Trajectory Analysis", initial_sidebar_state="expanded")
st.title("🚀 Trajectory Analysis")

# --- Constants for Session State Keys ---
STATE_FILE_ID = 'file_identifier'
STATE_RAW_DF = 'raw_df'
STATE_AVAILABLE_DATES = 'available_dates'
STATE_FILTERED_DF = 'filtered_df'
STATE_PROCESSED_DF = 'processed_df'
STATE_TRAJECTORY_DATA = 'trajectory_data'
STATE_TRAJECTORY_IDS = 'trajectory_ids'
STATE_DISTANCE_MATRIX = 'distance_matrix'
STATE_LABELS = 'labels'
STATE_RAN_CLUSTERING = 'ran_clustering'
STATE_SELECTED_METRIC = 'selected_metric'
STATE_LAST_METRIC_RUN = 'last_metric_run'
STATE_PROCESSED_DF_LABELED = 'processed_df_with_labels'
STATE_PROTOTYPE_INDICES = 'prototype_indices'
STATE_SELECTED_ALGO = 'selected_algo'
STATE_TRIGGER_PROCESSING = 'trigger_processing'
STATE_SELECTED_ANIM_ID = 'selected_anim_id' 

# Filter Cache Keys
CACHE_SELECTED_DATES = 'selected_dates_cache'
CACHE_HOUR_RANGE = 'hour_range_cache'
CACHE_APPLY_ANOMALY = 'apply_anomaly_cache'
CACHE_MAX_SPEED = 'max_speed_cache'
CACHE_DBSCAN_EPS_PREFIX = 'dbscan_eps_cache_'
CACHE_DBSCAN_MIN_SAMPLES = 'dbscan_min_samples_cache'
CACHE_KMEDOIDS_K_PREFIX = 'kmedoids_k_cache_'
CACHE_KMEDOIDS_METHOD = 'kmedoids_method_cache'

# --- Initialize Session State ---
default_values = {
    STATE_FILE_ID: None, STATE_RAW_DF: None, STATE_AVAILABLE_DATES: [],
    STATE_FILTERED_DF: None, STATE_PROCESSED_DF: None,
    STATE_TRAJECTORY_DATA: None, STATE_TRAJECTORY_IDS: None, STATE_DISTANCE_MATRIX: None,
    STATE_LABELS: None, STATE_RAN_CLUSTERING: False, STATE_SELECTED_METRIC: 'dtw',
    STATE_LAST_METRIC_RUN: None, STATE_PROCESSED_DF_LABELED: None, STATE_PROTOTYPE_INDICES: None,
    STATE_SELECTED_ALGO: 'DBSCAN', STATE_TRIGGER_PROCESSING: False, STATE_SELECTED_ANIM_ID: None,
    'active_tab': '📍 Overview', 

    CACHE_SELECTED_DATES: [], CACHE_HOUR_RANGE: (0, 24),
    CACHE_APPLY_ANOMALY: False, CACHE_MAX_SPEED: 100, 
    CACHE_DBSCAN_MIN_SAMPLES: 3, CACHE_KMEDOIDS_METHOD: 'pam'
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---
def reset_all_state():
    st.cache_data.clear()
    for metric in ['dtw', 'frechet', 'hausdorff']:
        st.session_state.pop(f'{CACHE_DBSCAN_EPS_PREFIX}{metric}', None)
        st.session_state.pop(f'{CACHE_KMEDOIDS_K_PREFIX}{metric}', None)
    for key, value in default_values.items():
        st.session_state[key] = value
    st.info("Cleared cache and reset all settings.")

def reset_downstream_clustering_state():
    keys_to_reset = [
        STATE_DISTANCE_MATRIX, STATE_LABELS, STATE_RAN_CLUSTERING,
        STATE_LAST_METRIC_RUN, STATE_PROCESSED_DF_LABELED, STATE_PROTOTYPE_INDICES
    ]
    for key in keys_to_reset:
        st.session_state[key] = default_values.get(key)

def reset_processed_data_state():
     keys_to_reset = [
        STATE_FILTERED_DF, STATE_PROCESSED_DF,
        STATE_TRAJECTORY_DATA, STATE_TRAJECTORY_IDS, STATE_SELECTED_ANIM_ID
     ]
     for key in keys_to_reset:
        st.session_state[key] = default_values.get(key)
     reset_downstream_clustering_state()


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Filters & Load")
    st.subheader("Load Data")
    uploaded_file = st.file_uploader(
        "Upload CSV (Req: TaxiID, DateTime, Latitude, Longitude)",
        type="csv",
        key="file_uploader_widget"
    )

    # --- Data Loading Logic ---
    if uploaded_file is not None:
        new_file_id = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get(STATE_FILE_ID) != new_file_id:
            reset_all_state()
            st.session_state[STATE_FILE_ID] = new_file_id
            file_content = io.BytesIO(uploaded_file.getvalue())
            with st.spinner("Loading and cleaning data..."):
                raw_df_load = upload_data(file_content)
                if raw_df_load is not None and not raw_df_load.empty:
                    st.session_state[STATE_RAW_DF] = raw_df_load
                    if pd.api.types.is_datetime64_any_dtype(raw_df_load['DateTime']):
                         st.session_state[STATE_AVAILABLE_DATES] = sorted(raw_df_load['DateTime'].dt.date.unique())
                    else: st.session_state[STATE_AVAILABLE_DATES] = []
                    st.session_state[STATE_TRIGGER_PROCESSING] = True
                else:
                    st.warning("Uploaded file is empty or invalid.")
                    st.session_state[STATE_RAW_DF] = None
                    st.session_state[STATE_AVAILABLE_DATES] = []
                st.session_state[CACHE_SELECTED_DATES] = st.session_state[STATE_AVAILABLE_DATES]

    # --- Data Filters ---
    apply_filters_clicked = False
    if st.session_state.get(STATE_RAW_DF) is not None:
        st.subheader("Data Filters")
        current_dates = st.session_state.get(CACHE_SELECTED_DATES, st.session_state[STATE_AVAILABLE_DATES])
        current_hour_range = st.session_state.get(CACHE_HOUR_RANGE, (0, 24))
        current_anomaly = st.session_state.get(CACHE_APPLY_ANOMALY, False)
        current_speed = st.session_state.get(CACHE_MAX_SPEED, 100)

        with st.expander("Date & Time", expanded=True):
            selected_dates = st.multiselect("Dates Filter", options=st.session_state.get(STATE_AVAILABLE_DATES), default=st.session_state.get(STATE_AVAILABLE_DATES), format_func=lambda d: d.strftime('%Y-%m-%d'), key='ms_dates')
            hour_range = current_hour_range
            hour_range = st.slider("Hours Filter", 0, 24, value=current_hour_range, key='sl_hours')

        with st.expander("Cleaning"):
            apply_anomaly = st.checkbox("Filter Invalid Moves", value=current_anomaly, key='cb_anomaly')
            max_speed = current_speed
            if apply_anomaly:
                c1, c2 = st.columns(2)
                max_speed = c1.number_input("Max Speed (km/h)", 100, 300, value=current_speed, key='ni_speed')
        if selected_dates != []:
            if st.button("🔄 Apply Filters & Update", key="btn_apply_filters", type="primary", use_container_width=True):
                apply_filters_clicked = True
                st.session_state[CACHE_SELECTED_DATES] = selected_dates
                st.session_state[CACHE_HOUR_RANGE] = hour_range
                st.session_state[CACHE_APPLY_ANOMALY] = apply_anomaly
                st.session_state[CACHE_MAX_SPEED] = max_speed
                st.session_state[STATE_TRIGGER_PROCESSING] = True

# --- Data Processing Pipeline ---
if st.session_state.get(STATE_TRIGGER_PROCESSING, False):
    st.session_state[STATE_TRIGGER_PROCESSING] = False
    raw_df_current = st.session_state.get(STATE_RAW_DF)
    if raw_df_current is not None and not raw_df_current.empty:
        with st.spinner("Processing dataset..."):
            if st.session_state[STATE_PROCESSED_DF] is None:
                df_step4 = calculate_trajectory_features(raw_df_current)
            else:
                sd = st.session_state.get(CACHE_SELECTED_DATES)
                hr = st.session_state.get(CACHE_HOUR_RANGE, (0, 24))
                aaf = st.session_state.get(CACHE_APPLY_ANOMALY, False)
                ms = st.session_state.get(CACHE_MAX_SPEED, 100)
                df_step1 = filter_data_by_date(raw_df_current, sd, len(st.session_state[STATE_AVAILABLE_DATES]))
                df_step2 = filter_data_by_hours(df_step1, hr)
                df_step3 = calculate_trajectory_features(df_step2)
                df_step4 = df_step3.copy()
                if st.session_state[CACHE_APPLY_ANOMALY]:
                    df_step4 = filter_invalid_moves(df_step4, ms) if aaf else df_step4
            traj_data, proc_df, traj_ids = preprocess_data(df_step4)

            st.session_state[STATE_FILTERED_DF] = df_step4
            st.session_state[STATE_PROCESSED_DF] = proc_df
            st.session_state[STATE_TRAJECTORY_DATA] = traj_data
            st.session_state[STATE_TRAJECTORY_IDS] = traj_ids
            st.session_state[STATE_SELECTED_ANIM_ID] = None 

            reset_downstream_clustering_state()
            st.success("Data processing complete!")

    elif apply_filters_clicked:
         st.warning("Cannot apply filters: No raw data loaded.")
         reset_processed_data_state()


# --- Main Area ---
raw_df_main = st.session_state.get(STATE_RAW_DF)
processed_df_main = st.session_state.get(STATE_PROCESSED_DF)
traj_data_main = st.session_state.get(STATE_TRAJECTORY_DATA)
traj_ids_main = st.session_state.get(STATE_TRAJECTORY_IDS)

if raw_df_main is None:
    st.info("👋 Welcome! Please upload a CSV file via the sidebar.")
else:
    st.header("📊 Data Summary")
    display_stats(len(processed_df_main), processed_df_main, traj_data_main)
    st.markdown("---")

    df_viz_processed = processed_df_main if isinstance(processed_df_main, pd.DataFrame) else pd.DataFrame()
    
    # --- Tabs ---
    tab_list = ["📍 Overview", "🚗 Animation", "🚦 Congestion", "🧩 Clustering"]
    selected_tab_label_tuple = st.tabs(tab_list)
    
    # --- Tab Content Containers ---
    tab1, tab2, tab3, tab4 = selected_tab_label_tuple

    with tab1: # Corresponds to "📍 Overview"
        st.subheader("🗺️ Trajectory Overview & Density")
        if not df_viz_processed.empty:
            map_type = st.radio("Map Type", ["Folium", "Plotly"], key='rb_map_type_overview', horizontal=True)
            if map_type == "Folium":
                visualize_1(df_viz_processed)
            else:
                visualize_2(df_viz_processed)
            st.markdown("---")
            visualize_heatmap(df_viz_processed)
        else: st.info("No trajectories to visualize.")

    with tab2: # Corresponds to "🚗 Animation"
        st.subheader("🚗 Single Taxi Animation")
        if not df_viz_processed.empty:
            # Use the full dataframe with features for ID selection and filtering
            available_ids_anim = sorted(df_viz_processed['TaxiID'].unique())

            if available_ids_anim:
                # Function to update selected ID in state
                def update_selected_anim_id():
                    st.session_state[STATE_SELECTED_ANIM_ID] = st.session_state.sb_anim_id_widget

                # Get default ID: use state if exists and valid, else first ID
                default_anim_id = st.session_state.get(STATE_SELECTED_ANIM_ID)
                if default_anim_id not in available_ids_anim:
                    default_anim_id = available_ids_anim[0]
                    st.session_state[STATE_SELECTED_ANIM_ID] = default_anim_id 

                sel_id_anim = st.selectbox(
                    "Select TaxiID",
                    available_ids_anim,
                    index=available_ids_anim.index(default_anim_id),
                    key="sb_anim_id_widget", # Widget key
                    on_change=update_selected_anim_id # Callback to update state
                )

                df_taxi_anim = df_viz_processed[df_viz_processed['TaxiID'] == sel_id_anim].copy()

                if not df_taxi_anim.empty:   
                    # --- Animation Controls ---
                    speed_multiplier = st.slider(
                        "Animation Speed", 0.25, 2.0, 1.0, 0.25, "%.2fx", key='sl_anim_speed'
                    )
                    with st.expander("Stop Detection Settings"):
                        speed_thresh = st.slider("Stop Speed (km/h)", 0.1, 5.0, 2.0, 0.1, key='sl_stop_spd')
                        min_dur = st.slider("Min Stop Duration (s)", 300, 750, 300, 30, key='sl_stop_dur')
                        st.caption(f" {min_dur // 60} minutes {min_dur % 60} seconds")
                    # --- Data Prep & Visualization ---
                    # Detect stops (cached based on df_taxi_anim)
                    df_stops = detect_stops(df_taxi_anim, speed_thresh, min_dur)

                    # Generate the Plotly figure
                    fig_anim = visualize_single_trajectory_animation_plotly(df_stops, speed_multiplier=speed_multiplier)

                    # Display with a dynamic key based on the selected ID
                    if fig_anim:
                        st.plotly_chart(fig_anim, use_container_width=True, key=f"anim_plot_{sel_id_anim}")
                    else:
                        st.warning(f"Could not generate animation for Taxi {sel_id_anim}.")
                else:
                     st.info(f"No data found for selected Taxi ID: {sel_id_anim}")
            else: st.info("No unique Taxi IDs found in the filtered data.")
        else: st.info("No data with features available for animation (Apply filters first).")

    with tab3: # Corresponds to "🚦 Congestion"
        st.subheader("🚦 Congestion Hotspots")
        if not df_viz_processed.empty:
            cong_speed = st.slider("Max Speed for Congestion (km/h)", 5, 40, 10, key='sl_cong_spd_tab3')
            visualize_congestion(df_viz_processed, cong_speed)
        else: st.info("No data with features available for congestion analysis.")

    with tab4: # Corresponds to "🧩 Clustering"
        st.subheader("🧩 Trajectory Clustering")
        num_traj_cluster = len(traj_data_main) if traj_data_main else 0

        if processed_df_main is not None and not processed_df_main.empty and num_traj_cluster > 0:
            if num_traj_cluster < 2:
                st.info("Need at least 2 trajectories for clustering.")
            else:
                st.markdown("**Configure Clustering:**")
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        metrics = ['dtw', 'frechet', 'hausdorff']
                        sel_metric = st.selectbox("Distance Metric", metrics, index=metrics.index(st.session_state.get(STATE_SELECTED_METRIC, 'dtw')), key='sb_metric_clust')
                        algos = ["DBSCAN", "K-Medoids"]
                        sel_algo = st.selectbox("Algorithm", algos, index=algos.index(st.session_state.get(STATE_SELECTED_ALGO, 'DBSCAN')), key='sb_algo_clust')

                    model = None
                    dbscan_eps_key = f'{CACHE_DBSCAN_EPS_PREFIX}{sel_metric}'
                    kmedoids_k_key = f'{CACHE_KMEDOIDS_K_PREFIX}{sel_metric}'

                    with c2:
                        if sel_algo == "DBSCAN":
                            st.write("**DBSCAN Params**")
                            min_eps, max_eps, def_eps = 0.01, 10.0, 1.0
                            dist_mat = st.session_state.get(STATE_DISTANCE_MATRIX); last_metric = st.session_state.get(STATE_LAST_METRIC_RUN)
                            if dist_mat is not None and last_metric == sel_metric and dist_mat.size > 10:
                                nz = dist_mat[dist_mat > 1e-9];
                                if nz.size > 10:
                                    try: min_eps, def_eps, max_eps = max(0.01, round(np.percentile(nz, 1), 2)), max(min_eps, round(np.percentile(nz, 10), 2)), max(def_eps * 1.5, round(np.percentile(nz, 90), 2) * 1.2)
                                    except: pass
                            eps_val = st.session_state.get(dbscan_eps_key, def_eps); eps_val = max(min_eps, min(max_eps, eps_val))
                            dbscan_eps = st.slider(f"Epsilon ({sel_metric})", min_eps, max_eps, value=eps_val, step=max(0.01, (max_eps-min_eps)/100), key=f'sl_eps_{sel_metric}_clust')
                            min_samp_val = st.session_state.get(CACHE_DBSCAN_MIN_SAMPLES, 3)
                            dbscan_min_samp = st.slider("Min Samples", 1, max(2, num_traj_cluster // 10 + 1), value=min_samp_val, key='sl_min_samp_clust')
                            model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samp, metric="precomputed")
                        elif sel_algo == "K-Medoids":
                            st.write("**K-Medoids Params**")
                            max_k = max(2, num_traj_cluster - 1); k_val = st.session_state.get(kmedoids_k_key, min(3, max_k)); k_val = max(2, min(max_k, k_val))
                            kmedoids_k = st.slider(f"Num Clusters ({sel_metric})", 2, max_k, value=k_val, key=f"sl_k_{sel_metric}_clust")
                            methods = ['pam', 'alternate']; method_val = st.session_state.get(CACHE_KMEDOIDS_METHOD, 'pam')
                            kmedoids_method = st.radio("Method", methods, horizontal=True, index=methods.index(method_val), key='rb_kmed_method_clust')
                            model = KMedoids(n_clusters=kmedoids_k, metric="precomputed", random_state=42, method=kmedoids_method)

                st.session_state[STATE_SELECTED_METRIC] = sel_metric
                st.session_state[STATE_SELECTED_ALGO] = sel_algo
                if sel_algo == "DBSCAN": st.session_state[dbscan_eps_key], st.session_state[CACHE_DBSCAN_MIN_SAMPLES] = dbscan_eps, dbscan_min_samp
                elif sel_algo == "K-Medoids": st.session_state[kmedoids_k_key], st.session_state[CACHE_KMEDOIDS_METHOD] = kmedoids_k, kmedoids_method

                if st.button(f"▶️ Run {sel_algo} ({sel_metric.upper()})", key='btn_run_cluster_tab4', type="primary", use_container_width=True):
                    if model is not None:
                        reset_downstream_clustering_state(); metric_run = sel_metric
                        dist_mat_current = st.session_state.get(STATE_DISTANCE_MATRIX); last_metric_run = st.session_state.get(STATE_LAST_METRIC_RUN)
                        if dist_mat_current is None or last_metric_run != metric_run:
                             if num_traj_cluster >= 2:
                                 with st.spinner(f'Calculating {metric_run.upper()} matrix...'):
                                    dist_mat_new = calculate_distance_matrix(traj_data_main, metric=metric_run)
                                    st.session_state[STATE_DISTANCE_MATRIX], st.session_state[STATE_LAST_METRIC_RUN] = dist_mat_new, metric_run
                             else: st.session_state[STATE_DISTANCE_MATRIX] = np.array([])
                        else: st.info(f"Using existing distance matrix for {metric_run.upper()}.")
                        dist_mat_final = st.session_state.get(STATE_DISTANCE_MATRIX)
                        if dist_mat_final is not None and dist_mat_final.shape == (num_traj_cluster, num_traj_cluster):
                            with st.spinner(f"Running {sel_algo}..."):
                                try:
                                    labels_pred = model.fit_predict(dist_mat_final)
                                    st.session_state[STATE_LABELS], st.session_state[STATE_RAN_CLUSTERING] = labels_pred, True
                                    prototypes = {}
                                    if sel_algo == "K-Medoids": prototypes = {labels_pred[i]: i for i in model.medoid_indices_}
                                    elif sel_algo == "DBSCAN": prototypes = find_dbscan_prototypes(traj_data_main, labels_pred, dist_mat_final)
                                    st.session_state[STATE_PROTOTYPE_INDICES] = prototypes
                                    if processed_df_main is not None and traj_ids_main is not None and labels_pred is not None and len(traj_ids_main) == len(labels_pred):
                                         id_label_map = dict(zip(traj_ids_main, labels_pred)); df_labeled = processed_df_main.copy()
                                         df_labeled['ClusterLabel'] = df_labeled['TaxiID'].map(id_label_map).fillna(-1).astype(int); st.session_state[STATE_PROCESSED_DF_LABELED] = df_labeled
                                    else: st.warning("Could not map cluster labels back.")
                                    st.success(f"{sel_algo} ({metric_run.upper()}) complete!")
                                except Exception as e: st.error(f"Clustering failed: {e}"); reset_downstream_clustering_state()
                        else: st.error("Distance matrix invalid. Cannot cluster.")
                    else: st.warning("Clustering algorithm not configured.")

                st.markdown("---"); st.subheader("📊 Clustering Results & Analysis")
                if st.session_state.get(STATE_RAN_CLUSTERING, False):
                     metric_disp, algo_disp = st.session_state.get(STATE_LAST_METRIC_RUN, 'N/A'), st.session_state.get(STATE_SELECTED_ALGO, 'N/A')
                     st.info(f"Displaying: {algo_disp} with {metric_disp.upper()}")
                     labels_disp, prototypes_disp = st.session_state.get(STATE_LABELS), st.session_state.get(STATE_PROTOTYPE_INDICES)
                     dist_mat_disp, df_labeled_disp = st.session_state.get(STATE_DISTANCE_MATRIX), st.session_state.get(STATE_PROCESSED_DF_LABELED)
                     visualize_clusters(processed_df_main, traj_data_main, labels_disp, traj_ids_main, prototypes_disp)
                     st.markdown("---"); st.subheader("📈 Cluster Summary & Performance")
                     summary_df = calculate_clustering_summary(df_labeled_disp, traj_data_main, traj_ids_main, labels_disp)
                     if not summary_df.empty: st.dataframe(summary_df)
                     else: st.info("Could not generate cluster summary.")
                     c1r, c2r = st.columns(2)
                     with c1r:
                         st.write("**Performance**"); sil_score = calculate_silhouette(dist_mat_disp, labels_disp)
                         if sil_score is not None: st.metric("Silhouette Score", f"{sil_score:.3f}")
                         else: st.info("Silhouette N/A")
                     with c2r: plot_cluster_sizes(labels_disp)
                else: st.info("Run clustering to see results.")
        else: st.info("No data available for clustering.")