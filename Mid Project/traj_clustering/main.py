# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering # Added
from sklearn_extra.cluster import KMedoids
# Assuming these modules contain necessary functions
from processing import *
from tab_1 import *
from tab_2 import *
from tab_3 import *
from tab_4 import *
import random

random.seed(42)
np.random.seed(42)

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Trajectory Analysis", initial_sidebar_state="expanded")
st.title("🚀 Trajectory Analysis")

# --- Session State Keys ---
# (Keep existing keys, add new ones for Agglomerative Clustering if needed for caching)
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
# --- Cache Keys ---
CACHE_SELECTED_DATES = 'selected_dates_cache'
CACHE_HOUR_RANGE = 'hour_range_cache'
CACHE_APPLY_ANOMALY = 'apply_anomaly_cache'
CACHE_MAX_SPEED = 'max_speed_cache'
CACHE_DBSCAN_EPS_PREFIX = 'dbscan_eps_cache_'
CACHE_DBSCAN_MIN_SAMPLES = 'dbscan_min_samples_cache'
CACHE_KMEDOIDS_K_PREFIX = 'kmedoids_k_cache_'
CACHE_KMEDOIDS_METHOD = 'kmedoids_method_cache' # Note: KMedoids method parameter isn't used in clustering UI? Assume 'pam' is fixed or remove if unused.
CACHE_AGGLO_K_PREFIX = 'agglo_k_cache_' # Added
CACHE_AGGLO_LINKAGE = 'agglo_linkage_cache' # Added

# --- Initialize Session State ---
default_values = {
    STATE_FILE_ID: None, STATE_RAW_DF: None, STATE_AVAILABLE_DATES: [],
    STATE_FILTERED_DF: None, STATE_PROCESSED_DF: None,
    STATE_TRAJECTORY_DATA: None, STATE_TRAJECTORY_IDS: None, STATE_DISTANCE_MATRIX: None,
    STATE_LABELS: None, STATE_RAN_CLUSTERING: False, STATE_SELECTED_METRIC: 'dtw',
    STATE_LAST_METRIC_RUN: None, STATE_PROCESSED_DF_LABELED: None, STATE_PROTOTYPE_INDICES: None,
    STATE_SELECTED_ALGO: 'K-Medoids', STATE_TRIGGER_PROCESSING: False, STATE_SELECTED_ANIM_ID: None,
    'active_tab': '📍 Overview', # Assuming this might be used internally by Streamlit for tabs

    CACHE_SELECTED_DATES: [], CACHE_HOUR_RANGE: (0, 23),
    CACHE_APPLY_ANOMALY: False, CACHE_MAX_SPEED: 100,
    CACHE_DBSCAN_MIN_SAMPLES: 3, CACHE_KMEDOIDS_METHOD: 'pam',
    CACHE_AGGLO_LINKAGE: 'average' # Default linkage for Agglomerative
    # Note: Prefixed cache keys (eps, k) are handled dynamically later
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---
def reset_all_state():
    st.cache_data.clear() # Clears Streamlit's function cache if used elsewhere
    # Clear dynamically generated cache keys
    metrics = ['dtw', 'frechet']
    prefixes = [CACHE_DBSCAN_EPS_PREFIX, CACHE_KMEDOIDS_K_PREFIX, CACHE_AGGLO_K_PREFIX]
    for metric in metrics:
        for prefix in prefixes:
             st.session_state.pop(f'{prefix}{metric}', None)
    # Reset all defined state keys to defaults
    for key, value in default_values.items():
        st.session_state[key] = value
    st.info("Cleared cache and reset all settings.")

def reset_downstream_clustering_state():
    keys_to_reset = [
        STATE_DISTANCE_MATRIX, STATE_LABELS, STATE_RAN_CLUSTERING,
        STATE_LAST_METRIC_RUN, STATE_PROCESSED_DF_LABELED, STATE_PROTOTYPE_INDICES
    ]
    for key in keys_to_reset:
        st.session_state[key] = default_values.get(key) # Reset to initial default

def reset_processed_data_state():
     keys_to_reset = [
        STATE_FILTERED_DF, STATE_PROCESSED_DF,
        STATE_TRAJECTORY_DATA, STATE_TRAJECTORY_IDS, STATE_SELECTED_ANIM_ID
     ]
     for key in keys_to_reset:
        st.session_state[key] = default_values.get(key) # Reset to initial default
     reset_downstream_clustering_state() # Also reset clustering results if processed data changes

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Filters & Load")
    st.subheader("Load Data")
    uploaded_file = st.file_uploader(
        "Upload CSV (Req: TaxiID, DateTime, Latitude, Longitude)",
        type="csv",
        # Removed key to potentially reduce unnecessary state complexity if not explicitly needed
    )

    # --- Data Loading Logic ---
    if uploaded_file is not None:
        # Use file content and metadata to create a more robust ID
        new_file_id = (uploaded_file.name, uploaded_file.size, uploaded_file.type)
        if st.session_state.get(STATE_FILE_ID) != new_file_id:
            reset_all_state()
            st.session_state[STATE_FILE_ID] = new_file_id
            try:
                # Read directly into BytesIO without intermediate variable
                raw_df_load = upload_data(io.BytesIO(uploaded_file.getvalue()))
                if raw_df_load is not None and not raw_df_load.empty:
                    st.session_state[STATE_RAW_DF] = raw_df_load
                    # Defensive check for DateTime column existence and type
                    if 'DateTime' in raw_df_load.columns and pd.api.types.is_datetime64_any_dtype(raw_df_load['DateTime']):
                         st.session_state[STATE_AVAILABLE_DATES] = sorted(raw_df_load['DateTime'].dt.date.unique())
                    else:
                         st.warning("DateTime column not found or not in expected format.")
                         st.session_state[STATE_AVAILABLE_DATES] = []
                    # Initialize selected dates to all available dates upon new file upload
                    st.session_state[CACHE_SELECTED_DATES] = st.session_state[STATE_AVAILABLE_DATES][:]
                    st.session_state[STATE_TRIGGER_PROCESSING] = True # Trigger processing for the new file
                else:
                    st.warning("Uploaded file is empty or has an invalid format after initial processing.")
                    # Ensure state is clean if upload fails
                    st.session_state[STATE_RAW_DF] = None
                    st.session_state[STATE_AVAILABLE_DATES] = []
                    st.session_state[CACHE_SELECTED_DATES] = []
            except Exception as e:
                st.error(f"Error loading or processing uploaded file: {e}")
                reset_all_state() # Reset if loading fails critically

    # --- Data Filters ---
    apply_filters_clicked = False # Local flag, reset each run
    if st.session_state.get(STATE_RAW_DF) is not None:
        st.subheader("Data Filters")
        # Get current filter values from state, falling back to defaults if not set
        current_dates = st.session_state.get(CACHE_SELECTED_DATES, st.session_state.get(STATE_AVAILABLE_DATES, []))
        current_hour_range = st.session_state.get(CACHE_HOUR_RANGE, (0, 23))
        current_anomaly = st.session_state.get(CACHE_APPLY_ANOMALY, False)
        current_speed = st.session_state.get(CACHE_MAX_SPEED, 100)

        with st.expander("Date & Time", expanded=True):
            # Ensure options are always available dates from state
            available_dates_options = st.session_state.get(STATE_AVAILABLE_DATES, [])
            selected_dates = st.multiselect(
                "Dates Filter",
                options=available_dates_options,
                default=current_dates,
                format_func=lambda d: d.strftime('%Y-%m-%d')
                # Removed key, rely on variable assignment
            )
            hour_range = st.slider(
                "Hours Filter", 0, 23,
                value=current_hour_range
                # Removed key
            )

        with st.expander("Cleaning"):
            apply_anomaly = st.checkbox(
                "Filter Invalid Moves",
                value=current_anomaly
                # Removed key
            )
            max_speed = current_speed # Default to current speed
            if apply_anomaly:
                c1, c2 = st.columns(2)
                # Corrected max speed input placement
                max_speed = c1.number_input(
                    "Max Speed (km/h)",
                    min_value=50, # Set a reasonable minimum
                    max_value=300, # Set a reasonable maximum
                    value=current_speed,
                    step=10
                    # Removed key
                 )

        # Only show Apply button if there are dates to select from
        if available_dates_options:
            if st.button("🔄 Apply Filters", key="btn_apply_filters", type="primary", use_container_width=True):
                apply_filters_clicked = True # Set flag
                # Check if filters actually changed to avoid unnecessary processing
                filters_changed = (
                    set(st.session_state.get(CACHE_SELECTED_DATES, [])) != set(selected_dates) or
                    st.session_state.get(CACHE_HOUR_RANGE) != hour_range or
                    st.session_state.get(CACHE_APPLY_ANOMALY) != apply_anomaly or
                    (apply_anomaly and st.session_state.get(CACHE_MAX_SPEED) != max_speed)
                )

                if filters_changed:
                    # Update state only if changes occurred
                    st.session_state[CACHE_SELECTED_DATES] = selected_dates
                    st.session_state[CACHE_HOUR_RANGE] = hour_range
                    st.session_state[CACHE_APPLY_ANOMALY] = apply_anomaly
                    st.session_state[CACHE_MAX_SPEED] = max_speed
                    st.session_state[STATE_TRIGGER_PROCESSING] = True # Trigger processing due to filter change
                    # Clear only downstream state affected by filtering/processing
                    reset_processed_data_state()
                else:
                    st.info("No changes detected in filters.")


# --- Data Processing Pipeline ---
# This block runs only if the trigger is set (by file upload or filter apply)
if st.session_state.get(STATE_TRIGGER_PROCESSING, False):
    st.session_state[STATE_TRIGGER_PROCESSING] = False # Consume the trigger
    raw_df_current = st.session_state.get(STATE_RAW_DF)

    if raw_df_current is not None and not raw_df_current.empty:
        with st.spinner("Processing dataset..."):
            try:
                # Get filter parameters from session state
                sd = st.session_state.get(CACHE_SELECTED_DATES)
                hr = st.session_state.get(CACHE_HOUR_RANGE, (0, 23))
                aaf = st.session_state.get(CACHE_APPLY_ANOMALY, False)
                ms = st.session_state.get(CACHE_MAX_SPEED, 100)

                # Apply filters sequentially
                df_step1 = filter_data_by_date(raw_df_current, sd, len(st.session_state.get(STATE_AVAILABLE_DATES, [])))
                df_step2 = filter_data_by_hours(df_step1, hr)
                df_step3 = calculate_trajectory_features(df_step2) # Calculate features *after* date/time filtering
                df_step4 = df_step3.copy()
                if aaf: # Apply anomaly filter only if enabled
                    df_step4 = filter_invalid_moves(df_step4, ms)

                # Final preprocessing step
                traj_data, proc_df, traj_ids = preprocess_data(df_step4)

                # Update state with new processed data
                st.session_state[STATE_FILTERED_DF] = df_step4 # Keep the filtered df before final processing
                st.session_state[STATE_PROCESSED_DF] = proc_df
                st.session_state[STATE_TRAJECTORY_DATA] = traj_data
                st.session_state[STATE_TRAJECTORY_IDS] = traj_ids
                st.session_state[STATE_SELECTED_ANIM_ID] = None # Reset selected animation ID

                reset_downstream_clustering_state() # Reset clustering as data changed
                st.success("Data processing complete!")

            except Exception as e:
                st.error(f"Error during data processing: {e}")
                reset_processed_data_state() # Reset if processing fails

    elif apply_filters_clicked:
         # This case should ideally not be hit if button logic is correct, but as a safeguard:
         st.warning("Cannot apply filters: No raw data loaded.")
         reset_processed_data_state()


# --- Main Area ---
# Retrieve potentially processed data from state for display
raw_df_main = st.session_state.get(STATE_RAW_DF)
processed_df_main = st.session_state.get(STATE_PROCESSED_DF)
traj_data_main = st.session_state.get(STATE_TRAJECTORY_DATA)
traj_ids_main = st.session_state.get(STATE_TRAJECTORY_IDS)

if raw_df_main is None:
    st.info("👋 Welcome! Please upload a CSV file via the sidebar.")
else:
    # Display only if data has been successfully processed
    if processed_df_main is not None and not processed_df_main.empty and traj_data_main is not None:
        st.header("📊 Data Summary")
        display_stats(len(processed_df_main['TaxiID'].unique()), processed_df_main, traj_data_main) # Use unique IDs count
        st.markdown("---")

        # Ensure taxi_ids are derived from the *actually* processed data
        taxi_ids = sorted(processed_df_main['TaxiID'].unique())
        if taxi_ids: # Only show if there are IDs
             selected_id = st.selectbox("Select TaxiID to see detailed infomation", taxi_ids, index=0) # Removed key
             info = get_taxi_info(processed_df_main, selected_id)

             if info:
                 c1, c2, c3 = st.columns(3)
                 c1.write(f"**Start Time:** {info['Start Time']}")
                 c2.write(f"**End Time:** {info['End Time']}")
                 c3.write(f"**# Points:** {info['Num Points']}")

                 c4, c5 = st.columns(2)
                 c4.write(f"**Total Distance:** {info['Total Distance (m)']/1000:.2f} km")
                 c5.write(f"**Avg Speed (w/o 0km/h):** {info['Average Speed (km/h)']:.2f} km/h")
        else:
            st.info("No trajectories available after filtering.")

        # Define df_viz_processed based on the state
        df_viz_processed = processed_df_main

        # --- Tabs ---
        tab_list = ["📍 Overview", "🚗 Animation", "🧩 Clustering"]
        # Use st.tabs directly, it handles its own state
        tab1, tab2, tab3 = st.tabs(tab_list)

        with tab1: # "📍 Overview"
            st.subheader("🗺️ Trajectory Overview & Density")
            if df_viz_processed is not None and not df_viz_processed.empty:
                visualize_trajectories_on_map(df_viz_processed)
                st.markdown("---")
                visualize_heatmap(df_viz_processed)
            else: st.info("No trajectories to visualize. Apply filters or upload data.")

        with tab2: # "🚗 Animation"
            st.subheader("🚗 Single Taxi Animation")
            if df_viz_processed is not None and not df_viz_processed.empty:
                available_ids_anim = sorted(df_viz_processed['TaxiID'].unique())

                if available_ids_anim:
                    # Manage selected ID state within this tab's context
                    current_anim_id = st.session_state.get(STATE_SELECTED_ANIM_ID)
                    default_index_anim = 0
                    if current_anim_id in available_ids_anim:
                        default_index_anim = available_ids_anim.index(current_anim_id)
                    else:
                        # If previous selection is invalid or not set, default to first ID
                        st.session_state[STATE_SELECTED_ANIM_ID] = available_ids_anim[0]

                    # Use a callback to update the state immediately when selection changes
                    # This makes the state consistent before the rest of the script reruns
                    def update_selected_anim_id():
                        st.session_state[STATE_SELECTED_ANIM_ID] = st.session_state.sb_anim_id_widget_key

                    sel_id_anim = st.selectbox(
                        "Select TaxiID",
                        available_ids_anim,
                        index=default_index_anim,
                        key="sb_anim_id_widget_key", # Use a unique key for the widget
                        on_change=update_selected_anim_id
                    )

                    # Filter data based on the *current* selection from state
                    df_taxi_anim = df_viz_processed[df_viz_processed['TaxiID'] == st.session_state[STATE_SELECTED_ANIM_ID]].copy()

                    if not df_taxi_anim.empty:
                        speed_multiplier = st.slider(
                            "Animation Speed", 0.25, 2.0, 1.0, 0.25, "%.2fx" # Removed key
                        )
                        with st.expander("Stop Detection Settings"):
                            speed_thresh = st.slider("Stop Speed (km/h)", 0.1, 5.0, 2.0, 0.1) # Removed key
                            min_dur = st.slider("Min Stop Duration (s)", 300, 750, 300, 30) # Removed key
                            st.caption(f" {min_dur // 60} minutes {min_dur % 60} seconds")

                        df_stops = detect_stops(df_taxi_anim, speed_thresh, min_dur)
                        fig_anim = visualize_single_trajectory_animation_plotly(df_stops, speed_multiplier=speed_multiplier)

                        if fig_anim:
                            # Use a dynamic key for plotly chart to force re-render on ID change if necessary
                            st.plotly_chart(fig_anim, use_container_width=True, key=f"anim_plot_{st.session_state[STATE_SELECTED_ANIM_ID]}")
                        else:
                            st.warning(f"Could not generate animation for Taxi {st.session_state[STATE_SELECTED_ANIM_ID]}.")
                    else:
                         st.info(f"No data found for selected Taxi ID: {st.session_state[STATE_SELECTED_ANIM_ID]}")
                else: st.info("No unique Taxi IDs found in the filtered data.")
            else: st.info("No processed data available for animation. Apply filters or upload data.")

        # with tab3: # "🚦 Congestion"
        #     st.subheader("🚦 Congestion Hotspots")
        #     if df_viz_processed is not None and not df_viz_processed.empty:
        #         # Use reasonable defaults and steps
        #         cong_speed = st.slider("Max Speed for Congestion (km/h)", 5, 40, 10, step=1) # Removed key
        #         visualize_congestion(df_viz_processed, cong_speed)
        #     else: st.info("No processed data available for congestion analysis.")

        with tab3: # "🧩 Clustering"
            st.subheader("🧩 Trajectory Clustering")
            num_traj_cluster = len(traj_data_main) if traj_data_main else 0

            # Check for processed data and sufficient trajectories
            if processed_df_main is not None and not processed_df_main.empty and traj_data_main and num_traj_cluster > 0:
                if num_traj_cluster < 2:
                    st.info("Need at least 2 trajectories for clustering.")
                else:
                    st.markdown("**Configure Clustering:**")
                    with st.container(border=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            metrics = ['dtw', 'frechet']
                            # Get selected metric from state, default to 'dtw'
                            sel_metric = st.selectbox(
                                "Distance Metric", metrics,
                                index=metrics.index(st.session_state.get(STATE_SELECTED_METRIC, 'dtw')),
                                key='sb_metric_clust' # Keep key if needed for state updates
                            )

                            # Update selected metric in state immediately if it changes
                            if st.session_state[STATE_SELECTED_METRIC] != sel_metric:
                                st.session_state[STATE_SELECTED_METRIC] = sel_metric
                                # Potentially reset parts of clustering state if metric changes?
                                # reset_downstream_clustering_state() # Decided against auto-reset here, rely on Run button

                            algos = ["DBSCAN", "K-Medoids", "Agglomerative"] # Added Agglomerative
                            # Get selected algo from state, default to 'K-Medoids'
                            sel_algo = st.selectbox(
                                "Algorithm", algos,
                                index=algos.index(st.session_state.get(STATE_SELECTED_ALGO, 'K-Medoids')),
                                key='sb_algo_clust' # Keep key if needed for state updates
                            )
                            # Update selected algo in state immediately if it changes
                            if st.session_state[STATE_SELECTED_ALGO] != sel_algo:
                                 st.session_state[STATE_SELECTED_ALGO] = sel_algo
                                 # No reset here, parameters will adjust below

                        model = None
                        # Define cache keys based on selected metric
                        dbscan_eps_key = f'{CACHE_DBSCAN_EPS_PREFIX}{sel_metric}'
                        kmedoids_k_key = f'{CACHE_KMEDOIDS_K_PREFIX}{sel_metric}'
                        agglo_k_key = f'{CACHE_AGGLO_K_PREFIX}{sel_metric}' # Added

                        # Max cluster selection based on requirement (<= N/2)
                        max_possible_k = max(2, num_traj_cluster // 2)

                        with c2:
                            # --- DBSCAN Parameter Selection ---
                            if sel_algo == "DBSCAN":
                                st.write("**DBSCAN Params**")
                                min_eps, max_eps, default_eps = 0.001, 5.0, 1.0 # Updated range and default
                                step_eps = 0.001 # Updated step

                                # Try to dynamically suggest eps based on distance matrix percentile
                                dist_mat = st.session_state.get(STATE_DISTANCE_MATRIX)
                                last_metric = st.session_state.get(STATE_LAST_METRIC_RUN)
                                # if dist_mat is not None and last_metric == sel_metric and dist_mat.ndim == 2 and dist_mat.shape[0] > 1:
                                #     try:
                                #         # Calculate percentiles on non-zero distances only if matrix is valid
                                #         nz_dists = dist_mat[dist_mat > 1e-9] # Use a small threshold for non-zero
                                #         if nz_dists.size > 10: # Need enough points for percentiles
                                #             p1 = np.percentile(nz_dists, 1)
                                #             p10 = np.percentile(nz_dists, 10)
                                #             p90 = np.percentile(nz_dists, 90)
                                #             # Clamp suggestions within the allowed range [0.001, 5.0]
                                #             suggested_min = max(min_eps, round(p1, 3))
                                #             suggested_def = max(suggested_min, round(p10, 3))
                                #             suggested_max = min(max_eps, max(suggested_def * 1.5, round(p90, 3) * 1.2))
                                #             # Use these suggestions to refine defaults if they are valid
                                #             min_eps = suggested_min
                                #             default_eps = suggested_def
                                #             max_eps = suggested_max
                                #     except Exception:
                                #         pass # Ignore percentile errors, use fixed defaults

                                # Get current value from state or use default, ensure it's within bounds
                                current_eps = st.session_state.get(dbscan_eps_key, default_eps)
                                current_eps = max(min_eps, min(max_eps, current_eps))

                                dbscan_eps = st.number_input(
                                    f"Epsilon ({sel_metric})",
                                    min_value=min_eps,
                                    max_value=max_eps,
                                    value=current_eps,
                                    step=step_eps,
                                    format="%.3f", # Use 3 decimal places for format
                                    key=f'num_eps_{sel_metric}_clust' # Unique key per metric
                                )
                                # Use max(2, ...) for min_samples slider max value
                                min_samp_val = st.session_state.get(CACHE_DBSCAN_MIN_SAMPLES, 3) # Default to 3
                                dbscan_min_samp = st.slider(
                                    "Min Samples", 3, max(3, num_traj_cluster // 2),
                                    value=min_samp_val,
                                    key='sl_min_samp_clust'
                                )
                                model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samp, metric="precomputed")

                            # --- K-Medoids Parameter Selection ---
                            elif sel_algo == "K-Medoids":
                                st.write("**K-Medoids Params**")
                                # Ensure k does not exceed max_possible_k
                                current_k = st.session_state.get(kmedoids_k_key, min(4, max_possible_k)) # Default k=4 or max_possible_k
                                current_k = max(2, min(max_possible_k, current_k)) # Clamp between 2 and max_possible_k

                                kmedoids_k = st.slider(
                                    f"Num Clusters ({sel_metric})", 2, max_possible_k,
                                    value=current_k,
                                    key=f"sl_kmed_{sel_metric}_clust" # Unique key
                                )
                                # Assuming method 'pam' is fixed or decided elsewhere
                                model = KMedoids(n_clusters=kmedoids_k, metric="precomputed", method='pam', random_state=42)

                            # --- Agglomerative Clustering Parameter Selection ---
                            elif sel_algo == "Agglomerative":
                                st.write("**Agglomerative Params**")
                                # Num Clusters (similar to K-Medoids k)
                                current_agglo_k = st.session_state.get(agglo_k_key, min(4, max_possible_k)) # Default k=4 or max_possible_k
                                current_agglo_k = max(2, min(max_possible_k, current_agglo_k)) # Clamp

                                agglo_k = st.slider(
                                    f"Num Clusters ({sel_metric})", 2, max_possible_k,
                                    value=current_agglo_k,
                                    key=f"sl_agglo_k_{sel_metric}_clust" # Unique key
                                )

                                # Linkage Method
                                linkage_options = ['average', 'complete']
                                current_linkage = st.session_state.get(CACHE_AGGLO_LINKAGE, 'ward')
                                # Ensure 'ward' is only possible with euclidean, but we use precomputed.
                                # Scikit-learn handles precomputed with linkage. Let's allow all.
                                agglo_linkage = st.selectbox(
                                    "Linkage Method", linkage_options,
                                    index=linkage_options.index(current_linkage),
                                    key='sb_agglo_linkage_clust' # Unique key
                                )
                                # Note: 'ward' requires n_clusters. Other methods can use distance_threshold instead.
                                # We are always providing n_clusters here.
                                model = AgglomerativeClustering(
                                    n_clusters=agglo_k, metric="precomputed", linkage=agglo_linkage
                                )


                    # --- Store Parameter Choices in State (Before Button Press) ---
                    # This ensures the UI reflects the latest selection even before running
                    st.session_state[STATE_SELECTED_METRIC] = sel_metric
                    st.session_state[STATE_SELECTED_ALGO] = sel_algo
                    if sel_algo == "DBSCAN":
                        st.session_state[dbscan_eps_key] = dbscan_eps
                        st.session_state[CACHE_DBSCAN_MIN_SAMPLES] = dbscan_min_samp
                    elif sel_algo == "K-Medoids":
                        st.session_state[kmedoids_k_key] = kmedoids_k
                    elif sel_algo == "Agglomerative":
                         st.session_state[agglo_k_key] = agglo_k
                         st.session_state[CACHE_AGGLO_LINKAGE] = agglo_linkage


                    st.markdown("### Distance Matrix Configuration")
                    # Logic for using precomputed matrix weights remains similar
                    use_matrix_weights = st.checkbox(
                        "Use matrix weights (only apply for unfiltered data)",
                        key="use_matrix_weights"
                    )

                    matrix_option = None
                    if use_matrix_weights:
                        matrix_options_map = {
                            "dtw": ["Beijing - DTW", "Rome - DTW"],
                            "frechet": ["Beijing - Frechet", "Rome - Frechet"]
                        }
                        display_metric_name = sel_metric.capitalize() # Display "Dtw" or "Frechet"
                        available_matrix_options = matrix_options_map.get(sel_metric.lower(), [])
                        if available_matrix_options:
                             matrix_option = st.selectbox(
                                 f"Chọn loại ma trận khoảng cách ({display_metric_name})",
                                 available_matrix_options,
                                 key=f"matrix_option_{sel_metric}" # Dynamic key based on metric
                             )
                        else:
                            st.warning(f"No pre-computed matrix options defined for {sel_metric}")


                    # --- Run Clustering Button ---
                    if st.button(f"▶️ Run {sel_algo} ({sel_metric.upper()})", key='btn_run_cluster_tab4', type="primary", use_container_width=True):
                        if model is not None:
                            reset_downstream_clustering_state() # Clear previous results before running new clustering
                            metric_run = sel_metric # Use the metric selected in the UI
                            dist_mat_current = st.session_state.get(STATE_DISTANCE_MATRIX)
                            last_metric_run = st.session_state.get(STATE_LAST_METRIC_RUN)

                            # Calculate distance matrix ONLY if it doesn't exist or the metric changed
                            if dist_mat_current is None or last_metric_run != metric_run:
                                if num_traj_cluster >= 2:
                                    dist_mat_new = None # Initialize
                                    # Load pre-computed matrix if selected and valid
                                    if use_matrix_weights and matrix_option is not None:
                                        matrix_paths = {
                                            "Beijing - DTW": "matrix_weights/b_dtw_dist_matrix.npy",
                                            "Rome - DTW": "matrix_weights/r_dtw_dist_matrix.npy",
                                            "Beijing - Frechet": "matrix_weights/b_frechet_dist_matrix.npy",
                                            "Rome - Frechet": "matrix_weights/r_frechet_dist_matrix.npy",
                                        }
                                        selected_matrix_path = matrix_paths.get(matrix_option)
                                        if selected_matrix_path:
                                            try:
                                                with st.spinner(f"Loading {matrix_option} distance matrix..."):
                                                     dist_mat_new = np.load(selected_matrix_path)
                                                # Basic validation
                                                if dist_mat_new.shape == (num_traj_cluster, num_traj_cluster):
                                                     st.success(f"Loaded distance matrix from: {selected_matrix_path}")
                                                else:
                                                     st.error(f"Loaded matrix shape {dist_mat_new.shape} doesn't match data shape ({num_traj_cluster},{num_traj_cluster}). Recalculating...")
                                                     dist_mat_new = None # Force recalculation
                                            except Exception as e:
                                                st.error(f"Failed to load distance matrix '{selected_matrix_path}': {e}. Recalculating...")
                                                dist_mat_new = None # Force recalculation
                                        else:
                                             st.warning(f"Matrix option '{matrix_option}' selected but no path found. Recalculating...")

                                    # Calculate matrix if not loaded or if loading failed/invalidated
                                    if dist_mat_new is None:
                                        with st.spinner(f'Calculating {metric_run.upper()} matrix...'):
                                            # Simplify trajectories before distance calculation for performance
                                            traj_data_rdp, _ = simplify_trajectories(traj_data_main, epsilon=0.00006)
                                            dist_mat_new = calculate_distance_matrix(traj_data_rdp, metric=metric_run)

                                    # Store the newly calculated/loaded matrix in state
                                    if dist_mat_new is not None:
                                        st.session_state[STATE_DISTANCE_MATRIX] = dist_mat_new
                                        st.session_state[STATE_LAST_METRIC_RUN] = metric_run
                                    else:
                                        st.error("Failed to obtain a valid distance matrix.")
                                        # Prevent clustering without a matrix
                                        st.stop()
                                else:
                                    st.warning("Need at least 2 trajectories to calculate distance matrix.")
                                    st.session_state[STATE_DISTANCE_MATRIX] = None # Ensure it's None
                                    st.session_state[STATE_LAST_METRIC_RUN] = None
                                    st.stop() # Prevent clustering
                            else:
                                st.info(f"Using existing distance matrix for {metric_run.upper()}.")

                            # --- Perform Clustering ---
                            dist_mat_final = st.session_state.get(STATE_DISTANCE_MATRIX)
                            if dist_mat_final is not None and dist_mat_final.shape == (num_traj_cluster, num_traj_cluster):
                                with st.spinner(f"Running {sel_algo}..."):
                                    try:
                                        # Fit the selected model
                                        labels_pred = model.fit_predict(dist_mat_final)
                                        st.session_state[STATE_LABELS] = labels_pred
                                        st.session_state[STATE_RAN_CLUSTERING] = True

                                        # Find prototypes based on algorithm
                                        prototypes = {}
                                        if sel_algo == "K-Medoids":
                                            # K-Medoids provides medoid indices directly
                                            prototypes = {labels_pred[i]: i for i in model.medoid_indices_}
                                        elif sel_algo == "DBSCAN":
                                            # Use helper function for DBSCAN prototypes
                                            prototypes = find_dbscan_prototypes(labels_pred, dist_mat_final)
                                        elif sel_algo == "Agglomerative":
                                            # Placeholder: No standard prototype for Agglomerative, maybe implement centroid/medoid later if needed
                                            prototypes = find_agglomerative_prototypes(labels_pred, dist_mat_final)

                                        st.session_state[STATE_PROTOTYPE_INDICES] = prototypes

                                        # Map labels back to the original processed DataFrame
                                        if processed_df_main is not None and traj_ids_main is not None and labels_pred is not None and len(traj_ids_main) == len(labels_pred):
                                             id_label_map = dict(zip(traj_ids_main, labels_pred))
                                             df_labeled = processed_df_main.copy()
                                             # Map labels, fill missing mappings (shouldn't happen ideally) with -1
                                             df_labeled['ClusterLabel'] = df_labeled['TaxiID'].map(id_label_map).fillna(-1).astype(int)
                                             st.session_state[STATE_PROCESSED_DF_LABELED] = df_labeled
                                        else: st.warning("Could not map cluster labels back to the DataFrame.")

                                        st.success(f"{sel_algo} ({metric_run.upper()}) complete!")

                                    except Exception as e:
                                        st.error(f"Clustering failed: {e}")
                                        reset_downstream_clustering_state() # Reset results on failure
                            else:
                                st.error("Distance matrix is invalid or unavailable. Cannot run clustering.")

                        else:
                             st.warning("Clustering algorithm not configured correctly.")

                    st.markdown("---")
                    st.subheader("📊 Clustering Results & Analysis")
                    # --- Display Clustering Results ---
                    # Check if clustering has been run successfully
                    if st.session_state.get(STATE_RAN_CLUSTERING, False):
                         # Retrieve results from state for display
                         metric_disp = st.session_state.get(STATE_LAST_METRIC_RUN, 'N/A')
                         algo_disp = st.session_state.get(STATE_SELECTED_ALGO, 'N/A') # Algo used for the displayed results
                         st.info(f"Displaying: {algo_disp} results using {metric_disp.upper()} metric")

                         labels_disp = st.session_state.get(STATE_LABELS)
                         prototypes_disp = st.session_state.get(STATE_PROTOTYPE_INDICES)
                         dist_mat_disp = st.session_state.get(STATE_DISTANCE_MATRIX)
                         df_labeled_disp = st.session_state.get(STATE_PROCESSED_DF_LABELED)

                         # Check if essential results are available
                         if labels_disp is not None and dist_mat_disp is not None and df_labeled_disp is not None:
                             # Call visualization functions (ensure these functions handle potential empty prototypes gracefully)
                            #  st.write(traj_data_main)
                             cong_speed = st.slider("Max Speed for Congestion (km/h)", 5, 40, 10, step=1) # Removed key
                             visualize_clusters_with_congestion(df_viz_processed, df_labeled_disp, traj_data_main, labels_disp, traj_ids_main, prototypes_disp, cong_speed) # Use df_labeled_disp

                             # Allow selecting TaxiID to see its cluster (only if there are valid IDs)
                             if traj_ids_main:
                                 selected_id_cluster_view = st.selectbox("Select TaxiID to see its cluster", traj_ids_main) # Removed key
                                 plot_selected_cluster(traj_data_main, traj_ids_main, labels_disp, selected_id_cluster_view) # Pass necessary data
                             else:
                                 st.info("No trajectory IDs available to select for cluster view.")


                             st.markdown("---")
                             st.subheader("📈 Cluster Summary & Performance")
                             summary_df = calculate_clustering_summary(df_labeled_disp, traj_data_main, traj_ids_main, labels_disp) # Use labeled df

                             if not summary_df.empty:
                                 st.dataframe(summary_df)
                             else: st.info("Could not generate cluster summary.")

                             # Performance metrics and plots in columns
                             c1r, c2r = st.columns(2)
                             with c1r:
                                 st.write("**Performance**")
                                 # Calculate Silhouette Score only if possible (>=2 clusters, < N labels)
                                 sil_score = calculate_silhouette(dist_mat_disp, labels_disp)
                                 if sil_score is not None:
                                     st.metric("Silhouette Score", f"{sil_score:.3f}")
                                 else: st.info("Silhouette Score N/A (Requires 2 to N-1 clusters).")

                                 st.write("**Cluster Visualization (2D)**")
                                 # Plot PCA/MDS projection
                                 plot_pca_projection(dist_mat_disp, labels_disp) # Ensure this handles potential errors

                             with c2r:
                                 # Plot cluster sizes
                                 plot_cluster_sizes(labels_disp)

                             # --- Download Clustered Data ---
                             st.subheader("📥 Download Clustered Data")
                             # Create dataframe for download
                             df_download = pd.DataFrame({
                                 'TaxiID': traj_ids_main,
                                 'Cluster': labels_disp
                             })
                             st.dataframe(df_download, use_container_width=True)

                             # Prepare CSV for download
                             csv_buffer = io.StringIO()
                             df_download.to_csv(csv_buffer, index=False)
                             st.download_button(
                                 label="📥 Download Clustering Result CSV",
                                 data=csv_buffer.getvalue(),
                                 file_name=f"clustered_trajectories_{algo_disp}_{metric_disp}.csv", # Dynamic filename
                                 mime="text/csv",
                                 use_container_width=True
                             )

                         else:
                              st.info("Clustering results are incomplete. Please run clustering again.")

                    else:
                         st.info("Run clustering using the button above to see results.")
            else:
                # Handle cases where data isn't ready for clustering
                if raw_df_main is None:
                    st.info("Upload data first.")
                elif processed_df_main is None or processed_df_main.empty:
                     st.info("Apply filters to process data for clustering.")
                elif not traj_data_main:
                    st.info("No trajectories extracted from the processed data.")

    else: # Fallback if raw_df exists but processed_df is None (e.g., initial state after upload before processing finishes)
        st.info("Data loaded. Applying initial filters or waiting for processing...")