import streamlit as st
from sklearn.cluster import DBSCAN, KMeans

from utils import *

st.title("Trajectory Clustering")

st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    tra_df = upload_data(uploaded_file)
    st.subheader("Raw Data")
    st.write(tra_df.head())

    visualize_scatter_plot(tra_df)

    taxi_trajectories(tra_df)

    visualize_3(tra_df)

    visualize_4(tra_df)

    visualize_5(tra_df)

    trajectory_data = preprocess_data(tra_df)

    distance_matrix = dtw_distance(trajectory_data)
    
    
    algo = st.sidebar.selectbox("Select Clustering Algorithm", ("DBSCAN", "K-Means"))
    
    if algo == "DBSCAN":
        eps = st.sidebar.slider("Epsilon", 0.1, 10.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algo == "K-Means":
        n_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    
    try:
        labels = model.fit_predict(distance_matrix)
        visualize_clusters(tra_df, trajectory_data, labels)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
