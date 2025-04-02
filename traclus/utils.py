from tslearn.metrics import dtw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px

def upload_data(uploaded_file):
    tra_df = pd.read_csv(uploaded_file)
    tra_df = tra_df.drop(columns=tra_df.columns[0], axis=1)
    tra_df = tra_df.rename(columns={"Date and Time": "DateTime", "DriveNo": "TaxiID"})
    return tra_df

def preprocess_data(tra_df):
    tra_df["DateTime"] = pd.to_datetime(tra_df["DateTime"], errors="coerce")
    tra_df['Longitude'] = pd.to_numeric(tra_df['Longitude'], errors='coerce')
    tra_df['Latitude'] = pd.to_numeric(tra_df['Latitude'], errors='coerce')
    grouped = tra_df.groupby('TaxiID', group_keys=True)
    trajectories = grouped.apply(lambda x: x.sort_values('DateTime'), include_groups=False)
    trajectories = pd.DataFrame(trajectories)
    trajectory_data = trajectories.groupby('TaxiID')[['Longitude', 'Latitude']].apply(lambda x: x.values.tolist()).tolist()
    return trajectory_data

def dtw_distance(trajectory_data):

    n = len(trajectory_data)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n): 
            distance_matrix[i, j] = distance_matrix[j, i] = dtw(trajectory_data[i], trajectory_data[j])
    return distance_matrix

def visualize_scatter_plot(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Longitude'], df['Latitude'], s=1, c='blue', alpha=0.5)
    plt.title('Scatter Plot of Taxi Trajectories')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    st.pyplot(plt)

def taxi_trajectories(df):
    df = df.sort_values(['TaxiID', 'DateTime']).reset_index(drop=True)

    unique_drives = df['TaxiID'].unique()
    plt.figure(figsize=(10, 6))
    for drive in unique_drives:
        drive_data = df[df['TaxiID'] == drive]
        plt.plot(drive_data['Longitude'], drive_data['Latitude'], marker='o', markersize=2, linestyle='-', label=f'Drive {drive}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Taxi Trajectories')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    st.pyplot(plt)   

def visualize_3(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['TaxiID', 'DateTime']).reset_index(drop=True)

    pickup_points = df.groupby('TaxiID').head(2)[['Latitude', 'Longitude']].values

    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    map_pickups = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for lat, lon in pickup_points:
        folium.CircleMarker(location=[lat, lon],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.6).add_to(map_pickups)
    folium_static(map_pickups)

def visualize_4(df):
    colors = [
        "red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige", "darkblue",
        "darkgreen", "cadetblue", "darkpurple", "pink", "gray", "black", "lightgray", "lightblue",
        "lightgreen", "lightpurple", "white", "cyan", "magenta", "lime", "gold", "maroon", "navy",
        "olive", "teal", "aqua", "fuchsia", "indigo", "khaki", "lavender", "salmon", "sienna",
        "tan", "tomato", "turquoise", "violet", "wheat", "yellow", "azure", "chartreuse", "coral",
        "crimson", "hotpink", "deepskyblue", "dodgerblue", "firebrick", "forestgreen", "goldenrod"
    ]

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    taxi_colors = {}
    taxi_routes = {}
    start_end_points = []
    for taxi_id, group in df.groupby("TaxiID"):
        group = group.sort_values("DateTime")
        route = list(zip(group["Latitude"], group["Longitude"]))
        taxi_routes[taxi_id] = route

        if taxi_id not in taxi_colors:
            taxi_colors[taxi_id] = colors[len(taxi_colors) % len(colors)]

    first_taxi_id = list(taxi_routes.keys())[0]
    map_center = taxi_routes[first_taxi_id][0]

    taxi_routes_map = folium.Map(location=map_center, zoom_start=12)

    for taxi_id, route in taxi_routes.items():
        folium.Marker(location=route[0], popup=f"Start (Driver {taxi_id})", icon=folium.Icon(color="green")).add_to(taxi_routes_map)
        folium.Marker(location=route[-1], popup=f"End (Driver {taxi_id})", icon=folium.Icon(color="red")).add_to(taxi_routes_map)

        folium.PolyLine(
            route,
            color=taxi_colors[taxi_id],
            weight=4,
            opacity=0.8
        ).add_to(taxi_routes_map)
    folium_static(taxi_routes_map)

def visualize_5(df):

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    all_latitudes, all_longitudes = [], []
    taxi_routes = {}

    for taxi_id, group in df.groupby("TaxiID"):
        group = group.sort_values("DateTime")
        route = list(zip(group["Latitude"], group["Longitude"]))
        taxi_routes[taxi_id] = route

        all_latitudes.extend(group["Latitude"].tolist())
        all_longitudes.extend(group["Longitude"].tolist())

    color_scale = px.colors.qualitative.Set3
    taxi_colors = {}

    fig = go.Figure()

    for i, (taxi_id, route) in enumerate(taxi_routes.items()):
        taxi_colors[taxi_id] = color_scale[i % len(color_scale)]

        latitudes, longitudes = zip(*route)

        fig.add_trace(go.Scattermapbox(
            lat=latitudes,
            lon=longitudes,
            mode="lines",
            line=dict(width=3, color=taxi_colors[taxi_id]),
            name=f"Driver {taxi_id}"
        ))

        fig.add_trace(go.Scattermapbox(
            lat=[latitudes[0], latitudes[-1]],
            lon=[longitudes[0], longitudes[-1]],
            mode="markers",
            marker=dict(size=10, color=["green", "red"]),
            name=f"Driver {taxi_id}"
        ))

    min_lat, max_lat = min(all_latitudes), max(all_latitudes)
    min_lon, max_lon = min(all_longitudes), max(all_longitudes)

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            bounds=dict(
                west=min_lon, east=max_lon,
                south=min_lat, north=max_lat
            )
        ),
        width=1200,
        height=800,
        margin=dict(r=0, t=0, l=0, b=0),
        title="Taxi Routes"
    )

    st.plotly_chart(fig)

def visualize_clusters(df, trajectory_data, labels):
    st.subheader("Trajectory Clusters on Map")
    map_center = [df['Longitude'].mean(), df['Latitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    colors = ["red", "blue", "cyan", "purple", "green", "pink", "orange"]
    
    for i, traj in enumerate(trajectory_data):
        color = colors[labels[i] % len(colors)] 
        folium.PolyLine(traj, color=color, weight=2.5, opacity=1).add_to(m)
        
    folium_static(m)