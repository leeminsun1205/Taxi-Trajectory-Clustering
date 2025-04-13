import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
import matplotlib.colors
import matplotlib.pyplot as plt

from streamlit_folium import folium_static

# --- Visualization Functions ---
def visualize_1(df_features):
    """Folium map showing selected trajectories with optional Beijing border."""
    st.subheader("Trajectories (Folium)")

    if df_features is None or df_features.empty:
        st.info("No data for Folium map.")
        return

    all_ids = df_features['TaxiID'].unique()
    selected_ids = st.multiselect("Select Taxi IDs to display:", all_ids, default=list(all_ids))

    if not selected_ids:
        st.warning("Please select at least one Taxi ID to display.")
        return

    filtered_df = df_features[df_features['TaxiID'].isin(selected_ids)]

    try:
        center = [filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()]
    except:
        center = [39.9, 116.4]

    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    cmap = plt.cm.get_cmap('tab20', len(all_ids))
    colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(all_ids))]
    id_colors = {tid: colors[i % len(colors)] for i, tid in enumerate(all_ids)}

    for tid, group in filtered_df.groupby('TaxiID'):
        g_sort = group.sort_values("DateTime")
        if len(g_sort) >= 2:
            route = list(zip(g_sort["Latitude"], g_sort["Longitude"]))
            popup = f"ID: {tid}<br>Pts: {len(route)}"
            if 'Speed_kmh' in g_sort:
                avg_spd = g_sort['Speed_kmh'].mean()
                if pd.notna(avg_spd):
                    popup += f"<br>Avg Spd: {avg_spd:.1f} km/h"
            folium.PolyLine(route, color=id_colors.get(tid, 'gray'),
                            weight=2, opacity=0.7, popup=popup).add_to(m)

    folium.LayerControl().add_to(m)
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