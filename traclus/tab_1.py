import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
import matplotlib.colors
import matplotlib.pyplot as plt

from streamlit_folium import folium_static

def visualize_1(df_features):
    """Folium map showing selected trajectories with optional Beijing border."""
    st.subheader("Trajectories")

    if df_features is None or df_features.empty:
        st.info("No data for Folium map.")
        return

    all_ids = sorted(df_features['TaxiID'].unique())

    if "selected_ids" not in st.session_state:
        st.session_state.selected_ids = []

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Select all"):
            st.session_state.selected_ids = list(all_ids)
    with col2:
        if st.button("Reset all"):
            st.session_state.selected_ids = []

    selected_ids = st.multiselect(
        "Select Taxi IDs",
        options=all_ids,
        default=st.session_state.selected_ids,
        key="selected_ids"
    )

    if not selected_ids:
        st.warning("Please select at least one Taxi ID.")
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

    # Thêm chú giải màu sắc (legend)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 200px;
        height: auto;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
        ">
        <b>Legend (TaxiID):</b><br>
    """
    for tid in selected_ids:
        color = id_colors.get(tid, 'gray')
        legend_html += f"""<i style="background:{color};width:10px;height:10px;display:inline-block;margin-right:5px;"></i>{tid}<br>"""

    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    folium_static(m, width=1200, height=600)

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
    folium_static(m, width=1200,height=600)