import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

def visualize_congestion(df_features, speed_thresh_kmh=10):
    """Visualizes congestion points using Folium MarkerCluster."""
    st.subheader(f"Congestion Hotspots (< {speed_thresh_kmh} km/h)")
    if df_features is None or df_features.empty or 'Speed_kmh' not in df_features.columns:
        st.info("No speed data for congestion analysis."); return

    congestion = df_features[(df_features['Speed_kmh'].notna()) & (df_features['Speed_kmh'] < speed_thresh_kmh)].copy()
    if congestion.empty: st.info(f"No points below {speed_thresh_kmh} km/h found."); return

    st.write(f"Found **{len(congestion):,}** potential congestion points.")
    try: center = [congestion['Latitude'].mean(), congestion['Longitude'].mean()]
    except: center = [39.9, 116.4]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    # Use MarkerCluster for performance
    mc = MarkerCluster(name="Congestion Points").add_to(m)
    for idx, r in congestion.iterrows():
         popup = (f"Time: {r['DateTime'].strftime('%H:%M:%S')}<br>"
                  f"Speed: {r['Speed_kmh']:.1f} km/h<br>ID: {r['TaxiID']}")
         folium.CircleMarker(location=[r['Latitude'], r['Longitude']], radius=4,
                             color='orange', fill=True, fill_color='red', fill_opacity=0.6,
                             popup=popup).add_to(mc)
    folium.LayerControl().add_to(m) # Optional layer control
    folium_static(m, width=1200, height=600)