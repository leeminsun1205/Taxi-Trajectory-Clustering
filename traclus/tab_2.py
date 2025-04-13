import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Stop Detection (Cached) ---
def detect_stops(_df_single_taxi, speed_thresh_kmh=3.0, min_duration_s=300):
    """Identifies stops in a single taxi's trajectory (requires features)."""
    if _df_single_taxi is None or _df_single_taxi.empty: return pd.DataFrame()
    if 'Speed_kmh' not in _df_single_taxi.columns or 'TimeDiff_s' not in _df_single_taxi.columns:
        st.warning("Stop detection requires Speed_kmh and TimeDiff_s.")
        df_out = _df_single_taxi.copy(); df_out['IsStop'] = False
        return df_out

    df = _df_single_taxi.sort_values('DateTime').copy()
    df['IsStop'] = False
    start_idx, current_duration = None, 0.0

    for i in range(len(df)):
        speed = df['Speed_kmh'].iloc[i]
        time_diff = df['TimeDiff_s'].iloc[i] # Duration *since previous point*

        is_slow = pd.notna(speed) and speed < speed_thresh_kmh
        is_valid_interval = pd.notna(time_diff) and time_diff > 0

        if is_slow and is_valid_interval:
            if start_idx is None: start_idx = i # Mark start of potential stop
            current_duration += time_diff # Accumulate duration
        else:
            # End of potential stop sequence (or invalid interval)
            if start_idx is not None and current_duration >= min_duration_s:
                df.loc[df.index[start_idx], 'IsStop'] = True # Mark the START point of the stop
            start_idx, current_duration = None, 0.0 # Reset

    # Check if trajectory ends in a stop
    if start_idx is not None and current_duration >= min_duration_s:
        df.loc[df.index[start_idx], 'IsStop'] = True

    return df

def visualize_single_trajectory_animation_plotly(df_single_taxi_with_stops, speed_multiplier=1.0):
    if df_single_taxi_with_stops is None or df_single_taxi_with_stops.empty:
        st.warning("No data for selected taxi animation.")
        return None
    required = ['DateTime', 'Latitude', 'Longitude', 'TaxiID', 'IsStop', 'Speed_kmh']
    if not all(col in df_single_taxi_with_stops.columns for col in required):
        st.error("Animation failed: Missing required columns.")
        return None

    df_anim = df_single_taxi_with_stops.copy()
    df_anim['DateTime'] = pd.to_datetime(df_anim['DateTime'], errors='coerce')
    df_anim = df_anim.dropna(subset=['DateTime', 'Latitude', 'Longitude']) \
                     .sort_values('DateTime') \
                     .reset_index(drop=True)
    if len(df_anim) < 2:
        st.warning("Need >= 2 points for animation.")
        return None

    df_anim['Speed_kmh_display'] = df_anim['Speed_kmh'].fillna(0.0)

    # Sử dụng tọa độ điểm xuất phát làm trung tâm ban đầu
    start_lat = df_anim["Latitude"].iloc[0]
    start_lon = df_anim["Longitude"].iloc[0]

    lat_min, lat_max = df_anim['Latitude'].min(), df_anim['Latitude'].max()
    lon_min, lon_max = df_anim['Longitude'].min(), df_anim['Longitude'].max()
    lat_pad = (lat_max - lat_min) * 0.1 or 0.02
    lon_pad = (lon_max - lon_min) * 0.1 or 0.02
    bounds = {"west": lon_min - lon_pad, "east": lon_max + lon_pad,
              "south": lat_min - lat_pad, "north": lat_max + lat_pad}

    base_frame_duration_ms = 150
    frame_duration = max(20, int(base_frame_duration_ms / speed_multiplier))
    transition_duration = 0

    fig = go.Figure()

    # Trace 0: Đoạn đường di chuyển (dynamic)
    fig.add_trace(go.Scattermapbox(
        lat=[start_lat],
        lon=[start_lon],
        mode='lines+markers',
        line=dict(width=2, color='rgba(255, 0, 0, 0.7)'),
        marker=dict(size=4, color='rgba(255, 0, 0, 0.7)'),
        name="Trajectory"
    ))
    # Trace 1: Marker cho các điểm dừng (static)
    stops_df = df_anim[df_anim['IsStop']]
    fig.add_trace(go.Scattermapbox(
        lat=stops_df["Latitude"],
        lon=stops_df["Longitude"],
        mode='markers',
        marker=dict(size=8, color='purple', symbol='circle'),
        name='Detected Stops',
        hoverinfo='text',
        text=[f"Stop @ {dt.strftime('%H:%M:%S')}" for dt in stops_df['DateTime']]
    ))
    # Trace 2: Marker cho điểm Start (static)
    fig.add_trace(go.Scattermapbox(
        lat=[start_lat],
        lon=[start_lon],
        mode='markers',
        marker=dict(size=14, color='green', symbol='circle'),
        name='Start',
        hoverinfo='text',
        text=f"Start: {df_anim['DateTime'].iloc[0].strftime('%H:%M:%S')}"
    ))
    # Trace 3: Marker cho điểm End (static) – chuyển sang màu xanh dương
    fig.add_trace(go.Scattermapbox(
        lat=[df_anim["Latitude"].iloc[-1]],
        lon=[df_anim["Longitude"].iloc[-1]],
        mode='markers',
        marker=dict(size=14, color='#1f77b4', symbol='circle'),
        name='End',
        hoverinfo='text',
        text=f"End: {df_anim['DateTime'].iloc[-1].strftime('%H:%M:%S')}"
    ))
    # Trace 4: Marker di chuyển (dynamic)
    fig.add_trace(go.Scattermapbox(
        lat=[start_lat],
        lon=[start_lon],
        mode='markers',
        marker=dict(size=16, color='blue', symbol='arrow', allowoverlap=True),
        name='Current Position',
        customdata=df_anim[['DateTime', 'Speed_kmh_display']].iloc[[0]],
        hovertemplate="<b>Current:</b><br>Time: %{customdata[0]|%Y-%m-%d %H:%M:%S}<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>"
    ))

    frames = []
    for k in range(len(df_anim)):
        row = df_anim.iloc[k]
        # Đoạn đường tạm tính từ điểm đầu đến frame hiện tại
        dynamic_path = go.Scattermapbox(
            lat=df_anim["Latitude"].iloc[:k+1].tolist(),
            lon=df_anim["Longitude"].iloc[:k+1].tolist(),
            mode='lines+markers',
            line=dict(width=2, color='rgba(255, 0, 0, 0.7)'),
            marker=dict(size=4, color='rgba(255, 0, 0, 0.7)'),
            name="Trajectory"
        )
        # Marker di chuyển tại điểm hiện tại
        moving_marker = go.Scattermapbox(
            lat=[row["Latitude"]],
            lon=[row["Longitude"]],
            mode='markers',
            marker=dict(size=16, color='blue', symbol='arrow', allowoverlap=True),
            name='Current Position',
            customdata=[[row['DateTime'], row['Speed_kmh_display']]],
            hovertemplate="<b>Current:</b><br>Time: %{customdata[0]|%Y-%m-%d %H:%M:%S}<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>"
        )
        info_text = f"Time: {row['DateTime'].strftime('%Y-%m-%d %H:%M:%S')}<br>Speed: {row['Speed_kmh_display']:.1f} km/h"
        frames.append(go.Frame(
            name=str(k),
            data=[dynamic_path, moving_marker],
            traces=[0, 4],
            layout=go.Layout(
                # Cập nhật annotation và mapbox.center để theo dõi marker
                annotations=[dict(
                    text=info_text,
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.7)',
                    borderpad=4
                )],
                mapbox_center=dict(lat=row["Latitude"], lon=row["Longitude"])
            )
        ))
    fig.frames = frames

    fig.update_layout(
        height=700,
        mapbox_style="carto-positron",
        mapbox_bounds=bounds,
        # Ban đầu căn giữa ở điểm xuất phát
        mapbox_center=dict(lat=start_lat, lon=start_lon),
        mapbox_zoom=12,
        margin=dict(r=5, t=10, l=5, b=5),
        showlegend=True,
        # Đặt legend trở về vị trí ban đầu (góc dưới trái) để không đè lên annotation tốc độ
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    bgcolor='rgba(255,255,255,0.7)'),
        annotations=[dict(
            text=f"Time: {df_anim['DateTime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')}<br>Speed: {df_anim['Speed_kmh_display'].iloc[0]:.1f} km/h",
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.7)',
            borderpad=4
        )],
        updatemenus=[dict(
            type="buttons",
            direction="right",
            showactive=False,
            # Nút Play/Pause được đặt cao hơn slider
            x=0.95,
            y=0.12,
            xanchor="right",
            yanchor="bottom",
            bgcolor='rgba(200,200,200,0.9)',
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": frame_duration, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": transition_duration},
                                 "mode": "immediate"}]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate", "transition": {"duration": 0}}]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Point Index: ", "font": {"size": 14, "color": "black"}},
            pad={"b": 30, "t": 50},
            len=0.8,
            x=0.1,
            y=0,
            yanchor="top",
            activebgcolor="blue",
            steps=[{"args": [[f.name],
                              {"frame": {"duration": frame_duration, "redraw": True},
                               "mode": "immediate", "transition": {"duration": transition_duration}}],
                    "label": str(k),
                    "method": "animate"} for k, f in enumerate(fig.frames)]
        )]
    )

    return fig

