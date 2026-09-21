import os
import json
import numpy as np
import xarray as xr
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

# Page Configuration
st.set_page_config(
    page_title="OceanEmbed - Subsurface Temperature Reconstruction",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">OceanEmbed</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Subsurface Ocean Temperature Reconstruction<br>North Indian Ocean | Daily | 0–1000 m</div>', unsafe_allow_html=True)

# Data Loading with Cache
@st.cache_data
def load_data():
    pred_nc = "data/processed/predictions_3d.nc"
    if not os.path.exists(pred_nc):
        pred_nc = "data/processed/predictions_3d_v2.nc"
        
    pred_npy = "data/processed/preds_3d.npy"
    raw_path = "data/processed/Y_raw.npy"
    mask_path = "data/processed/ocean_mask.npy"
    grid_path = "data/processed/grid.npz"
    metrics_path = "reports/metrics.json"

    grid = np.load(grid_path)
    lats = grid['lat']
    lons = grid['lon']
    depths = grid['depth']
    
    if os.path.exists(pred_npy):
        preds_3d = np.load(pred_npy)
        dates = [f"2020-01-{d+1:02d}" for d in range(preds_3d.shape[0])]
    elif os.path.exists(pred_nc):
        with xr.open_dataset(pred_nc) as ds:
            preds_3d = ds['predicted_temperature'].values
            dates = [str(d)[:10] for d in ds['time'].values]
    else:
        st.error("Prediction file not found. Please run prediction pipeline.")
        st.stop()

    Y_raw = np.load(raw_path) if os.path.exists(raw_path) else None
    ocean_mask = np.load(mask_path)
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    return preds_3d, dates, Y_raw, ocean_mask, lats, lons, depths, metrics

try:
    preds_3d, dates, Y_raw, ocean_mask, lats, lons, depths, metrics = load_data()
except Exception as e:
    st.error(f"Error loading prediction data: {e}")
    st.stop()

# Helper function to find nearest valid ocean cell
def get_nearest_ocean_cell(lat_val, lon_val, lats, lons, mask):
    lat_idx = np.abs(lats - lat_val).argmin()
    lon_idx = np.abs(lons - lon_val).argmin()
    
    if mask[lat_idx, lon_idx]:
        return lat_idx, lon_idx, lats[lat_idx], lons[lon_idx]
        
    # If land, find nearest ocean cell
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing='ij')
    ocean_lats = grid_lat[mask]
    ocean_lons = grid_lon[mask]
    
    ocean_coords = np.column_stack((ocean_lats, ocean_lons))
    target_coord = np.array([[lats[lat_idx], lons[lon_idx]]])
    
    distances = cdist(target_coord, ocean_coords)
    nearest_idx = np.argmin(distances)
    
    nearest_lat = ocean_coords[nearest_idx, 0]
    nearest_lon = ocean_coords[nearest_idx, 1]
    
    n_lat_idx = np.abs(lats - nearest_lat).argmin()
    n_lon_idx = np.abs(lons - nearest_lon).argmin()
    
    return n_lat_idx, n_lon_idx, nearest_lat, nearest_lon


# Sidebar Controls
st.sidebar.header("Dashboard Controls")

selected_date_idx = st.sidebar.selectbox("Select Date", range(len(dates)), format_func=lambda i: dates[i])
selected_depth_idx = st.sidebar.selectbox("Select Depth Level (2D View)", range(len(depths)), format_func=lambda i: f"{depths[i]:.0f} meters")

st.sidebar.markdown("---")
st.sidebar.header("Location")
selected_lat = st.sidebar.slider("Latitude (°N)", float(lats.min()), float(lats.max()), 15.0, step=0.25)
selected_lon = st.sidebar.slider("Longitude (°E)", float(lons.min()), float(lons.max()), 70.0, step=0.25)

# Calculate nearest valid ocean cell
lat_idx, lon_idx, valid_lat, valid_lon = get_nearest_ocean_cell(selected_lat, selected_lon, lats, lons, ocean_mask)
is_snapped = (valid_lat != selected_lat or valid_lon != selected_lon)

if is_snapped:
    st.sidebar.warning(f"Selected location ({selected_lat}°N, {selected_lon}°E) is over land. Automatically snapped to nearest ocean cell: ({valid_lat}°N, {valid_lon}°E).")
else:
    st.sidebar.success(f"Valid ocean location selected: ({valid_lat}°N, {valid_lon}°E).")


# Metrics Header
if metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{metrics.get('rmse_degC', 0.0):.3f} °C")
    with col2:
        st.metric("Correlation", f"{metrics.get('correlation', 0.0):.4f}")
    with col3:
        st.metric("Bias", f"{metrics.get('bias_degC', 0.0):.3f} °C")
    with col4:
        st.metric("Depth Coverage", f"{len(depths)} Levels (0-{int(depths[-1])}m)")

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Temperature Map", "3D Temperature", "Vertical Profile", "Model Evaluation"])

with tab1:
    st.subheader(f"2D Subsurface Temperature Field on {dates[selected_date_idx]} (Depth: {depths[selected_depth_idx]:.0f} m)")
    
    pred_field = preds_3d[selected_date_idx, selected_depth_idx]
    
    fig = px.imshow(
        pred_field,
        x=lons,
        y=lats,
        origin='lower',
        color_continuous_scale='Thermal',
        labels={'x': 'Longitude (°E)', 'y': 'Latitude (°N)', 'color': 'Temp (°C)'},
        title=f"OceanEmbed Predicted Subsurface Temperature (°C) at {depths[selected_depth_idx]:.0f}m"
    )
    
    fig.add_trace(go.Scatter(
        x=[valid_lon],
        y=[valid_lat],
        mode='markers+text',
        marker=dict(color='cyan', size=14, symbol='x', line=dict(width=2, color='black')),
        name='Selected Location',
        text=[f"({valid_lat}°N, {valid_lon}°E)"],
        textposition="top center"
    ))

    fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: Use the sliders in the sidebar to change the selected location. Invalid land points will automatically snap to the nearest ocean cell.")

with tab2:
    st.subheader("3D Subsurface Temperature")
    st.caption("Stacked 3D ocean temperature layers across depth levels (0m to 1000m). Rotate, zoom, and inspect temperature fields in 3D space.")

    # Downsample spatially for smooth rendering
    lat_sub = lats[::3]
    lon_sub = lons[::3]
    lat_idx_sub = np.arange(0, len(lats), 3)
    lon_idx_sub = np.arange(0, len(lons), 3)

    # Allow user to choose which depth layers to show in 3D
    default_selected_depths = [0.0, 30.0, 100.0, 200.0, 500.0, 1000.0]
    available_depths_list = [float(d) for d in depths]
    
    selected_3d_depths = st.multiselect(
        "Select Depths to Display in 3D Stack:",
        options=available_depths_list,
        default=[d for d in default_selected_depths if d in available_depths_list],
        format_func=lambda d: f"{d:.0f} meters"
    )

    if not selected_3d_depths:
        st.info("Please select at least one depth level to render the 3D map.")
    else:
        fig_3d = go.Figure()

        vmin = float(np.nanmin(preds_3d[selected_date_idx]))
        vmax = float(np.nanmax(preds_3d[selected_date_idx]))

        for d_val in sorted(selected_3d_depths):
            d_i = list(depths).index(d_val)
            temp_slice = preds_3d[selected_date_idx, d_i][np.ix_(lat_idx_sub, lon_idx_sub)]
            
            z_plane = np.full((len(lat_sub), len(lon_sub)), -d_i)

            fig_3d.add_trace(go.Surface(
                z=z_plane,
                x=lon_sub,
                y=lat_sub,
                surfacecolor=temp_slice,
                colorscale='Thermal',
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title='Temp (°C)'),
                name=f"{d_val:.0f}m Depth",
                showscale=(d_val == sorted(selected_3d_depths)[0]),
                hoverinfo='x+y+z+text',
                text=np.array([[f"Depth: {d_val:.0f}m<br>Temp: {temp_slice[i, j]:.2f}°C" for j in range(len(lon_sub))] for i in range(len(lat_sub))])
            ))

        z_ticks = [-i for i in range(len(depths)) if depths[i] in selected_3d_depths]
        z_labels = [f"{depths[i]:.0f}m" for i in range(len(depths)) if depths[i] in selected_3d_depths]

        fig_3d.update_layout(
            title=f"3D Stacked Ocean Temperature Field (Date: {dates[selected_date_idx]})",
            scene=dict(
                xaxis_title='Longitude (°E)',
                yaxis_title='Latitude (°N)',
                zaxis_title='Depth (meters)',
                zaxis=dict(
                    tickmode='array',
                    tickvals=z_ticks,
                    ticktext=z_labels,
                    autorange='reversed'
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.8, y=1.2, z=1.4),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
            ),
            height=700,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader(f"Vertical Subsurface Temperature Profile at ({valid_lat}°N, {valid_lon}°E)")
    
    pred_profile = preds_3d[selected_date_idx, :, lat_idx, lon_idx]
    actual_profile = Y_raw[selected_date_idx, :, lat_idx, lon_idx] if Y_raw is not None else None

    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=pred_profile,
        y=depths,
        mode='lines+markers',
        name='OceanEmbed Prediction',
        line=dict(color='#0284c7', width=3),
        marker=dict(size=6)
    ))
    
    if actual_profile is not None:
        fig_prof.add_trace(go.Scatter(
            x=actual_profile,
            y=depths,
            mode='lines+markers',
            name='GLORYS Reference',
            line=dict(color='#ef4444', width=2, dash='dash'),
            marker=dict(size=6)
        ))

    fig_prof.update_layout(
        title=f"Subsurface Temperature Profile (0m to 1000m) on {dates[selected_date_idx]}",
        xaxis_title="Temperature (°C)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        height=480
    )
    st.plotly_chart(fig_prof, use_container_width=True)

with tab4:
    st.subheader("Model Evaluation")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Training Loss**")
        if os.path.exists("reports/loss_curve.png"):
            st.image("reports/loss_curve.png", use_container_width=True)
        else:
            st.info("Loss curve plot not found.")
            
    with col_b:
        st.markdown("**RMSE by Depth**")
        if os.path.exists("reports/rmse_by_depth.png"):
            st.image("reports/rmse_by_depth.png", use_container_width=True)
        else:
            st.info("RMSE by depth plot not found.")
            
    st.markdown("---")
    
    col_c, col_d = st.columns(2)
    
    if "depth_metrics" in metrics:
        depth_data = metrics["depth_metrics"]
        d_vals = [m["depth_m"] for m in depth_data]
        corr_vals = [m["correlation"] for m in depth_data]
        bias_vals = [m["bias"] for m in depth_data]
        
        with col_c:
            st.markdown("**Correlation by Depth**")
            fig_corr = go.Figure(go.Scatter(x=corr_vals, y=d_vals, mode='lines+markers', line=dict(color='green', width=2)))
            fig_corr.update_layout(xaxis_title="Pearson Correlation", yaxis_title="Depth (m)", yaxis=dict(autorange="reversed"), height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_d:
            st.markdown("**Bias by Depth**")
            fig_bias = go.Figure(go.Scatter(x=bias_vals, y=d_vals, mode='lines+markers', line=dict(color='purple', width=2)))
            fig_bias.update_layout(xaxis_title="Bias (°C)", yaxis_title="Depth (m)", yaxis=dict(autorange="reversed"), height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bias, use_container_width=True)
    else:
        st.info("Detailed depth metrics not found in metrics.json.")

st.markdown("---")
st.caption("OceanEmbed | Built with PyTorch, Xarray, Streamlit & Plotly")
