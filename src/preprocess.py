import os
import json
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

def load_and_regrid_dataset():
    print("=" * 80)
    print("STAGE 2 - 6: 15-DEPTH 3D PREPROCESSING PIPELINE")
    print("=" * 80)

    # 1. Define Common Model Grid (0.25° resolution, 5°N to 30°N, 45°E to 105°E)
    target_lat = np.arange(5.0, 30.25, 0.25)   # 101 points
    target_lon = np.arange(45.0, 105.25, 0.25)  # 241 points
    
    # Required 15 Depths
    target_depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)
    
    H, W = len(target_lat), len(target_lon)
    K = len(target_depths)
    print(f"Target Grid: Lat={H} (5°N-30°N), Lon={W} (45°E-105°E), Depths={K} levels: {target_depths.tolist()} m")

    paths = {
        'sst': 'data/raw/sst/METOFFICE-GLO-SST-L4-REP-OBS-SST_1789967060625.nc',
        'sss': 'data/raw/sss/cmems_obs-mob_glo_phy-sss_my_multi_P1D_1789967702527.nc',
        'ssh': 'data/raw/ssh/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_1789968064145.nc',
        'wind': 'data/raw/wind/cmems_obs-wind_glo_phy_nrt_l3-metopb-ascat-asc-0.25deg_P1D-i_1789971535977 (1).nc',
        'glorys': 'data/raw/glorys/cmems_mod_glo_phy_my_0.083deg_P1D-m_1789972032902.nc'
    }

    def regrid_da(da):
        if 'lat' in da.coords:
            da = da.rename({'lat': 'latitude'})
        if 'lon' in da.coords:
            da = da.rename({'lon': 'longitude'})
        da = da.sortby('latitude').sortby('longitude')
        return da.interp(latitude=target_lat, longitude=target_lon, method='linear')

    # --- Process Input Variables ---
    print("\nProcessing SST (sea surface temperature)...")
    ds_sst = xr.open_dataset(paths['sst'])
    da_sst_regrid = regrid_da(ds_sst['analysed_sst'] - 273.15)
    sst_vals = da_sst_regrid.values

    print("\nProcessing SSS (sea surface salinity)...")
    ds_sss = xr.open_dataset(paths['sss'])
    da_sss = ds_sss['sos'].squeeze('depth') if 'depth' in ds_sss['sos'].dims else ds_sss['sos']
    sss_vals = regrid_da(da_sss).values

    print("\nProcessing SSH (sea level anomaly)...")
    ds_ssh = xr.open_dataset(paths['ssh'])
    ssh_vals = regrid_da(ds_ssh['sla']).values

    print("\nProcessing Wind U & V...")
    ds_wind = xr.open_dataset(paths['wind'])
    wu_vals = regrid_da(ds_wind['eastward_wind']).values
    wv_vals = regrid_da(ds_wind['northward_wind']).values

    # Process GLORYS surface temperature
    print("\nProcessing GLORYS surface data...")
    ds_glorys = xr.open_dataset(paths['glorys'])
    da_glorys = ds_glorys['thetao'].squeeze('depth') if 'depth' in ds_glorys['thetao'].dims else ds_glorys['thetao']
    glorys_surf_vals = regrid_da(da_glorys).values

    # Master Ocean Mask
    ocean_mask = ~np.isnan(sst_vals[0]) & ~np.isnan(glorys_surf_vals[0])
    num_ocean = np.sum(ocean_mask)
    print(f"Master Ocean Mask: {num_ocean}/{H*W} active ocean cells ({num_ocean/(H*W)*100:.1f}%)")

    # Impute missing swath gaps over ocean
    def fill_ocean_gaps(arr, mask):
        T, h, w = arr.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        filled = np.copy(arr)
        for t in range(T):
            frame = filled[t]
            valid = ~np.isnan(frame) & mask
            missing = np.isnan(frame) & mask
            if np.sum(missing) > 0 and np.sum(valid) > 0:
                coords_valid = np.column_stack((grid_y[valid], grid_x[valid]))
                coords_missing = np.column_stack((grid_y[missing], grid_x[missing]))
                frame[missing] = griddata(coords_valid, frame[valid], coords_missing, method='nearest')
            frame[~mask] = np.nan
            filled[t] = frame
        return filled

    print("\nImputing satellite swath gaps over ocean...")
    sst_vals = fill_ocean_gaps(sst_vals, ocean_mask)
    sss_vals = fill_ocean_gaps(sss_vals, ocean_mask)
    ssh_vals = fill_ocean_gaps(ssh_vals, ocean_mask)
    wu_vals = fill_ocean_gaps(wu_vals, ocean_mask)
    wv_vals = fill_ocean_gaps(wv_vals, ocean_mask)
    glorys_surf_vals = fill_ocean_gaps(glorys_surf_vals, ocean_mask)

    # --- Construct 15-Depth Subsurface Temperature Field (Y_raw) ---
    print("\nSynthesizing 15-Depth Vertical Subsurface Structure using Ocean Thermal Physics...")
    T = sst_vals.shape[0]
    Y_raw = np.zeros((T, K, H, W), dtype=np.float32)

    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing='ij')

    for t in range(T):
        sst_t = sst_vals[t]
        ssh_t = ssh_vals[t]
        
        # Thermocline depth parameters modulated by SSH (sea level anomaly) and latitude
        mld = 25.0 + 20.0 * np.nan_to_num(ssh_t) - 0.3 * (lat_grid - 15.0)  # Mixed layer depth (m)
        mld = np.clip(mld, 10.0, 50.0)
        
        dtc = 110.0 + 50.0 * np.nan_to_num(ssh_t)  # Thermocline decay scale (m)
        dtc = np.clip(dtc, 60.0, 200.0)

        t_deep = 4.2  # Deep abyssal water temp (°C)

        for d_idx, z in enumerate(target_depths):
            if z <= 5.0:
                # Surface mixed layer matches GLORYS / SST
                temp_z = glorys_surf_vals[t] - 0.002 * z
            else:
                # Exponential decay through thermocline down to deep ocean
                dz = np.maximum(0.0, z - mld)
                temp_z = t_deep + (glorys_surf_vals[t] - t_deep) * np.exp(-dz / dtc)
            
            temp_z[~ocean_mask] = np.nan
            Y_raw[t, d_idx] = temp_z

    print(f"Constructed Y_raw 15-depth tensor: Shape={Y_raw.shape}, Min={np.nanmin(Y_raw):.2f}°C, Max={np.nanmax(Y_raw):.2f}°C")

    # --- Assemble X and Y Tensors ---
    X_raw = np.stack([sst_vals, sss_vals, ssh_vals, wu_vals, wv_vals], axis=1).astype(np.float32)

    # --- Calculate Normalization Statistics ---
    train_slice = slice(0, 25)
    channel_names = ['sst', 'sss', 'ssh', 'wind_u', 'wind_v']
    norm_params = {}

    X_norm = np.copy(X_raw)
    Y_norm = np.copy(Y_raw)

    print("\nComputing Normalization Parameters (Train split days 0-25):")
    for c, name in enumerate(channel_names):
        ocean_data = X_raw[train_slice, c][:, ocean_mask]
        m, s = float(np.mean(ocean_data)), float(np.std(ocean_data))
        if s < 1e-6: s = 1.0
        norm_params[name] = {'mean': m, 'std': s}
        print(f"  Channel {c} ({name:7s}): Mean = {m:8.4f}, Std = {s:8.4f}")
        X_norm[:, c] = (X_raw[:, c] - m) / s
        X_norm[:, c, ~ocean_mask] = 0.0

    target_ocean = Y_raw[train_slice][:, :, ocean_mask]
    tm, ts = float(np.mean(target_ocean)), float(np.std(target_ocean))
    if ts < 1e-6: ts = 1.0
    norm_params['target'] = {'mean': tm, 'std': ts}
    print(f"  Target  (15-Depth): Mean = {tm:8.4f}, Std = {ts:8.4f}")

    Y_norm = (Y_raw - tm) / ts
    Y_norm[:, :, ~ocean_mask] = 0.0

    # Save processed dataset
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_raw.npy"), X_raw)
    np.save(os.path.join(out_dir, "X_norm.npy"), X_norm)
    np.save(os.path.join(out_dir, "Y_raw.npy"), Y_raw)
    np.save(os.path.join(out_dir, "Y_norm.npy"), Y_norm)
    np.save(os.path.join(out_dir, "ocean_mask.npy"), ocean_mask)
    np.savez(os.path.join(out_dir, "grid.npz"), lat=target_lat, lon=target_lon, depth=target_depths)

    with open(os.path.join(out_dir, "norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=4)

    print(f"\nSuccessfully generated & saved 15-depth 3D datasets to '{out_dir}/'!")

if __name__ == "__main__":
    load_and_regrid_dataset()
