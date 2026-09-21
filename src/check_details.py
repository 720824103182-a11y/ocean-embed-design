import os
import glob
import xarray as xr
import numpy as np

files = {
    'SST': 'data/raw/sst/METOFFICE-GLO-SST-L4-REP-OBS-SST_1789967060625.nc',
    'SSS': 'data/raw/sss/cmems_obs-mob_glo_phy-sss_my_multi_P1D_1789967702527.nc',
    'SSH': 'data/raw/ssh/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_1789968064145.nc',
    'WIND': 'data/raw/wind/cmems_obs-wind_glo_phy_nrt_l3-metopb-ascat-asc-0.25deg_P1D-i_1789971535977 (1).nc',
    'GLORYS': 'data/raw/glorys/cmems_mod_glo_phy_my_0.083deg_P1D-m_1789972032902.nc'
}

for name, path in files.items():
    print("=" * 80)
    print(f"DATASET: {name}")
    print(f"Path: {path}")
    ds = xr.open_dataset(path)
    
    print(f"Time range: {ds['time'].values[0]} to {ds['time'].values[-1]} ({len(ds['time'])} steps)")
    print(f"Latitude range: {ds['latitude'].values[0]} to {ds['latitude'].values[-1]} ({len(ds['latitude'])} points)")
    print(f"Longitude range: {ds['longitude'].values[0]} to {ds['longitude'].values[-1]} ({len(ds['longitude'])} points)")
    if 'depth' in ds.coords:
        print(f"Depth levels: {ds['depth'].values}")
    else:
        print("Depth levels: None (Surface only)")
        
    for varname in ds.data_vars:
        da = ds[varname]
        vals = da.values
        nan_cnt = np.isnan(vals).sum()
        total_cnt = vals.size
        valid_cnt = total_cnt - nan_cnt
        print(f"  Var: '{varname}' | Dtype: {da.dtype} | Units: {da.attrs.get('units', 'N/A')} | Min: {np.nanmin(vals):.3f} | Max: {np.nanmax(vals):.3f} | NaNs: {nan_cnt}/{total_cnt} ({nan_cnt/total_cnt*100:.1f}%)")
    ds.close()
    print("\n")
