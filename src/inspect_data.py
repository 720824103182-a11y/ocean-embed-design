import os
import glob
import xarray as xr
import numpy as np

def inspect_nc_file(filepath, log_file):
    def p(text=""):
        print(text)
        log_file.write(text + "\n")

    p("=" * 80)
    p(f"FILE: {filepath}")
    p("=" * 80)
    
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        p(f"Error opening file {filepath}: {e}\n")
        return

    p("\n--- DATASET OVERVIEW ---")
    p(str(ds))
    
    p("\n--- DIMENSIONS ---")
    for d, s in ds.sizes.items():
        p(f"  {d}: {s}")

    p("\n--- COORDINATES ---")
    for name, coord in ds.coords.items():
        val = coord.values
        min_val = np.nanmin(val) if val.size > 0 and np.issubdtype(val.dtype, np.number) else "N/A"
        max_val = np.nanmax(val) if val.size > 0 and np.issubdtype(val.dtype, np.number) else "N/A"
        
        if np.issubdtype(val.dtype, np.datetime64):
            min_val = str(np.min(val))
            max_val = str(np.max(val))
            
        res = "N/A"
        if len(val) > 1 and np.issubdtype(val.dtype, np.number):
            diffs = np.diff(val)
            res = f"{np.mean(diffs):.4f} (approx)"

        p(f"  Coord: '{name}' | Shape: {coord.shape} | Range: [{min_val} to {max_val}] | Resolution: {res}")

    p("\n--- DATA VARIABLES ---")
    for var_name, var in ds.data_vars.items():
        attrs = var.attrs
        units = attrs.get('units', 'N/A')
        standard_name = attrs.get('standard_name', 'N/A')
        long_name = attrs.get('long_name', 'N/A')
        fill_val = attrs.get('_FillValue', attrs.get('missing_value', 'N/A'))
        
        p(f"  Variable: '{var_name}'")
        p(f"    - Dimensions: {var.dims}")
        p(f"    - Shape: {var.shape}")
        p(f"    - Dtype: {var.dtype}")
        p(f"    - Standard Name: {standard_name}")
        p(f"    - Long Name: {long_name}")
        p(f"    - Units: {units}")
        p(f"    - Fill Value: {fill_val}")

    ds.close()
    p("\n" + "=" * 80 + "\n")

def main():
    raw_dir = os.path.join("data", "raw")
    nc_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.nc"), recursive=True))
    
    with open("inspect_report.txt", "w", encoding="utf-8") as log_file:
        if not nc_files:
            log_file.write(f"No .nc files found in {raw_dir}\n")
            print(f"No .nc files found in {raw_dir}")
            return

        header = f"Found {len(nc_files)} NetCDF file(s) for inspection:\n"
        for f in nc_files:
            header += f" - {f}\n"
        header += "\n" + "="*80 + "\n"
        print(header)
        log_file.write(header)

        for f in nc_files:
            inspect_nc_file(f, log_file)

if __name__ == "__main__":
    main()
