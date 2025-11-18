#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 09:42:31 2025

@author: vpremier
"""
import os
import glob
import numpy as np
import time
import pandas as pd
import xarray as xr
from pyproj import CRS

from scipy.ndimage import binary_dilation

from utils import *

# def upload_sca(sca_path, dem_path, subbasin):
    
#     # read the dataframe
#     csv_path = sca_path.replace('.nc','.csv')
#     df = pd.read_csv(csv_path, index_col='Unnamed: 0')
#     df.index = pd.to_datetime(df.index)
        
#     stack_HR = xr.open_dataset(sca_path) 
    
#     _, mask_shape_cut, info = get_mask_info(dem_path, subbasin, resType='HR', pixel_ratio=10)
    
#     # get epsg code
#     crs = CRS.from_wkt(info['projection'])
#     epsg_code = crs.to_epsg()

    
#     mask_2d = xr.DataArray(mask_shape_cut, coords=[stack_HR.y, stack_HR.x], dims=["y", "x"])
#     stack_HR = stack_HR.where(mask_2d)
    
#     return stack_HR, df, info


def load_micromet(wd, hy, chunks={"time": 1, "x": 512, "y": 512}, verbose=True):
    
    t0 = time.perf_counter()
    
    date_start, date_end = f"20{hy[2:4]}-04-01", f"20{hy[4:6]}-03-31"
    
    # Extract start and end years from hy
    year_start, year_end = hy[2:4], hy[4:6]
    
    # Create file patterns for the two years
    year1_pattern = os.path.join(wd, f"*_downscaled_20{year_start}_*.nc")
    year2_pattern = os.path.join(wd, f"*_downscaled_20{year_end}_*.nc")
    
    # --- Detect Zarr vs NetCDF (filter by year pattern) ---
    zarr_dirs = sorted(
        glob.glob(os.path.join(wd, f"*20{year_start}*.zarr")) +
        glob.glob(os.path.join(wd, f"*20{year_end}*.zarr"))
    )
    nc_files = sorted(glob.glob(os.path.join(wd, f"*_downscaled_20{year_start}_*.nc")) +
                    glob.glob(os.path.join(wd, f"*_downscaled_20{year_end}_*.nc")))


    if zarr_dirs:
        if verbose:
            print(f"📦 Found {len(zarr_dirs)} Zarr dataset(s): {zarr_dirs}")
        datasets = [xr.open_zarr(z, chunks=chunks) for z in zarr_dirs]
        micromet = xr.concat(datasets, dim="time")

    elif nc_files:
        if verbose:
            print(f"📂 Found {len(nc_files)} NetCDF file(s).")
        datasets = [xr.open_dataset(f, chunks="auto") for f in nc_files]
        micromet = xr.concat(datasets, dim="time")

    else:
        raise FileNotFoundError(f"No Zarr or NetCDF files found for {hy} in {wd}")
    
    # Sort by time and select hydrological year
    micromet = micromet.sortby("time").sel(time=slice(date_start, date_end))
    micromet = micromet.rio.write_crs("EPSG:4326", inplace=True)
    

    if verbose:
        print(f"✅ Loaded micromet for {hy} in {time.perf_counter() - t0:.2f} s")
                                                    
        
    return micromet


def load_era5land(wd, hy):
    
    
    date_start, date_end = f"20{hy[2:4]}-04-01", f"20{hy[4:6]}-03-31"
    
    # Extract start and end years from hy
    year_start, year_end = hy[2:4], hy[4:6]
    
    # Create file patterns for the two years
    year1_pattern = os.path.join(wd, f"era_20{year_start}_*.nc")
    year2_pattern = os.path.join(wd, f"era_20{year_end}_*.nc")
    
    files = glob.glob(year1_pattern) + glob.glob(year2_pattern)


    # Open each file individually (no Dask)
    datasets = [xr.open_dataset(f) for f in files]
    
    # Open multiple NetCDF files for both years
    # Concatenate along time or valid_time
    era5land = xr.concat(datasets, dim="valid_time")
    era5land = era5land.sortby("valid_time")
    era5land = era5land.sel(valid_time=slice(date_start, date_end))
    

    era5land = era5land.rio.write_crs("EPSG:4326", inplace=True)  
    era5land = era5land.rio.set_spatial_dims(x_dim="longitude", 
                                             y_dim="latitude", 
                                             inplace=True).rename({'longitude': 'x', 
                                                                   'latitude': 'y'})
                                                                   
    era5land = era5land.rename({'valid_time': 'time'})
    
    
    var = era5land.resample(time='1D').mean() # solo delle ore positive?


    return var



def buffer(ds, n = 1):
    
    # Define a structuring element for the dilation (e.g., square/circular kernel)
    structuring_element = np.ones((2 * n + 1, 2 * n + 1), dtype=bool)
    
    # Apply dilation to each time slice
    def dilate_2d(data_slice):
        return binary_dilation(data_slice, structure=structuring_element)
    
    # Use xarray's apply_ufunc for efficient application across time slices
    dilated_data = xr.apply_ufunc(
        dilate_2d,
        ds,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True
    )
    return dilated_data



def upload_status(ta, pr):
    # 1: accumulation, 0: melting 
    status = (ta['t2m_interp'] < 1) & (pr['pr'] > 0)
    status = status.compute()
    
    status = buffer(status, n = 1)
    status = status.transpose('time', 'lat', 'lon')
    
    status = status.sortby('lat', ascending=False)
    
    return status



def set_all_to_one_per_timestep(mask):
    """
    If at least one '1' exists in any (x, y) location at a given time step, 
    set all values to '1' for that time step.

    Parameters:
    - mask (xarray.DataArray): Boolean mask (0 and 1) with dimensions (time, x, y).

    Returns:
    - xarray.DataArray: Updated mask where all values are set to 1 if any 1 exists in that time step.
    """
    # Check if there's at least one '1' in each time step
    any_ones_per_timestep = mask.max(dim=("x", "y"))

    # Expand back to (time, x, y) shape
    return mask.where(any_ones_per_timestep == 0, 1)