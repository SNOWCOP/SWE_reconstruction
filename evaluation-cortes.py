#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:58:12 2026

@author: vpremier
"""
import xarray as xr
import os
import glob
import rioxarray as rxr
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.enums import Resampling


shp_name = r'/mnt/CEPH_PROJECTS/SNOWCOP/AOI/Maipo.geojson'
gdf = gpd.read_file(shp_name)


csv_path = r'/mnt/CEPH_PROJECTS/SNOWCOP/SWE_Cortes/SWE_CHM_Yeso_2015.csv'
df_chm = pd.read_csv(csv_path)


# glacier
glacier_path = r'/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/Landsat/LC08/01_TEST_auxiliary_folder/glacier_mask.tif'
glacier = rxr.open_rasterio(glacier_path).squeeze()


# SWE dataset
swe = xr.open_dataset('/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/SWE_bias/Area06_hy1516_harm.nc')
swe = swe.rio.write_crs("EPSG:32719")

gdf = gdf.to_crs(swe.rio.crs)
gdf_wgs84 = gdf.to_crs("EPSG:4326")

swe_clip = swe.rio.clip(gdf.geometry, gdf.crs, drop=True)
swe_clip = swe_clip.where(swe_clip>=0)
swe_clip = swe_clip.where(glacier == 0)



cortes_dir = r'/mnt/CEPH_PROJECTS/SNOWCOP/SWE_Cortes/2015'
swe_cortes_list = glob.glob(cortes_dir + os.sep + '*.nc')



datasets = []

for i, f in enumerate(sorted(swe_cortes_list)):
    ds = xr.open_dataset(f)
    
    ds = ds.rio.write_crs("EPSG:4326")

    # Clip immediately
    ds_clip = ds.rio.clip(gdf_wgs84.geometry, gdf_wgs84.crs, drop=True)
    
    
    if i==0:
        glacier2 = glacier.rio.reproject_match(ds_clip)
        glacier2 = glacier2.rename({"y": "lat", "x": "lon"})

    
    # Extract date from filename (ADAPT THIS!)
    # Example: SWE_20150101.nc → 2015-01-01
    date_str = os.path.basename(f).split('_')[1].split('.')[0]
    time = pd.to_datetime(date_str, format="%Y%m%d")
    
    print(date_str)

    # Expand dataset with a time dimension
    ds_clip = ds_clip.expand_dims(time=[time])

    datasets.append(ds_clip)
    

# Concatenate along new time dimension
swe_cortes = xr.concat(datasets, dim="time")
swe_cortes = swe_cortes.astype("float32")

swe_cortes = swe_cortes.where(swe_cortes > 0, 0)

swe_cortes = swe_cortes.where(glacier2==0)

swe_cortes_masked = swe_cortes.rio.clip(
    gdf_wgs84.geometry,
    gdf_wgs84.crs,
    drop=False   # keeps grid, sets outside → NaN
)


mean_cortes = swe_cortes_masked["SWE"].mean(dim=("lat", "lon"))
mean_eurac = swe_clip["SWE"].mean(dim=("x", "y"))

mean_cortes["time"] = pd.to_datetime(mean_cortes["time"].values)
mean_eurac["time"] = pd.to_datetime(mean_eurac["time"].values)

plt.figure(figsize=(10, 5))

# Plot both time series
plt.plot(mean_cortes.time, mean_cortes.values, label="Cortes SWE", linewidth=2)
plt.plot(mean_eurac.time, mean_eurac.values, label="SNOWCOP SWE", linewidth=2)
# plt.plot(mean_eurac.time, df_chm.swe, label="CHM SWE", linewidth=2)

# Styling
plt.title("Spatially Averaged SWE Comparison")
plt.xlabel("Time")
plt.ylabel("SWE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.show()
plt.savefig(r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/Area06_hy1314.png')

dd

# swe_cortes_masked.to_netcdf("/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/SWE_bias/Area06_hy1314_cortes.nc")
# swe_clip.to_netcdf("/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/SWE_bias/Area06_hy1314_harm2.nc")


sca_cortes = (swe_cortes_masked > 50).SWE.sum(dim=("lat", "lon"))* 100/(swe_cortes_masked >= 0).SWE.sum(dim=("lat", "lon"))
sca_eurac = (swe_clip > 0).SWE.sum(dim=("x", "y")) * 100 /(swe_clip >= 0).SWE.sum(dim=("x", "y"))

# Plot both time series
plt.plot(mean_cortes.time, sca_cortes.values, label="Cortes SCA", linewidth=2)
plt.plot(mean_eurac.time, sca_eurac.values, label="EURAC SCA", linewidth=2)

# Styling
plt.title("SCA Comparison")
plt.xlabel("Time")
plt.ylabel("SCA")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


date_to_plot = "2015-10-01"
modis_scf = rxr.open_rasterio("/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/MOD10A1_NDSIthresh/MOD10A1_20150624.tif").squeeze()

modis_scf = modis_scf.rio.reproject_match(
    swe_cortes_masked.SWE.sel(time=date_to_plot),
    resampling=Resampling.cubic
)
modis_scf = modis_scf.rename({"y": "lat", "x": "lon"})

scf_clip = modis_scf.rio.clip(gdf_wgs84.geometry, gdf_wgs84.crs, drop=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmax = 200
# Cortes
swe_cortes_masked.SWE.sel(time=date_to_plot).plot(
    ax=axes[0],
    vmax=vmax,
    cmap="Blues"
)
axes[0].set_title("Cortes SWE")

# EURAC
swe_clip.SWE.sel(time=date_to_plot).plot(
    ax=axes[1],
    vmax=vmax,
    cmap="Blues"
)
axes[1].set_title("SNOWCOP SWE")

# MODIS SCF
scf_clip.where(modis_scf<=100).plot(
    ax=axes[2],
    cmap="viridis",
    vmin=0,
    vmax=100  # assuming SCF is %
)
axes[2].set_title("MODIS SCF")




plt.tight_layout()
plt.show()
vmax = 500
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cortes
swe_cortes_masked.SWE.sel(time=date_to_plot).plot(
    ax=axes[0],
    vmax=vmax,
    cmap="Blues"
)
axes[0].set_title("Cortes SWE")

# EURAC
swe_clip.SWE.sel(time=date_to_plot).plot(
    ax=axes[1],
    vmax=vmax,
    cmap="Blues"
)
axes[1].set_title("EURAC SWE")

plt.tight_layout()
plt.show()
