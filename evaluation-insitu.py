#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:46:51 2025

@author: vpremier
"""

import pandas as pd
import geopandas as gpd


ta_shp = r'/mnt/CEPH_PROJECTS/SNOWCOP/Dati/data-merged/air_temperature.shp'
ta_csv = r'/mnt/CEPH_PROJECTS/SNOWCOP/Dati/data-merged/air_temperature.csv'

swe_shp = r'/mnt/CEPH_PROJECTS/SNOWCOP/Dati/data-merged/SWE.shp'
swe_csv = r'/mnt/CEPH_PROJECTS/SNOWCOP/Dati/data-merged/SWE.csv'

SW_shp = r'/mnt/CEPH_PROJECTS/SNOWCOP/Dati/data-merged/SW_radiation.shp'
SW_csv = r'/mnt/CEPH_PROJECTS/SNOWCOP/Dati/data-merged/SW_radiation.csv'

outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/albedo_pomeroy'

ta_df = pd.read_csv(ta_csv)
ta_stas = gpd.read_file(ta_shp)

swe_df = pd.read_csv(swe_csv)
swe_stas = gpd.read_file(swe_shp)

sw_df = pd.read_csv(SW_csv)
sw_stas = gpd.read_file(SW_shp)


# Get spatial bounds of the dataset (assuming `ta` is another GeoDataFrame)
x_min, x_max = float(ta.x.min()), float(ta.x.max())
y_min, y_max = float(ta.y.min()), float(ta.y.max())

start, end = ta.time.min().values, ta.time.max().values


dem = rasterio.open(dem_path)

# Option 1 — manual bounding box filter
selected = []
for idx, row in swe_stas.iterrows():
    x, y = row.geometry.x, row.geometry.y
    if (x_min <= x <= x_max) and (y_min <= y <= y_max):
        selected.append(row)
        
        
  
# Convert back to a GeoDataFrame
selected_stations = gpd.GeoDataFrame(selected, crs=swe_stas.crs)

print(f"✅ Selected {len(selected_stations)} stations within dataset bounds")
    

# --- Loop over selected stations ---
for _, s in selected_stations.iterrows():
    sta_name = s.get("sta_name", "unknown")
    source_file = s.get("source_fil", None)
    x_sta, y_sta = s.geometry.x, s.geometry.y
    
    print(f"\n📍 Processing station: {sta_name} ({source_file})")
    
    iy, ix = dem.index(x_sta, y_sta)
    
    matching_cols = [c for c in swe_df.columns if source_file in c]
    
    for col in matching_cols:

        # Extract observed series and ensure proper datetime index
        obs_series = swe_df[col]
        obs_series.index = pd.to_datetime(swe_df['Date'])
        
        # Align observed and modelled periods
        obs_series = obs_series[(obs_series.index >= start) & (obs_series.index <= end)]
        
        if obs_series.isna().all():
            print(f"⚠️ No overlapping data for {sta_name} — column: {col}")
            continue
    
    
        sca_ts = SCA.SCA[:, iy, ix].values
        pr_ts  = pr_reprojected[:, iy, ix].values
        status_ts = status[:, iy, ix].values
        melt_ts = melt[:, iy, ix]  # new melt array
        swe_ts = swe[:, iy, ix]  # new melt array
        sw_ts = SW.SW[:, iy, ix]  # new melt array
        temp_ts = ta.t2m[:, iy, ix]  # new melt array


     
        time_vals = SCA.time.values
     
        # --- Create figure with 3 subplots ---
        fig, (ax1, ax3, ax4) = plt.subplots(
            3, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={'hspace': 0.3}
        )
        
        # =========================
        # Subplot 1: Albedo, SCA, Precipitation, Status
        # =========================
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        color3 = 'tab:green'
        
        ax1.set_xlabel('')
        ax1.set_ylabel('SCA (scaled)', color=color1)
        ax1.plot(time_vals, sca_ts / np.nanmax(sca_ts), color=color2, linestyle='--', label='SCA (scaled)')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # --- Twin axis for precipitation ---
        ax2 = ax1.twinx()
        ax2.set_ylabel('Precipitation (mm)', color=color3)
        ax2.plot(time_vals, pr_ts, color=color3, alpha=0.6, label='Precipitation')
        ax2.tick_params(axis='y', labelcolor=color3)
        
        # --- Shading where status == 1 ---
        ax1.fill_between(
            time_vals,
            0,
            1,
            where=status_ts == 1,
            color='gray',
            alpha=0.3,
            transform=ax1.get_xaxis_transform(),
            label='Status = 1',
            zorder=0
        )
        
        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title(f"SCA and Precipitation at Pixel ({iy}, {ix})")
        
        # =========================
        # Subplot 2: SWE (Melt)
        # =========================
        ax3.plot(time_vals, swe_ts, color='tab:red', label='SWE reconstructed (mm)')
        ax3.plot(time_vals, obs_series.values, color='black', label='SWE station (mm)')
        ax3.set_ylabel('SWE (mm)')
        ax3.set_xlabel('Time')
        
        # Shading where status == 1
        ax3.fill_between(
            time_vals,
            np.min(swe_ts),
            np.max(swe_ts),
            where=status_ts == 1,
            color='gray',
            alpha=0.2,
            label='Status = 1',
        )
        ax3.legend(loc='upper right')
        ax3.set_title('SWE evolution')
        
        # =========================
        # Subplot 3: Temperature and Shortwave Radiation
        # =========================
        color_temp = 'tab:red'
        color_rad = 'tab:purple'
        
        ax4.set_ylabel('Temperature (°C)', color=color_temp)
        ax4.plot(time_vals, temp_ts, color=color_temp, label='Temperature')
        ax4.tick_params(axis='y', labelcolor=color_temp)
        
        # --- Twin axis for shortwave radiation ---
        ax5 = ax4.twinx()
        ax5.set_ylabel('Shortwave Radiation (W/m²)', color=color_rad)
        ax5.plot(time_vals, sw_ts, color=color_rad, linestyle='--', alpha=0.7, label='Shortwave Radiation')
        ax5.tick_params(axis='y', labelcolor=color_rad)
        
        # Legends
        ax4.legend(loc='upper left')
        ax5.legend(loc='upper right')
        ax4.set_xlabel('Time')
        ax4.set_title('Temperature and Shortwave Radiation')
        
        # =========================
        # Final formatting
        # =========================
        fig.suptitle(f"Hydro-Meteorological Variables at Station: {sta_name}", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for the suptitle
        plt.savefig(os.path.join(outdir, f'{sta_name}_{hy_xxxx}.png' ))

  

