#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:53:12 2025

@author: vpremier
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def get_status_and_delta(SCA, ta, era5, temp_thres=1.0, prec_thres=1.0):
    """
    Compute:
      (1) Boolean accumulation mask: True = accumulation, False = melting/other
      (2) Fraction of precipitation contributing to SWE accumulation (per timestep)

    Parameters
    ----------
    SCA : xarray.DataArray
        Snow cover area classification (dims: time,x,y)
    ta : xarray.DataArray or Dataset
        Air temperature time series (°C) with variable 't2m'
    era5 : xarray.Dataset or DataArray
        ERA5-Land dataset containing variable 'tp' (precipitation, meters)
    temp_thres : float
        Temperature threshold for accumulation (default = 1°C)
    prec_thres : float
        Precipitation threshold for accumulation (default = 1 mm/day)

    Returns
    -------
    status : xarray.DataArray (bool)
        True where accumulation conditions are met
        False where melting or no-precipitation
    delta : xarray.DataArray (float32)
        Fraction of total accumulation at each timestep
        Sum over time per pixel ≈ 1 (where accumulation occurs)
    """

    # ERA5 tp : precipitation
    pr = era5['tp'] 
    pr = pr.rio.write_crs("EPSG:4326", inplace=True)
    

    # Reproject onto SCA grid
    pr_reprojected = pr.rio.reproject_match(SCA)

    # Boolean accumulation mask
    status = xr.where(
                    (ta['t2m'] < temp_thres) & (pr_reprojected > prec_thres),
                    1,
                    -1
                ).astype('int8')
    
    
    # Masked precipitation
    masked_pr = pr_reprojected.where(status)
    
    # Total accumulation per pixel (lazy reduction)
    sum_pr = masked_pr.sum(dim='time')

    # Safe denominator → avoid division by zero
    safe_sum_pr = sum_pr.where(sum_pr > 0)

    # Redistribute accumulation precipitation fractionally
    delta = masked_pr / safe_sum_pr

    # No accumulation → 0
    delta = delta.fillna(0).astype('float32')

    # Load explicitly for speed in later computations
    status = status.load()
    delta = delta.load()

    return status, delta, pr_reprojected




def compute_state_and_accumulation(SCA, melt, status, delta):
    """
    Compute cumulative snow accumulation (as a fraction of total precipitation,
                                          ie. the deltas)
    and total melt energy that is assumed to correspond to the total 
    accumulation (mass conservation) for each snow period between melt-out 
    events.

    The function loops over daily Snow Cover Area (SCA) maps and tracks
    accumulation and melting through time for each pixel. Each "snow period"
    (from the first accumulation until complete melt-out) is treated as a 
    self-contained event. The cumulative precipitation fraction (sca_sum)
    and total melt energy (tot_acc) are both reset whenever the pixel becomes
    snow-free.

    Parameters
    ----------
    SCA : xr.DataArray
        Snow classification (SCA) over time with dims ('time','x','y')
        Values: 0 = no snow, 100 = snow, 205 = cloud/no data
    melt : xr.DataArray
        Melt energy (e.g., degree-day or energy balance proxy) with same time and 
        spatial dimensions as SCA.
    status : xr.DataArray (int)
        Accumulation/melting state mask, with values:
            1  = accumulation (cold + precipitation)
           -1  = melting / no accumulation
    delta : xr.DataArray (float)
        Fraction of daily precipitation contributing to accumulation, 
        such that the sum over time during the hydrological year ≈ 1 per pixel.
    a : float
        Degree-day melt factor (°C * mm or similar units)

    Returns
    -------
    sca_sum_xr : xr.DataArray
        Cumulative fraction of precipitation per snow period.
    tot_acc_xr : xr.DataArray
        Total melt energy accumulated during each snow period (same logic as sca_sum_xr).
    
    Notes
    -----
    - The `changes` array is used internally to track pixel transitions:
        +2 → first day of snow accumulation (snow onset)
        +1 → snow-covered and accumulating
         0 → snow-free (after the date of snow end)
        -1 → snow-covered but melting
        -2 → last melt-out
    - Both `sca_sum` and `tot_acc` are reset to zero whenever the pixel becomes snow-free.
    - Cloud/no-data pixels (205) are treated as snow-covered for continuity.
    """

    # --- Initialize dimensions and arrays ---
    time = SCA.time
    dim =  tuple(SCA.sizes.values())  # (time,x,y)
    
    sca_sum = np.zeros(dim, dtype=np.float32)
    tot_acc = np.zeros(dim, dtype=np.float32)
    changes = np.zeros(dim, dtype=np.float32)
    
    # === Time iteration over SCA ===
    for i in range(len(time) - 1):
        date = pd.Timestamp(time[i + 1].values).strftime("%Y-%m-%d")
        print(f"Processing {i}: {date}")
    
        # Snow cover for previous and current day
        snow_prev = SCA[dict(time=i)]['SCA'].values
        snow_curr = SCA[dict(time=i+1)]['SCA'].values
        
        # Melt for the current day
        melt_curr = melt[dict(time=i)].values.copy()

        # --- Assign snow state transitions ---
        mask_snow = np.logical_or(snow_curr == 100, snow_curr == 205)
        changes[i+1,:,:][mask_snow] = status[dict(time=i)].values[mask_snow]

        # Start of new snow period
        mask_snow_start = np.logical_and(snow_curr == 100, snow_prev == 0)
        changes[i+1,:,:][mask_snow_start] = 2
        
        # End of snow period (melt-out)
        mask_snow_end = np.logical_and(snow_curr == 0, snow_prev == 100)
        changes[i+1,:,:][mask_snow_end] = -2

        # --- Compute total accumulation (or total melt) ---
        melt_curr[changes[i+1,:,:] > 0] = 0  # skip accumulation pixels

        tot_acc[i+1,:,:] = tot_acc[i,:,:] + melt_curr
        tot_acc[i+1,:,:][changes[i+1,:,:] == 0] = 0 # reset where snow-free
        tot_acc[i+1,:,:][mask_snow_start] = melt_curr[mask_snow_start]


        # --- Compute cumulative accumulation fraction (precipitation delta) ---
        delta_sca = delta[dict(time=i+1)].values.copy()
        delta_sca[status[dict(time=i+1)].values != 1] = 0

        sca_sum[i+1,:,:] = sca_sum[i,:,:] + delta_sca
        sca_sum[i+1,:,:][changes[i+1,:,:] == 0] = 0 # reset where snow-free
        sca_sum[i+1,:,:][mask_snow_start] = delta_sca[mask_snow_start]

    # --- Final masking: keep only the values when melt-out ---
    sca_sum[changes != -2] = 0
    tot_acc[changes != -2] = 0

    # --- Convert to xarray and interpolate missing periods ---
    sca_sum_xr = xr.DataArray(sca_sum, dims=('time','x','y'),
                              coords={'time': SCA.time})
    tot_acc_xr = xr.DataArray(tot_acc, dims=('time','x','y'),
                              coords={'time': SCA.time})

    # Fill missing (zero) values backward in time within snow events
    sca_sum_xr = sca_sum_xr.where(sca_sum_xr != 0).bfill(dim='time')
    tot_acc_xr = tot_acc_xr.where(tot_acc_xr != 0).bfill(dim='time')

    # Convert to NumPy arrays for downstream SWE reconstruction
    sca_sum_xr = np.array(sca_sum_xr)
    tot_acc_xr = np.array(tot_acc_xr)

    return sca_sum_xr, tot_acc_xr




def get_swe(SCA, melt, status, delta, sca_sum_xr, tot_acc_xr):
    """
    Compute Snow Water Equivalent (SWE) time series using:
    - status_xr: accumulation/melting mask (+1/-1)
    - delta: fractional precipitation contributions
    - sca_sum_xr: fractional snow accumulation
    - tot_acc_xr: total accumulation energy
    - a: degree-day melt factor

    Parameters
    ----------
    SCA : xr.DataArray
        Snow classification (SCA) over time with dims ('time','x','y')
        Values: 0 = no snow, 100 = snow, 205 = cloud/no data
    ta : xr.Dataset or DataArray
        Air temperature dataset containing 't2m' in °C
    status : xr.DataArray
        Accumulation status mask (+1 accumulation, -1 melting)
    delta : xr.DataArray
        Fractional precipitation available for SWE accumulation
    sca_sum_xr : np.ndarray
        Fractional snow accumulation contributions
    tot_acc_xr : np.ndarray
        Thermal energy available for melting
    verbose : bool
        Print progress

    Returns
    -------
    swe : np.ndarray
        Snow Water Equivalent array (time, x, y)
    """

    dim =  tuple(SCA.sizes.values())  # (time,x,y)
    swe = np.zeros(dim, dtype=np.float32)

    for i, date in enumerate(SCA.time[1:]):
        print(f"{i}: {pd.Timestamp(date.values).strftime('%Y-%m-%d')}")

        melt_curr = melt.isel(time=i).values.copy()
        snow_curr = SCA.isel(time=i+1)['SCA'].values

        # Masks
        mask_acc = status.isel(time=i+1).values == 1
        mask_melt = status.isel(time=i+1).values == -1

        # Spatial increment for accumulation
        dsca = delta.isel(time=i+1).values.copy() / sca_sum_xr[i+1]
        dsca[mask_melt] = 0  # zero where not accumulating

        # Update SWE
        swe[i+1][mask_acc] = swe[i][mask_acc] + dsca[mask_acc] * tot_acc_xr[i+1][mask_acc]
        swe[i+1][mask_melt] = swe[i][mask_melt] - melt_curr[mask_melt]

        # Invalid snow codes (cloud / no data)
        swe[i+1][snow_curr > 100] = np.nan
        swe[i+1][snow_curr == 0] = 0
        # swe[i+1][swe[i+1]<0] = 0

    # # Remove negative SWE values (true melt-out)
    swe[swe < 0] = 0

    return swe



def get_melt_prognostic(SCA, ta, pr_reprojected, SW, status, TF=1.2, SRF=0.2256):
    """
    Compute snowmelt over time and space using ETI with a prognostic albedo 
    decay function.


    Parameters
    ----------
    SCA : xarray.Dataset or xarray.DataArray
        Snow-covered area dataset, must contain variable 'SCA' with dimensions (time, y, x).
    ta : xarray.Dataset or xarray.DataArray
        Air temperature dataset, must contain variable 't2m' (°C) with same dimensions as SCA.
    pr_reprojected : xarray.DataArray
        Precipitation (mm/day) with a 'time' dimension matching SCA.
    SW : xarray.Dataset or xarray.DataArray
        Incoming shortwave radiation dataset, must contain variable 'SW' (W/m² or MJ/m²/day).
    status : xarray.DataArray
        Binary mask (0/1) defining active snow/melt areas per timestep.
    TF : float, optional
        Temperature factor for degree-day melt component (default = 1.2).
    SRF : float, optional
        Shortwave radiation factor for radiative melt component (default = 0.0094).

    Returns
    -------
    melt_da : xarray.DataArray
        Melt (same units as TF/SRF * input fields), dimensions (time, y, x),
        with the same coordinates as the input `SCA`.

    """

    # --- Parameters ---
    asmn = 0.5  # min albedo
    asmx = 0.8 # max albedo
    tau_cold = 1000 # hours
    tau_melt = 100  # hours
    Salb = 10  # mm

    # --- Initialize output arrays ---
    dim = tuple(SCA.sizes.values())  # (time, y, x)
    albs = np.zeros(dim, dtype=np.float32)
    melt = np.zeros(dim, dtype=np.float32)

    time = SCA.time

    # --- Time loop ---
    for i in range(len(time) - 1):
        date = pd.Timestamp(time[i + 1].values).strftime("%Y-%m-%d")
        print(f"Processing {i}: {date}")

        # Previous albedo
        alb_prev = albs[i, :, :].copy()

        # Current timestep variables
        sca_curr = SCA.SCA[i + 1, :, :]
        status_curr = status[i + 1, :, :]
        ta_prev = ta.t2m[i, :, :]
        ta_curr = ta.t2m[i + 1, :, :]
        SW_prev = SW.SW[i, :, :]
        SW_curr = SW.SW[i + 1, :, :]

        # Precipitation
        pr_curr = pr_reprojected.sel(time=date).where(status_curr == 1, 0).values

        # Compute only where snow cover > 0
        mask = sca_curr > 0

        # --- Step 1: Compute melt using previous albedo ---
        melt_prev = TF * ta_prev.where(ta_prev > 0, 0) + SRF * SW_prev * (1 - alb_prev)

        # --- Step 2: Dynamic decay time ---
        tdec = np.where(melt_prev > 0, tau_melt, tau_cold)

        # --- Step 3: Albedo equilibrium ---
        alim = (asmn / tdec + asmx * pr_curr / Salb) / (1 / tdec + pr_curr / Salb)

        # --- Step 4: Albedo evolution ---
        alb_curr = alb_prev.copy()
        alb_curr[mask] = alim[mask] + (alb_prev[mask] - alim[mask]) * np.exp(
            -(1 / tdec[mask] + pr_curr[mask] / Salb) * 24
        )

        # --- Step 5: Clip albedo and save ---
        alb_curr = np.clip(alb_curr, asmn, asmx)
        albs[i + 1, :, :] = alb_curr

        # --- Step 6: Compute current melt with updated albedo ---
        melt[i + 1, :, :] = (
            TF * ta_curr.where(ta_curr > 0, 0) + SRF * SW_curr * (1 - alb_curr)
        )

    # --- Convert melt array to xarray.DataArray ---
    melt_da = xr.DataArray(
        melt,
        dims=("time", "y", "x"),
        coords={"time": SCA.time, "y": SCA.y, "x": SCA.x},
        name="melt",
        attrs={
            "units": "mm water equivalent/day",
            "description": "Snowmelt computed dynamically using evolving albedo."
        },
    )
    
    del melt

    return melt_da



def get_melt_pomeroy(SCA, ta, pr_reprojected, SW, status, TF=1.2, SRF=0.2256):

    # --- Parameters ---
    d_wet = 0.005*24 
    d_dry = 0.0003*24 
    asmn = 0.6  # min albedo
    asmx = 0.9 # max albedo

    Salb = 10  # 10 mm/h

    # --- Initialize output arrays ---
    dim = tuple(SCA.sizes.values())  # (time, y, x)
    albs = np.zeros(dim, dtype=np.float32) + 0.9
    melt = np.zeros(dim, dtype=np.float32)

    time = SCA.time

    # --- Time loop ---
    for i in range(len(time) - 1):
        date = pd.Timestamp(time[i + 1].values).strftime("%Y-%m-%d")
        print(f"Processing {i}: {date}")

        # Previous albedo
        alb_prev = albs[i, :, :].copy()

        # Current timestep variables
        sca_curr = SCA.SCA[i + 1, :, :]
        status_curr = status[i + 1, :, :]
        ta_prev = ta.t2m[i, :, :]
        ta_curr = ta.t2m[i + 1, :, :]
        SW_prev = SW.SW[i, :, :]
        SW_curr = SW.SW[i + 1, :, :]

        # Precipitation
        pr_curr = pr_reprojected.sel(time=date).where(status_curr == 1, 0).values

        # Compute only where snow cover > 0
        mask = sca_curr > 0
        
        alb_dry = alb_prev - d_dry
        alb_wet = ((alb_prev - asmn) * np.exp(-d_wet)) + asmn
        
        alb_t = alb_dry.copy()
        alb_t = np.where(ta_curr < 0, alb_dry, alb_wet)
        
        alb_curr = alb_prev.copy()
        alb_curr[mask] = alb_t[mask] + (asmx - alb_t[mask]) * (pr_curr[mask] / Salb)
        
        # --- Clip albedo and save ---
        alb_curr = np.clip(alb_curr, asmn, asmx)
        albs[i + 1, :, :] = alb_curr
        

        # --- Step 6: Compute current melt with updated albedo ---
        melt[i + 1, :, :] = (
            TF * ta_curr.where(ta_curr > 0, 0) + SRF * SW_curr * (1 - alb_curr)
        )

    # --- Convert melt array to xarray.DataArray ---
    melt_da = xr.DataArray(
        melt,
        dims=("time", "y", "x"),
        coords={"time": SCA.time, "y": SCA.y, "x": SCA.x},
        name="melt",
        attrs={
            "units": "mm water equivalent/day",
            "description": "Snowmelt computed dynamically using evolving albedo."
        },
    )
    
    del melt

    return melt_da
    
    
    
    


# Choose pixel
# iy, ix = 1805, 325

# # Extract time series
# time_vals = SCA.time.values
# alb_ts   = albs[:, iy, ix]
# temp_ts  = ta.t2m[:, iy, ix]
# sca_ts   = SCA.SCA[:, iy, ix]
# pr_ts    = pr_reprojected[:, iy, ix].values
# melt_ts  = melt_da[:, iy, ix].values      # <<–– your melt array

# # ----------------------------------------------------------
# # Create figure with 3 subplots
# # ----------------------------------------------------------
# fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# # ==========================================================
# # 1) ALBEDO + SCA (same subplot)
# # ==========================================================
# ax = axes[0]

# # Albedo line
# ax.plot(time_vals, alb_ts, color='tab:blue', lw=2, label='Albedo')
# ax.set_ylabel("Albedo [-]")
# ax.set_ylim(0, 1)
# ax.grid(alpha=0.3)

# # SCA shaded
# ax.fill_between(time_vals, sca_ts, color='tab:green', alpha=0.2, label='SCA')

# ax.legend(loc='upper right')
# ax.set_title(f"Pixel ({ix}, {iy}) — Albedo, SCA, Temperature, Precipitation, Melt")

# # ==========================================================
# # 2) TEMPERATURE + PRECIPITATION (twin axis)
# # ==========================================================
# ax1 = axes[1]

# # Temperature line
# ax1.plot(time_vals, temp_ts, color='tab:red', lw=2, label='Temperature')
# ax1.axhline(0, color='black', lw=1, ls='--')
# ax1.set_ylabel("Temperature [°C]")
# ax1.grid(alpha=0.3)

# # Twin axis for precipitation
# ax2 = ax1.twinx()
# ax2.plot(time_vals, pr_ts, color='tab:cyan', alpha=0.35, label='Precipitation', lw=2)
# ax2.set_ylabel("Precipitation [mm/day]")

# # Legends merged
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# # ==========================================================
# # 3) MELT
# # ==========================================================
# ax = axes[2]
# ax.plot(time_vals, melt_ts, color='tab:purple', lw=2)
# ax.set_ylabel("Melt [mm/day]")
# ax.set_xlabel("Time")
# ax.grid(alpha=0.3)

# # Finish layout
# fig.tight_layout()
# plt.show()

