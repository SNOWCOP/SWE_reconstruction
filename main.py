#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:03:35 2025

@author: vpremier
"""

import glob
import geopandas as gpd
import pickle
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import gc
import time

from loading import *
from melting import *
from utils import *

config_path = r'/home/vpremier/Documents/git/SNOWCOP/06_swe_reconstruction/config.json'


def run_swe_reconstruction(config):
    
    #%%------------------------ Setup: load the config------------------------
    
    
    #config = load_config(config_path)
    
    # catchment name
    catchment = config['catchment']
    
    emisphere = config['emisphere']
    
    # check date settings
    if "hy_xxxx" in config:
        hy_xxxx = config["hy_xxxx"]
        
        if emisphere == 'South':
            date_start = f"20{hy_xxxx[2:4]}-04-01"
            date_end = f"20{hy_xxxx[4:6]}-03-31"
        elif emisphere == 'North':
            date_start = f"20{hy_xxxx[2:4]}-10-01"
            date_end = f"20{hy_xxxx[4:6]}-09-30"

    elif "date_start" in config and "date_end" in config:
        hy_xxxx = None  # not used in this case
        date_start = config["date_start"]
        date_end = config["date_end"]

    else:
        raise ValueError(
            "Config must specify either the hydrological season 'hy_xxxx' and the emisphere OR both 'date_start' and 'date_end'."
        )
        

    dirname = config['dirname']
    meteodir = config['meteodir']
    temp_dir = os.path.join(meteodir, 'outputs', 'Temperature_biascorr')
    SW_dir = os.path.join(meteodir, 'outputs', 'SW')

    era5_dir = os.path.join(meteodir, 'inputs', 'climate')
    
 
    

    outdir = config['outdir']
    os.makedirs(outdir, exist_ok=True)
    
    
    dem_path = config['DEM_path']

    sca_path = glob.glob(dirname + os.sep + catchment + '*' + hy_xxxx + '*.nc')[0]
    
        
    outname = outdir + os.sep + os.path.basename(sca_path).replace('harm','swe')
    
    if os.path.exists(outname):
        print('File %s has been already created' %outname)
    
    else:
        
        # ---- loading the snow cover area and related information
        sca_old_path = os.path.dirname(sca_path).replace('DAILY_HARM','DAILY') 
        csv_path = os.path.join(sca_old_path, f"{catchment}_{hy_xxxx}_SCF_interp.csv")
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')
        
        SCA, epsg_code = upload_sca(sca_path, dem_path, None)
        SCA = SCA.rio.write_crs(epsg_code['projection'], inplace=True)  
        SCA = SCA.load()
        
        
        # load the DEM        
        DEM, mask_shape_cut, info = get_mask_info(dem_path, None, resType='HR', 
                                                  pixel_ratio=10)

        
        # --- meteorological data ---
        ta_kelvin = load_micromet(temp_dir, hy_xxxx)
        ta = ta_kelvin - 273.15
        # ta = ta.chunk({'time':1, 'x':512, 'y':512})
        # ta = ta.load()
        SW = load_micromet(SW_dir, hy_xxxx)
        # SW = SW.load()
        # SW = SW.chunk({'time':1, 'x':512, 'y':512})
        
        era5 = load_era5land(era5_dir, hy_xxxx)
        
        
        temperature_threshold = config.get('temperature_threshold', 275.15)
        precipitation_threshold = config.get('precipitation_threshold', 5)
        sca_increase_threshold = config.get('sca_increase_threshold', 10)

        # Use the same basin-wide SCA series as the harmonization workflow
        # when it is available; otherwise derive it from the SCA cube.
        sca_percent = None
        if 'SCA final' in df.columns:
            df.index = pd.to_datetime(df.index)
            sca_percent = df['SCA final']
        # era5 = era5.load()
        
        # extract the status and delta
        # status: 1 = accumulation, -1 = melting, 0 = neutral
        # delta: fraction of precipitation contributing to SWE accumulation (per timestep)
        status, delta, pr_reprojected = get_status_and_delta(
            SCA,
            ta_kelvin,
            era5,
            sca_percent=sca_percent,
            temperature_threshold=temperature_threshold,
            precipitation_threshold=precipitation_threshold,
            sca_increase_threshold=sca_increase_threshold,
        )
       
       
        # compute the potential melt 
        # with a fixed albedo
        # melt = TF*ta.where(ta>0, 0).t2m + SRF*SW.SW*(1-asmx)
        
        # with albedo varying with a prognostic function
        TF = 0.24 # melt factor mm / (°C day)
        SRF = 0.15 # melt factor mm / (°C day)
        # melt = get_melt(SCA, ta, pr_reprojected, SW, status, TF = TF, SRF = SRF)
        melt = get_melt_pomeroy(
            SCA, ta, pr_reprojected, SW, status, TF=TF, SRF=SRF,
            T_thresh=temperature_threshold - 273.15,
        )

        
        sca_sum_xr, tot_acc_xr = compute_state_and_accumulation(SCA, melt, status, delta)
     
        swe = get_swe(SCA, melt, status, delta, sca_sum_xr, tot_acc_xr)
        
        swe_int = np.rint(swe).astype("int16")  # or int32 if needed
        
        swe = None
        status = None
        delta = None
        melt = None 
        sca_sum_xr = None
        SCA = None 
        ta = None 
        SW = None
        
        save_nc(outname, swe_int, info, df, "SWE", "mm", dtype = 'float32', complevel = 5)
        
        swe_int = None
        gc.collect()
        




if __name__ == "__main__":
    
    import json 
    
    config_path = r'/home/vpremier/Documents/git/SNOWCOP/SWE_reconstruction/config.json'
    # run_swe_reconstruction(config_path)
    
    # load base config once
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    for year in ["2021"]: #, "2122", "2223","1314", "1415", "1516", "1617", "1718", "1819", "1920"]:  # adjust range as needed
        config = base_config.copy()  # avoid overwriting original
        
        config["hy_xxxx"] = f"hy{year}"
        
        # run_swe_reconstruction(config)
