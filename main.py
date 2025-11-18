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
from utils import *
from s1_wetness import get_moistening
import xarray as xr
import time

from loading import *
from melting import *

config_path = r'/home/vpremier/Documents/git/SNOWCOP/06_swe_reconstruction/config.json'


def run_swe_reconstruction(config_path):
    
    #%%------------------------ Setup: load the config------------------------
    
    
    config = load_config(config_path)
    
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
    temp_dir = os.path.join(meteodir, 'outputs', 'T_zarr')
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
        sca_old_path = os.path.dirname(sca_path).replace('sca_harm','daily_sca') 
        csv_path = os.path.join(sca_old_path, f"{catchment}_{hy_xxxx}.csv")
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')
        
        SCA, epsg_code = upload_sca(sca_path, dem_path, None)
        SCA = SCA.rio.write_crs(epsg_code['projection'], inplace=True)  
        SCA = SCA.load()
        
        
        # load the DEM        
        DEM, mask_shape_cut, info = get_mask_info(dem_path, None, resType='HR', 
                                                  pixel_ratio=10)

        
        # --- meteorological data ---
        ta = load_micromet(temp_dir, hy_xxxx) - 273.15
        # ta = ta.chunk({'time':1, 'x':512, 'y':512})
        # ta = ta.load()
        SW = load_micromet(SW_dir, hy_xxxx)
        # SW = SW.load()
        # SW = SW.chunk({'time':1, 'x':512, 'y':512})
        
        era5 = load_era5land(era5_dir, hy_xxxx)
        # era5 = era5.load()
        
        # extract the status and delta
        # status : boolean accumulation mask: True = accumulation, False = melting/other
        # delta: fraction of precipitation contributing to SWE accumulation (per timestep)
        status, delta, pr_reprojected = get_status_and_delta(SCA, ta, era5)
       
       
        # compute the potential melt 
        # with a fixed albedo
        # melt = TF*ta.where(ta>0, 0).t2m + SRF*SW.SW*(1-asmx)
        
        # with albedo varying with a prognostic function
        TF = 1.2 # melt factor mm / (°C day)
        SRF = 0.2256 # melt factor mm / (°C day)
        # melt = get_melt(SCA, ta, pr_reprojected, SW, status, TF = TF, SRF = SRF)
        melt = get_melt_pomeroy(SCA, ta, pr_reprojected, SW, status, TF = TF, SRF = SRF)

        
        sca_sum_xr, tot_acc_xr = compute_state_and_accumulation(SCA, melt, status, delta)
     
        swe = get_swe(SCA, melt, status, delta, sca_sum_xr, tot_acc_xr)

        # save_nc(outname, swe, info, df, "SWE", "mm", scale=1, dtype = 'float32',
        #         complevel=9)
        





