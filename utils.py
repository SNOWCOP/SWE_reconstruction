# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:59:41 2019

@author: WPremier
"""
from osgeo import gdal, ogr, osr
import glob
import numpy as np
import re
from datetime import datetime, timedelta
import os
from decimal import Decimal
import xarray as xr
import pickle
import pandas as pd
import netCDF4
from affine import Affine
from rasterio.features import geometry_mask
import rasterio.features
import json

from tqdm import tqdm  # Progress bar (optional)
#from kriging import get_kriging


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_window(mask, ncol, nrow, pixel_ratio):
    
    rows = np.shape(mask)[0]
    cols = np.shape(mask)[1]
    
    rows_lr = int(rows/pixel_ratio)
    cols_lr = int(cols/pixel_ratio)
    
    
    row_slice = int(np.ceil(rows_lr/nrow))
    col_slice = int(np.ceil(cols_lr/ncol))
    
    lr_mask = np.ones(shape=(rows_lr, cols_lr)) * 0
    hr_mask = np.ones(shape=(rows, cols)) * 0

    nwindow = 1
    for j in range(0,rows_lr,row_slice):  
        for i in range(0, cols_lr, col_slice):
 
            lr_mask[j:j + row_slice,i: i + col_slice]+=nwindow
            hr_mask[j*pixel_ratio:j*pixel_ratio + pixel_ratio*row_slice,
                    i*pixel_ratio: i*pixel_ratio + pixel_ratio*col_slice]+=nwindow
            nwindow+=1

    return hr_mask, nwindow



def get_mask_indices(mask, info, shape, resType='HR', pixel_ratio=20):
    
    if shape:
        # Create the affine transformation from the geotransform
        affine_transform = Affine(info['geotransform'][1],
                                  info['geotransform'][2],
                                  info['geotransform'][0],
                                  info['geotransform'][4],
                                  info['geotransform'][5],
                                  info['geotransform'][3])
        dims = info['X_Y_raster_size'][1],info['X_Y_raster_size'][0]
        mask = geometry_mask(
                        shape, 
                        transform=affine_transform, 
                        invert=True,  # True means inside the shape is True
                        out_shape=dims
                    )
        
        # convert to LR-> at least 1 valid to be valid 
        mask_LR = get_LR_mask(mask, pixel_ratio=pixel_ratio, nv_thres=1)
        
        # get back to HR
        mask_HR = np.repeat(mask_LR, pixel_ratio, axis=0)
        mask_HR = np.repeat(mask_HR, pixel_ratio, axis=1)
        
        if resType=='HR':
            mask_shape = mask_HR
        else:
            mask_shape = mask_LR
            
        j0 = int(np.where(mask_shape.any(axis=1))[0][0])
        jend = int(np.where(mask_shape.any(axis=1))[0][-1])
        i0 = int(np.where(mask_shape.any(axis=0))[0][0])
        iend = int(np.where(mask_shape.any(axis=0))[0][-1])
    else:
        j0 = 0
        jend = np.shape(mask)[0]-1
        i0 = 0
        iend = np.shape(mask)[1]-1
        
        mask_shape = np.ones(mask.shape, dtype=int)

    indices = (i0, iend, j0, jend)
    return indices, mask_shape



def load_ts(path):
    """
    Get the extension of the data   
    """
    
    ext = os.path.basename(path).split('.')[-1]
    if ext == 'nc':
        data = xr.open_dataset(path)
        data = data.load()
    elif ext == 'pickle':
        data = pickle.load(open(path, 'rb'))
 
    return data



def metrics(est, ref):
    """
    Get bias, rmse etc between two arrays
    """
    bias = (est-ref).sum()/est.size
    rmse =  np.sqrt(((est - ref) ** 2).sum() / est.size)
    unb_rmse = np.sqrt((((est - est.mean()) - (ref - ref.mean())) ** 2).sum() /
                   est.size)

    num = ((est - est.mean()) * (ref - ref.mean())).sum()
    den = ((est - est.mean()) ** 2).sum() * ((ref - ref.mean()) ** 2).sum()
    cross_corr = num / np.sqrt(den)
    
    return (rmse, unb_rmse, bias, cross_corr)



def PointInRaster(information,LonLat):
    

    xMin = information['extent'][0]
    xMax = information['extent'][2]
    yMin = information['extent'][1]
    yMax = information['extent'][3]
    
    
    if xMin <= LonLat[0] <= xMax:
        if yMin <= LonLat[1] <= yMax:            
            # Find the pixel coordinates
            
            # check if the point is exactely at the border of the pixel and add delta/2 in this case
            # to be sure that the considered pixel is always the one on the bottom left corner
            if Decimal(str(LonLat[0] - xMin)) % Decimal(str(information['geotransform'][1])) == 0:
                posx = int((LonLat[0] + information['geotransform'][1]/2 - xMin)/information['geotransform'][1])
            else:
                posx = int((LonLat[0] - xMin)/information['geotransform'][1])
                
            if Decimal(str(LonLat[1] - yMax)) % Decimal(str(information['geotransform'][5])) == 0:
                posy = int((LonLat[1] + information['geotransform'][5]/2 - yMax)/information['geotransform'][5])     
            else:
                posy = int((LonLat[1] - yMax)/information['geotransform'][5])
    else:
        posx = np.nan
        posy = np.nan
            
    return posx,posy




def open_image(image_path,ncdf_layer='fsc'):
    """Opens an image and reads its metadata.
    
    Parameters
    ----------
    image_path : str
        path to an image
    
    Returns
    -------
    image : osgeo.gdal.Dataset
        the opened image
    information : dict
        dictionary containing image metadata    
    """
    
    ext = os.path.basename(image_path).split('.')[-1]
    
    if ext == 'nc':
        nc_data = netCDF4.Dataset(image_path,'r')
        vars_nc = list(nc_data.variables)
        scf_name = list(filter(lambda x: x.startswith(ncdf_layer), vars_nc))[0]        
        dataset = gdal.Open("NETCDF:{0}:{1}".format(image_path, scf_name))
        proj = dataset.GetProjection()        
        geotransform = dataset.GetGeoTransform()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + geotransform[1] * cols
        miny = maxy + geotransform[5] * rows        
        extent = [minx, miny, maxx, maxy]        
        X_Y_raster_size = [cols, rows]
        information = {}
        information['geotransform'] = geotransform
        information['extent'] = extent
        information['geotransform'] = tuple(map(lambda x: round(x, 4) or x, information['geotransform']))
        information['extent'] = tuple(map(lambda x: round(x, 4) or x, information['extent'])) 
        information['X_Y_raster_size'] = X_Y_raster_size
        information['projection'] = proj
        
        image_output = np.array(dataset.ReadAsArray(0, 0,cols, rows))            

    else:
        image = gdal.Open(image_path)
        cols = image.RasterXSize
        rows = image.RasterYSize
        geotransform = image.GetGeoTransform()
        proj = image.GetProjection()
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + geotransform[1] * cols
        miny = maxy + geotransform[5] * rows
        X_Y_raster_size = [cols, rows]
        extent = [minx, miny, maxx, maxy]
        information = {}
        information['geotransform'] = geotransform
        information['extent'] = extent
        information['X_Y_raster_size'] = X_Y_raster_size
        information['projection'] = proj
        image_output = np.array(image.ReadAsArray(0, 0,cols, rows))
        
    if image is None:
        print('could not open ' + image_path)
        return
        
    return image_output, information



def save_tif(outdir, fname, info, array):

    fileName_output = outdir + os.sep + fname + '.tif'
    #Create the map
    out = gdal.GetDriverByName('GTiff').Create(fileName_output,info['X_Y_raster_size'][0],
                              info['X_Y_raster_size'][1],1,gdal.GDT_Float32)     
    outband = out.GetRasterBand(1)
    
    # Set the geographic information
    out.SetGeoTransform(info['geotransform'])
    out.SetProjection(info['projection'])
    outband.WriteArray(array)
        
    outband.FlushCache()
    out = None
    
    
    
def reproj_point(x, y, from_epsg, to_epsg):
    """Reproject a point into a defined coordinate system.
    
    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    srIn : osgeo.osr.SpatialReference
        spatial reference of the input coordinate system
    srOut : osgeo.osr.SpatialReference
        spatial reference of the output coordinate system
        
    Returns
    -------
    (x, y) : tuple
        the transformed coordinates
    """
    
    
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    
    srIn = osr.SpatialReference()
    srIn.ImportFromEPSG(from_epsg)
    
    srOut = osr.SpatialReference()
    srOut.ImportFromEPSG(to_epsg)
    
    # coordinate transformation
    coordTransform = osr.CoordinateTransformation(srIn, srOut)
    
    # reproject point
    point.Transform(coordTransform)
    
    # reprojected point in the new coordinate system
    (x, y) = point.GetX(), point.GetY()
    
    return (x, y)




# read shape file and get the extent in a selected reference system and rounded 
# to a selected resolution

def get_Shape_extent(shape_name, epsg=3035, outres =500):
    shp = ogr.Open(shape_name)
    lyr = shp.GetLayer()
    crs_shp = lyr.GetSpatialRef() 
    
    # Extract the polygon coordinates
    for f in range(lyr.GetFeatureCount()):
        feat = lyr.GetFeature(f)
        geom = feat.GetGeometryRef()
        if f==0:
            points_shp = np.array(geom.GetGeometryRef(0).GetPoints())
        else:
            points_shp = np.vstack((points_shp, 
                                    np.array(geom.GetGeometryRef(0).GetPoints())))
            
    # Compute the coordinates for getting the maximum extent. 
    # Rounded with respect to the outres
    xmin = min(points_shp[:, 0])
    ymin = min(points_shp[:, 1])
    xmax = max(points_shp[:, 0])
    ymax = max(points_shp[:, 1])
    
    # check if the reference system is the same
    srOut = osr.SpatialReference()
    srOut.ImportFromEPSG(epsg)
    if crs_shp != srOut:
        print('The input shapefile is in another reference system. Reprojecting...')
        xmin, ymin = reproj_point(xmin, ymin, crs_shp, srOut)
        xmax, ymax = reproj_point(xmax, ymax, crs_shp, srOut)
    
    # round with respect to the outres
    xMin = round(int(xmin / outres) * outres, 5)
    yMin = round(int(ymin / outres) * outres, 5)
    xMax = round(np.ceil(xmax / outres) * outres, 5)
    yMax = round(np.ceil(ymax / outres) * outres, 5)

    return xMin, yMin, xMax, yMax


def get_pixelcoord(rasterName, shapeExtent):
    
    """Get pixel coordinates corresponding to an extent
    
    Parameters
    ----------
    rasterName : string
                 name of the raster file
    shapeExtent : tuple
            extent (xMin, yMin, xMax, yMax)
        
    Returns
    -------
    (i, j, cols, rows) : tuple
        i, j indices of the upper left pixel,
        cols, rows number of pixels
    """
    
    img, info = open_image(rasterName) 
    
    xMin, yMin, xMax, yMax = shapeExtent
    
    
    xMin_r, yMin_r, xMax_r, yMax_r = info['extent']
    i = int((xMin - xMin_r) / info['geotransform'][1])
    j = int((yMax - yMax_r) / info['geotransform'][5])
    cols = int((xMax - xMin) / info['geotransform'][1])
    rows = int((yMax - yMin) / info['geotransform'][1])
    
    return (i, j, cols, rows)


def dateFromFileName(string):
    match = re.search(r'\d{8}T\d{6}', string)
    date = datetime.strptime(match.group(), '%Y%m%dT%H%M%S')
    return(date)
    
    
def get_s2_date(scene):
    """Extracts the date from a valid Sentinel 2 scene identifier.
    
    Parameters
    ----------
    scene : str
        a valid Sentinel 2 scene identifier
    
    Returns
    -------
    date : datetime.date
        the date of the Sentinel 2 scene
    """
    date = scene.split('_')[2].split('T')[0]
    return datetime.datetime.strptime(date, '%Y%m%d').date()


def get_scene_extent(i1, i2):
    """Calculates the extent of one image in another one, eventually with
    different resolutions. This function is used to retrieve the extent of the
    high resolution Landsat or Sentinel 2 scenes in the lower resolution
    reference products AVHRR, MODIS, etc.
    
    Parameters
    ----------
    i1 : str
        path to a .tif image
    i2 : str
        path to a .tif image

    Returns
    -------
    x_tl : float
        the x-coordinate of the top left corner point of i1 in i2
    y_br : float
        the y-coordinate of the bottom right corner point of i1 in i2
    x_br : float
        the x-coordinate of the bottom right corner point of i1 in i2
    y_tl : float
        the y-coordinate of the top left corner point of i1 in i2
    """
    
    # read the two images
    i1_ds, i1_info = open_image(i1)
    i2_ds, i2_info = open_image(i2)
    
    # difference in i2 grid points of the x-coordinate of the top
    # left corner between i1 and i2
    diff_x = ((i1_info['geotransform'][0] - i2_info['geotransform'][0]) /
            i2_info['geotransform'][1])
    
    # x-coordinate of the top left corner of i1 relative to i2
    x_tl = int(round(diff_x,5))
    
    # number of i1 grid points i1 is shifted in x-direction with respect to the
    # nearest i2 grid point
    if round((diff_x % 1), 5) == 1:
        x_shift = 0
    else:
        x_shift = int(round((round((diff_x % 1), 5) * i2_info['geotransform'][1]) /
                   i1_info['geotransform'][1],5))
    
    # difference in i2 grid points of the y-coordinate of the top
    # left corner between i1 and i2
    diff_y = ((i1_info['geotransform'][3] - i2_info['geotransform'][3]) /
            i2_info['geotransform'][5])
    
    # y-coordinate of the top left corner of i1 relative to i2
    y_tl = int(round(diff_y,5))
    
    # number of i1 grid points i1 is shifted in y-direction with respect to the
    # nearest i2 grid point
    if round((diff_y % 1), 5) == 1:
        y_shift = 0
    else:
        y_shift = int(round((round((diff_y % 1), 5) * i2_info['geotransform'][5]) /
                   i1_info['geotransform'][5],5))

    # get the indices of the bottom right corner of i1 relative to i2
    x_br = int(round(x_tl + ((x_shift + i1_info['X_Y_raster_size'][0]) *
                               i1_info['geotransform'][1]) /
               i2_info['geotransform'][1],5))
    y_br = int(round(y_tl + ((y_shift + i1_info['X_Y_raster_size'][1]) *
                               i1_info['geotransform'][5]) /
               i2_info['geotransform'][5],5))
    # check if the extent of i1 is contained in i2
    if x_tl < 0 or x_br > i2_ds.RasterXSize or y_br > i2_ds.RasterYSize or y_br < 0 or y_tl < 0:
        raise IndexError('{} domain exceeds {} domain. '
                         ''.format(os.path.basename(os.path.normpath(i1)),
                                   os.path.basename(os.path.normpath(i2))))
        return
    
    return x_tl, y_br, x_br, y_tl, x_shift, y_shift





def get_swe_point(df, optical_data, DD_data, j, i):
    # funzione per un singolo pixel
    swe = pd.DataFrame(index = df.index)
    swe['HR_ts'] = optical_data[j,i,:]
    
    # corresponding sca variation
    swe['delta_sca'] = df['sca'].diff()
    
    #potential melting
    swe['potential_melt'] = DD_data[j,i,:]
    
    
    swe['status'] = 0 # problema che i primi giorni ho nan!!!
    # 1 : melting
    swe['status'][swe['delta_sca'] < 0] = 1
    # 2 : accumulation
    swe['status'][swe['delta_sca'] > 0] = 2
    # 0 : snow free
    swe['status'][swe['HR_ts'] == 0] = 0
    
    
    #total accumulated snow
    swe['potential_melt'][swe['status']==1].sum()
    
    mask = (swe['status'].notnull()) & (swe['status']!=0)
    
    fds = swe['status'][mask.shift(periods=-1).fillna(False)]
    fds = fds[fds == 0]
    
    lds = swe['status'][mask.shift(periods=1).fillna(False)]
    lds = lds[lds == 0]
    
    swe['swe'] = 0
    for ix_fds,ix_lds in zip(fds.index,lds.index):
        
        
        swe['status'][ix_lds] = 1
        tot_melt = swe.shift()[ix_fds:ix_lds][swe['status'][ix_fds:ix_lds]==1]['potential_melt'].sum()
    
        acc_sca = swe[ix_fds:ix_lds][swe['status'][ix_fds:ix_lds]==2]['delta_sca']
        delta_acc = acc_sca/acc_sca.sum()
        
        
        swe[ix_fds:ix_lds]
        swe_tmp = np.zeros(len(swe[ix_fds:ix_lds]))
        swe_tmp[swe['status'][ix_fds:ix_lds]==2]= np.array(delta_acc*tot_melt)
        swe_tmp[swe['status'][ix_fds:ix_lds]==1] = -swe.shift()[ix_fds:ix_lds][swe['status'][ix_fds:ix_lds]==1]['potential_melt']
        
        swe['swe'][ix_fds:ix_lds] = swe_tmp
        cumsum = swe['swe'][ix_fds:ix_lds].cumsum()
        swe['swe'][ix_fds:ix_lds] = cumsum.copy()
        
        # se minore di zero, fisso a zero
        swe['swe'][ix_fds:ix_lds][cumsum < 0] = 0
        
    return swe






def class_analysis(array, mask, classes):
    value_class = []
    for iClass in range(0, len(classes)-1):
        array_masked = np.ma.masked_where(
                np.logical_or(np.isnan(mask),
                np.logical_or(mask<classes[iClass], 
                              mask >=classes[iClass+1])), array)
        

        N = np.sum(~array_masked.mask)
        print(N)
        
        if array_masked.mask.all():
            value_class.append(np.nan)
        else:
#            value_class.append(np.nansum(array_masked*10**(-9)*(25*25)))
            value_class.append(np.nansum(array_masked)/N)
    
    mean_class = np.divide(np.add(classes[1:],classes[:-1]),2)

    return mean_class, value_class


def class_analysis_2(estimator, reference, mask, classes,area):
    value_class = {'est' : [], 'ref' : []} 
    metr = {'bias' : [], 'rmse' : [], 'bias_list' : []} 
    for iClass in range(0, len(classes)-1):
        
        mask_total = np.logical_and.reduce((mask>=classes[iClass], 
                              mask <classes[iClass+1],~np.isnan(estimator)))
        est_masked = estimator[mask_total]
        ref_masked = reference[mask_total]
         
        N = np.sum(mask_total)
        
        
        if est_masked == []:
            value_class['est'].append(np.nan)
            value_class['ref'].append(np.nan)
            metr['bias'].append(np.nan)
            metr['rmse'].append(np.nan)
        else:
#            value_class['est'].append(np.nansum(est_masked)*10**(-9)*(area))
#            value_class['ref'].append(np.nansum(ref_masked)*10**(-9)*(area))
            
            value_class['est'].append(np.nansum(est_masked)/N)
            value_class['ref'].append(np.nansum(ref_masked)/N)
            
            metr['bias'].append(metrics(est_masked,ref_masked)[2])
            metr['rmse'].append(metrics(est_masked,ref_masked)[0])
            metr['bias_list'].append(est_masked-ref_masked)

            
    mean_class = np.divide(np.add(classes[1:],classes[:-1]),2)

    return mean_class, value_class, metr


def create_grid(info):
    # get the information about the extent and resolution
    pixel_size = info['geotransform'][1]
    x_min = info['extent'][0] 
    x_max = info['extent'][2]
    y_min = info['extent'][1]
    y_max = info['extent'][3]
    eps = 0.0001

    # get the coordinates of the center of each cell.
    # An epsilon is added to consider always the last cell, that may be excluded 
    # due to floating errors
    xcoords = np.arange(x_min + pixel_size/2, 
                        x_max - pixel_size/2 + eps, pixel_size)
    ycoords = np.arange(y_min + pixel_size/2, 
                        y_max - pixel_size/2 + eps, pixel_size)

    gridx, gridy = np.meshgrid(xcoords, ycoords)
    
    return gridx, gridy 




def get_LR_mask(mask, pixel_ratio=20, nv_thres=40):
    
    # rows and columns of the new mask
    nrows = int(np.shape(mask)[0]/pixel_ratio)
    ncols = int(np.shape(mask)[1]/pixel_ratio)
    
    #initialize array
    mask_LR = np.zeros(shape=(nrows, ncols), dtype=bool)
    
    # iterate over rows
    y_j = 0
    x_i = 0
    for j in range(0, nrows):                                    
        # reset column counter
        x_i = 0             
        # iterate over columns
        for i in range(0, ncols):

            # read the slice of the scene matching the current
            # estimator pixel
            data_ij = mask[y_j:y_j + pixel_ratio,x_i: x_i + pixel_ratio]

            # check how many pixels are not valid
            nv_sum = np.sum(data_ij)

            if nv_sum >= nv_thres:
                # if the number of masked pixels exceed the threshold, set as False
                mask_LR[j, i] = True

            # advance column counter by number of high resolution pixels
            # contained in one low resoution pixels
            x_i += pixel_ratio
        
        # advance row counter by number of high resolution pixels
        # contained in one low resoution pixels
        y_j += pixel_ratio
                
    return mask_LR





def save_nc(outname, array, info, df, varname, unit, scale=1, dtype = 'int32',
            complevel = 9):
   

    time = df.index
    reference_time = df.index[0]
    
        
 
    if type(array) is xr.core.dataset.Dataset:
        da = array
       
    else:
        # in this way, the coordinate is the upper left corner
        nx = np.shape(array)[2]; ny = np.shape(array)[1]
        
        x = info['geotransform'][0] + info['geotransform'][1]/2 + \
            info['geotransform'][1]*np.arange(nx)
        y = info['geotransform'][3] + info['geotransform'][5]/2 + \
            info['geotransform'][5]*np.arange(ny)
        
 
    
        da = xr.DataArray(
            name = varname,
            data=array,
            dims=["time", "y","x"],
            coords=dict(
                x=(["x"], x),
                y=(["y"], y),
                time=time,
                reference_time=reference_time,
            ),
            attrs=dict(
                units=unit,
            ),
        )
    
        da = da.transpose("time", "y", "x")

    if os.path.exists(outname):
        print("The output netcdf already exists")
        da_0 = load_ts(outname)
        da = xr.combine_by_coords([da_0, da])
        os.remove(outname)
        
    #get EPSG
    srs = osr.SpatialReference()
    srs.ImportFromWkt(info['projection'])
    
    # da = da.assign_coords({"crs": info['projection']})



    da.rio.write_crs("epsg:" + srs.GetAttrValue('AUTHORITY',1), 
                     inplace=True).rio.set_spatial_dims(
                         x_dim="x",
                         y_dim="y",inplace=True).rio.write_coordinate_system(inplace=True)
                         
    da.rio.write_coordinate_system("epsg:" + srs.GetAttrValue('AUTHORITY',1))
    
    encode = {
        varname: {
            'zlib': True,
            'complevel': complevel,
            'dtype': dtype
        }
    }
    
    # Only add scale_factor if different from 1
    if scale != 1:
        encode[varname]['scale_factor'] = scale
    
    da.to_netcdf(outname, encoding=encode)




def get_new_info(mask, info):
    
    # Find indices where mask is False (valid region)
    indices = np.where(~mask)
    
    # Ensure there are valid indices before accessing
    if indices[0].size > 0 and indices[1].size > 0:
        imin = indices[1].min()
        imax = indices[1].max()
        
        jmin = indices[0].min()
        jmax = indices[0].max()
    else:
        # Handle the case where all values are masked (e.g., set defaults)
        imin, imax, jmin, jmax = None, None, None, None
    
    
    xmin_new = info['geotransform'][0] + imin*info['geotransform'][1] 
    xmax_new = info['geotransform'][0] + (imax+1)*info['geotransform'][1] 

    ymax_new = info['geotransform'][3] + jmin*info['geotransform'][5] 
    ymin_new = info['geotransform'][3] + (jmax+1)*info['geotransform'][5] 
    
    info_new = info.copy()
    
    info_new['geotransform']= (xmin_new, info['geotransform'][1], 0,
                                   ymax_new, 0, info['geotransform'][5])
    info_new['extent'] = [xmin_new, ymin_new, xmax_new, ymax_new]
    info_new['X_Y_raster_size']= [imax-imin+1, jmax-jmin+1]
    
    return info_new



def upload_sca(sca_path, dem_path, subbasin):
    

        
    stack_HR = xr.open_dataset(sca_path) 
    
    _, mask_shape_cut, info = get_mask_info(dem_path, subbasin, resType='HR', pixel_ratio=10)

    
    mask_2d = xr.DataArray(mask_shape_cut, coords=[stack_HR.y, stack_HR.x], dims=["y", "x"])
    stack_HR = stack_HR.where(mask_2d)
    
    return stack_HR, info




def load_meteo(meteodir, hy, stack_HR, dem_path, subbasin):
    
    date_start, date_end = f"20{hy[2:4]}-04-01", f"20{hy[4:6]}-03-31"
    
    # Extract start and end years from hy
    year_start, year_end = hy[2:4], hy[4:6]
    
    #### meteo forcings
    ta_dir = os.path.join(meteodir, 't2m')
    pr_dir = os.path.join(meteodir, 'pr')

    # Create file patterns for the two years
    ta_pattern = os.path.join(ta_dir, f"CR2MET*.nc")
    ta_pattern_next_year = os.path.join(ta_dir, f"CR2MET*20{year_end}*.nc")
    
    pr_pattern = os.path.join(pr_dir, f"CR2MET*20{year_start}*.nc")
    pr_pattern_next_year = os.path.join(pr_dir, f"CR2MET*20{year_end}*.nc")
    

    ta_files = glob.glob(ta_pattern) + glob.glob(ta_pattern_next_year)
    pr_files = glob.glob(pr_pattern) + glob.glob(pr_pattern_next_year)

    
    # Open multiple NetCDF files for both years
    ta = xr.open_mfdataset(ta_files).sel(time=slice(date_start, date_end))
    pr = xr.open_mfdataset(pr_files).sel(time=slice(date_start, date_end))

    pr = pr.rio.write_crs("EPSG:4326", inplace=True)  
    pr = pr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True).rename({'lon': 'x', 'lat': 'y'})

    ta = ta.rio.write_crs("EPSG:4326", inplace=True) 
    ta = ta.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True).rename({'lon': 'x', 'lat': 'y'})


    stack_HR = stack_HR.rio.write_crs("EPSG:32719", inplace=True)

    # extract info for the study area
    pr_reprojected = pr.rio.reproject_match(stack_HR)
    
    DEM, mask_shape_cut, info = get_mask_info(dem_path, subbasin, resType='HR', pixel_ratio=10)


    # Initialize a list to store DataArrays
    interpolated_list = []
    
    # Extract coordinates from stack_HR
    x_coords = stack_HR['x'].values
    y_coords = stack_HR['y'].values
    
    # Iterate over all dates with a progress bar
    for date in tqdm(ta.time.values, desc="Interpolating temperature"):
        ta_interp = get_kriging(ta, 't2m', date, stack_HR, info, DEM, mask_shape_cut, dem_path)
        # Ensure ta_interp is an xarray.DataArray and add a time coordinate
        
        ta_interp_da = xr.DataArray(
        data=np.ma.filled(ta_interp, np.nan),  # Fill masked values with NaN
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords, "time": date},
        name="t2m_interp"
        )
        
        interpolated_list.append(ta_interp_da)


    # Convert the list of DataArrays into a single xarray.Dataset
    ta_interp_ds = xr.concat(interpolated_list, dim="time").to_dataset(name="t2m_interp")

    del ta, pr, interpolated_list
    
    return ta_interp_ds, pr_reprojected




def xr_rasterize(gdf,
                 da,
                 attribute_col=False,
                 crs=None,
                 transform=None,
                 name=None,
                 x_dim='x',
                 y_dim='y',
                 **rasterio_kwargs):    
    """
    Rasterizes a geopandas.GeoDataFrame into an xarray.DataArray.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A geopandas.GeoDataFrame object containing the vector/shapefile
        data you want to rasterise.
    da : xarray.DataArray
        The shape, coordinates, dimensions, and transform of this object 
        are used to build the rasterized shapefile. It effectively 
        provides a template. The attributes of this object are also 
        appended to the output xarray.DataArray.
    attribute_col : string, optional
        Name of the attribute column in the geodataframe that the pixels 
        in the raster will contain.  If set to False, output will be a 
        boolean array of 1's and 0's.
    crs : str, optional
        CRS metadata to add to the output xarray. e.g. 'epsg:3577'.
        The function will attempt get this info from the input 
        GeoDataFrame first.
    transform : affine.Affine object, optional
        An affine.Affine object (e.g. `from affine import Affine; 
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0) giving the 
        affine transformation used to convert raster coordinates 
        (e.g. [0, 0]) to geographic coordinates. If none is provided, 
        the function will attempt to obtain an affine transformation 
        from the xarray object (e.g. either at `da.transform` or
        `da.geobox.transform`).
    x_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for x coordinates. Defaults to 'x'.    
    y_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for y coordinates. Defaults to 'y'.
    export_tiff: str, optional
        If a filepath is provided (e.g 'output/output.tif'), will export a
        geotiff file. A named array is required for this operation, if one
        is not supplied by the user a default name, 'data', is used
    **rasterio_kwargs : 
        A set of keyword arguments to rasterio.features.rasterize
        Can include: 'all_touched', 'merge_alg', 'dtype'.
    
    Returns
    -------
    xarr : xarray.DataArray
    
    """
    
    # Check for a crs object
    try:
        crs = da.spatial_ref.crs_wkt
    except:
        if crs is None:
            raise Exception("Please add a `crs` attribute to the "
                            "xarray.DataArray, or provide a CRS using the "
                            "function's `crs` parameter (e.g. 'EPSG:3577')")
    

    # Check if transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    if transform is None:
        try:
            # First, try to take transform info from geobox
            transform = da.geobox.transform
        # If no geobox
        except:
            try:
                # Try getting transform from 'transform' attribute
                transform = da.SWE.transform
            except:
                # If neither of those options work, raise an exception telling the 
                # user to provide a transform
                raise Exception("Please provide an Affine transform object using the "
                        "`transform` parameter (e.g. `from affine import "
                        "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                        "6886890.0)`")
    
    # Get the dims, coords, and output shape from da
    da = da.squeeze()
    
    try:
        y, x = da.shape[1:]
    except:
        try:
            y, x = da.SWE.shape[1:]
        except :
            y, x = da.__xarray_dataarray_variable__.shape[1:]
    dims = list(da.dims)
    xy_coords = [da[y_dim], da[x_dim]]   
    
    # Reproject shapefile to match CRS of raster
    print(f'Rasterizing to match xarray.DataArray dimensions ({y}, {x}) '
          f'and projection system/CRS (e.g. {crs})')
    
    try:
        gdf_reproj = gdf.to_crs(crs=crs)
    except:
        #sometimes the crs can be a datacube utils CRS object
        #so convert to string before reprojecting
        gdf_reproj = gdf.to_crs(crs={'init':str(crs)})
    
    # If an attribute column is specified, rasterise using vector 
    # attribute values. Otherwise, rasterise into a boolean array
    if attribute_col:
        
        # Use the geometry and attributes from `gdf` to create an iterable
        shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute_col])

        # Convert polygons into a numpy array using attribute values
        arr = rasterio.features.rasterize(shapes=shapes,
                                          out_shape=(y, x),
                                          transform=transform,
                                          **rasterio_kwargs)
    else:
        # Convert polygons into a boolean numpy array 
        arr = rasterio.features.rasterize(shapes=gdf_reproj.geometry,
                                          out_shape=(y, x),
                                          transform=transform,
                                          **rasterio_kwargs)
        
    
                
    return arr


def get_mask_info(dem_path, subbasin, resType='HR', pixel_ratio=10):    
    # read the dem
    DEM, info = open_image(dem_path)
    # DEM = DEM.ReadAsArray()
    mask = np.logical_or(np.isnan(DEM),(DEM <= 0.001))


    indices, mask_shape =  get_mask_indices(mask, info, subbasin, 
                                            resType=resType, pixel_ratio=pixel_ratio)
    (i0, iend, j0, jend) = indices
    mask_shape_cut = mask_shape[j0:jend+1,i0:iend+1]
    
    info_new = get_new_info(~mask_shape, info)
    DEM_new = DEM[j0:jend+1,i0:iend+1]
    
    return DEM_new, mask_shape_cut, info_new