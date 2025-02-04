
# Load in Dependencies
# python3 -m pip install scikit-learn tqdm

from IPython.display import display

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets
import xarray as xr

# Geospatial raster data handling
import rioxarray as rxr

# Geospatial data analysis
import geopandas as gpd

# Geospatial operations
import rasterio
from rasterio import windows  
from rasterio import features  
from rasterio import warp
from rasterio.warp import transform_bounds 
from rasterio.windows import from_bounds 
from shapely.geometry import Point

# Image Processing
from PIL import Image

# Coordinate transformations
from pyproj import Proj, Transformer, CRS

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Planetary Computer Tools
import pystac_client
import planetary_computer as pc
from pystac.extensions.eo import EOExtension as eo

# Others
import os
from tqdm import tqdm

# -----------------------------------------------------------------------------
# EY Functions
# -----------------------------------------------------------------------------


# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1,dataset2):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.concat([dataset1, dataset2], axis=1)
    return data

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def buffer_meters(row, meters):
    return row['geometry'].buffer(meters)


def get_bbox_radius(df, meters):
    if isinstance(meters, int) or isinstance(meters, float):
        meters = [meters]
    elif not isinstance(meters, list):
        raise ValueError("meters should be a list or a number")
    
    gdf = gpd.GeoDataFrame(
        df[['Longitude', 'Latitude']], geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs='EPSG:4326' # Latitude-Longitude -> https://spatialreference.org/ref/epsg/4326/
    )
    
    gdf = gdf.to_crs(epsg=3395)
    
    transformer = Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)
    bbox_columns = []
    for m in meters:
        gdf[f'buffer_{m}m'] = gdf.apply(lambda x: buffer_meters(x, m), axis=1)

        # Calculate in EPSG:3395
        gdf[f'buffer_{m}m_bbox'] = gdf[f'buffer_{m}m'].apply(lambda x: x.bounds)

        # Transform to EPSG:4326
        gdf[f'buffer_{m}m_bbox_4326'] = gdf[f'buffer_{m}m_bbox'].apply(lambda bbox: [
            transformer.transform(bbox[0], bbox[1]),  # minx, miny
            transformer.transform(bbox[2], bbox[3])   # maxx, maxy
        ])

        bbox_columns.append(f'buffer_{m}m_bbox_4326')
    
    gdf = gdf.to_crs(epsg=4326)

    return gdf[bbox_columns]


def get_ndvi(data):
    """
    According to the nomenclature of Sentinel2_GeoTIFF.ipynb, the bands are:
        dst.write(data_slice.B01,1)
        dst.write(data_slice.B04,2)
        dst.write(data_slice.B06,3) 
        dst.write(data_slice.B08,4)
    """
    return (data.sel(band=4) - data.sel(band=2))/(data.sel(band=4) + data.sel(band=2))


def get_vegetation_ratio(ndvi, threshold=0.4) -> float:
    vegetation_pixels = np.sum(ndvi > threshold).item()
    total_pixels = np.sum(~np.isnan(ndvi)).item()

    return vegetation_pixels / total_pixels if total_pixels > 0 else 0

