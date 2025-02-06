
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
from sklearn.metrics import r2_score

# Planetary Computer Tools
import pystac_client
import planetary_computer as pc
from pystac.extensions.eo import EOExtension as eo

# Others
import os
from tqdm import tqdm

tqdm.pandas()

# -----------------------------------------------------------------------------
# EY Functions
# -----------------------------------------------------------------------------


# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1, dataset2):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.concat([dataset1, dataset2], axis=1)
    return data


# Extracts satellite band values from a GeoTIFF based on coordinates from a csv file and returns them in a DataFrame.
def map_satellite_data(tiff_path, csv_path_or_df):
    
    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    print(type(data))
    tiff_crs = data.rio.crs

    # Read the Excel file using pandas
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    elif isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        raise ValueError("csv_path_or_df must be a file path or a DataFrame")
    
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values

    # 3. Convert lat/long to the GeoTIFF's CRS
    # Create a Proj object for EPSG:4326 (WGS84 - lat/long) and the GeoTIFF's CRS
    proj_wgs84 = Proj(init='epsg:4326')  # EPSG:4326 is the common lat/long CRS
    proj_tiff = Proj(tiff_crs)
    
    # Create a transformer object
    transformer = Transformer.from_proj(proj_wgs84, proj_tiff)

    B01_values = []
    B04_values = []
    B06_values = []
    B08_values = []
    B02_values = []
    B03_values = []
    B05_values = []
    B07_values = []
    B8A_values = []
    B11_values = []
    B12_values = []

    # Iterate over the latitudes and longitudes, and extract the corresponding band values
    for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
    # Assuming the correct dimensions are 'y' and 'x' (replace these with actual names from data.coords)
    
        B01_value = data.sel(x=lon, y=lat,  band=1, method="nearest").values
        B01_values.append(B01_value)
    
        B04_value = data.sel(x=lon, y=lat, band=2, method="nearest").values
        B04_values.append(B04_value)
        
        B06_value = data.sel(x=lon, y=lat, band=3, method="nearest").values
        B06_values.append(B06_value)
    
        B08_value = data.sel(x=lon, y=lat, band=4, method="nearest").values
        B08_values.append(B08_value)
        
        B02_value = data.sel(x=lon, y=lat,  band=5, method="nearest").values
        B02_values.append(B02_value)
    
        B03_value = data.sel(x=lon, y=lat, band=6, method="nearest").values
        B03_values.append(B03_value)
        
        B05_value = data.sel(x=lon, y=lat, band=7, method="nearest").values
        B05_values.append(B05_value)
    
        B07_value = data.sel(x=lon, y=lat, band=8, method="nearest").values
        B07_values.append(B07_value)
        
        B8A_value = data.sel(x=lon, y=lat,  band=9, method="nearest").values
        B8A_values.append(B8A_value)
    
        B11_value = data.sel(x=lon, y=lat, band=10, method="nearest").values
        B11_values.append(B11_value)
        
        B12_value = data.sel(x=lon, y=lat, band=11, method="nearest").values
        B12_values.append(B12_value)

        # print(f"{B01_value=}, {B04_value=}, {B06_value=}, {B08_value=}")

    # Create a DataFrame with the band values
    # Create a DataFrame to store the band values
    df = pd.DataFrame()
    df['B01'] = B01_values
    df['B04'] = B04_values
    df['B06'] = B06_values
    df['B08'] = B08_values
    df['B02'] = B02_values
    df['B03'] = B03_values
    df['B05'] = B05_values
    df['B07'] = B07_values
    df['B8A'] = B8A_values
    df['B11'] = B11_values
    df['B12'] = B12_values
    
    return df


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


def get_bbox_selection(tiff_path, bbox):
    lower_left, upper_right = bbox

    data = rxr.open_rasterio(tiff_path)

    selection = data.sel(
        x=slice(lower_left[0], upper_right[0]),
        y=slice(upper_right[1], lower_left[1])
    )

    return selection
