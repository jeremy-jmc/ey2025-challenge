'''
Welcome to the EY Open Science AI & Data Challenge 2025!
The objective of this challenge is to build a machine learning model to
predict urban heat island (UHI) hotspots in a city. By the end of the
challenge, you will have developed a regression model capable of predicting
the intensity of the UHI effect.

Participants will be given ground-level air temperature data in an index
format, which was collected on 24th July 2021 on traverse points in the
Bronx and Manhattan regions of New York city. This dataset constitutes
traverse points (latitude and longitude) and their corresponding UHI
(Urban Heat Island) index values. Participants will use this dataset to
build a regression model to predict UHI index values for a given set of
locations. It is important to understand that the UHI Index at any given
location is indicative of the relative temperature elevation at that
specific point compared to the city's average temperature.

This challenge is designed for participants with varying skill levels in
data science and programming, offering a great opportunity to apply your
knowledge and enhance your capabilities in the field.
'''

# Load in Dependencies
# python3 -m pip install scikit-learn tqdm

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

# Load the training data from csv file and display the first few rows to inspect the data
ground_df = pd.read_csv("Training_data_uhi_index.csv")
ground_df.head()

# ground_df['datetime'].value_counts()
ground_df.groupby(['Longitude', 'Latitude']).agg({'datetime': 'nunique'}).sort_values('datetime', ascending=False)
# .reset_index(name='counts')

# `lower_left` and `upper_right` variables of the "Sentinel2_GeoTIFF" notebook
ground_df[['Longitude', 'Latitude']].describe()

'''
TIP 1:
Participants might explore other combinations of bands from the Sentinel-2
and from other satellite datasets as well. For example, you can use mathematical
combinations of bands to generate various indices </a> which can then be used as
features in your model. These bands or indices may provide insights into surface
characteristics, vegetation, or built-up areas that could influence UHI patterns.
'''

'''
TIP 2:
Rather than extracting the bands for a single day coincident with the ground-based
data collection, participants might explore other options to improve data quality.
For example, one could select a different single date with minimal or no cloud
cover or generate a median mosaic using several scenes within a time series.
See the Sentinel-2 sample notebook for examples.
'''

# Downloading GeoTIFF Image
# Reads and plots four bands (B04, B08, B06, B01) from the GeoTIFF file.

# Open the GeoTIFF file
tiff_path = "S2_sample.tiff"

# Read the bands from the GeoTIFF file
with rasterio.open(tiff_path) as src1:
    band1 = src1.read(1)  # Band [B01]
    band2 = src1.read(2)  # Band [B04]
    band3 = src1.read(3)  # Band [B06]
    band4 = src1.read(4)  # Band [B08]

# Plot the bands in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axes for easier indexing
axes = axes.flatten()

# Plot the first band (B01)
im1 = axes[0].imshow(band1, cmap='viridis')
axes[0].set_title('Band [B01]')
fig.colorbar(im1, ax=axes[0])

# Plot the second band (B04)
im2 = axes[1].imshow(band2, cmap='viridis')
axes[1].set_title('Band [B04]')
fig.colorbar(im2, ax=axes[1])

# Plot the third band (B06)
im3 = axes[2].imshow(band3, cmap='viridis')                 
axes[2].set_title('Band [B06]')
fig.colorbar(im3, ax=axes[2])

# Plot the fourth band (B08)
im4 = axes[3].imshow(band4, cmap='viridis')
axes[3].set_title('Band [B08]')
fig.colorbar(im4, ax=axes[3])

plt.tight_layout()
plt.show()

'''
TIP 3:
Instead of a single point data extraction, participants might explore the
approach of creating a focal buffer around the locations (e.g., 50 m, 100 m,
150 m etc). For example, if the specified distance was 50 m and the specified
band was “Band 2”, then the value of the output pixels from this analysis
would reflect the average values in band 2 within 50 meters of the specific
location. This approach might help reduction in error associated with spatial
autocorrelation. In this demonstration notebook, we are extracting the band
data for each of the locations without creating a buffer zone.
'''

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

        # print(f"{B01_value=}, {B04_value=}, {B06_value=}, {B08_value=}")

    # Create a DataFrame with the band values
    # Create a DataFrame to store the band values
    df = pd.DataFrame()
    df['B01'] = B01_values
    df['B04'] = B04_values
    df['B06'] = B06_values
    df['B08'] = B08_values
    
    return df

try_data = map_satellite_data("S2_sample.tiff", ground_df.head(5).copy())

# Mapping satellite data with training data.
final_data = map_satellite_data('S2_sample.tiff', 'Training_data_uhi_index.csv')

final_data.head()

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
# See the Sentinel-2 sample notebook for more information about the NDVI index
final_data['NDVI'] = (final_data['B08'] - final_data['B04']) / (final_data['B08'] + final_data['B04'])
final_data['NDVI'] = final_data['NDVI'].replace([np.inf, -np.inf], np.nan) 

# Joining the predictor variables and response variables

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

# Combining ground data and final data into a single dataset.
uhi_data = combine_two_datasets(ground_df,final_data)
uhi_data.head()

# DataFrame to GeoDataFrame
geo_df = gpd.GeoDataFrame(uhi_data, geometry=gpd.points_from_xy(uhi_data.Longitude, uhi_data.Latitude))
geo_df.to_file("uhi_data.geojson", driver='GeoJSON')

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = ['B01', 'B04', 'B06', 'B08', 'NDVI']

for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')
uhi_data.head()

# Resetting the index of the dataset
uhi_data=  uhi_data.reset_index(drop=True)

''' MODEL BUILDING '''

# Retaining only the columns for B01, B06, NDVI, and UHI Index in the dataset.
uhi_data = uhi_data[['B01','B06','NDVI','UHI Index']]

# Split the data into features (X) and target (y), and then into training and testing sets
X = uhi_data.drop(columns=['UHI Index']).values
y = uhi_data ['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

'''
TIP 4:
There are many data preprocessing methods available, which might help to
improve the model performance. Participants should explore various
suitable preprocessing methods as well as different machine learning
algorithms to build a robust model.
'''

# Scale the training and test data using standardscaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

''' MODEL TRAINING '''

# Train the Random Forest model on the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

# Make predictions on the training data
insample_predictions = model.predict(X_train)

# calculate R-squared score for in-sample predictions
Y_train = y_train.tolist()
r2_score(Y_train, insample_predictions)

# Make predictions on the test data
outsample_predictions = model.predict(X_test)

# calculate R-squared score for out-sample predictions
Y_test = y_test.tolist()
r2_score(Y_test, outsample_predictions)

''' SUBMISSION '''
#Reading the coordinates for the submission
test_file = pd.read_csv('Submission_template.csv')
test_file.head()

# Mapping satellite data for submission.
val_data = map_satellite_data('S2_sample.tiff', 'Submission_template.csv')

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
val_data['NDVI'] = (val_data['B08'] - val_data['B04']) / (val_data['B08'] + val_data['B04'])
val_data['NDVI'] = val_data['NDVI'].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN

# Extracting specific columns (B01, B06, and NDVI) from the validation dataset
submission_val_data = val_data.loc[:,['B01','B06','NDVI']]

# Feature Scaling 
submission_val_data = submission_val_data.values
transformed_submission_data = sc.transform(submission_val_data)

#Making predictions
final_predictions = model.predict(transformed_submission_data)
final_prediction_series = pd.Series(final_predictions)

#Combining the results into dataframe
submission_df = pd.DataFrame({
    'Longitude': test_file['Longitude'].values, 
    'Latitude':test_file['Latitude'].values, 
    'UHI Index':final_prediction_series.values
})

#Dumping the predictions into a csv file.
submission_df.to_csv("submission.csv",index = False)

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

# print(f"R² Scores: {r2_scores}")
print(f"Mean R² Score: {np.mean(r2_scores):.4f}")
# print(f"Standard Deviation of R² Scores: {np.std(r2_scores):.4f}")