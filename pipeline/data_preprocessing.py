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

'''
TIP 4:
There are many data preprocessing methods available, which might help to
improve the model performance. Participants should explore various
suitable preprocessing methods as well as different machine learning
algorithms to build a robust model.
'''
import sys
sys.path.append('..')

from baseline.utilities import *
# from utilities import *

TIFF_PATH = '../baseline/S2_sample.tiff' # './S2_sample_5res.tiff'

# Load the training data from csv file and display the first few rows to inspect the data
ground_df = pd.read_csv("../baseline/Training_data_uhi_index.csv")
display(ground_df.head())

# ground_df['datetime'].value_counts()
display(
    ground_df.groupby(['Longitude', 'Latitude']).agg({'datetime': 'nunique'})
    .sort_values('datetime', ascending=False)
)
# .reset_index(name='counts')

# `lower_left` and `upper_right` variables of the "Sentinel2_GeoTIFF" notebook
display(ground_df[['Longitude', 'Latitude']].describe())


# -----------------------------------------------------------------------------
# Feature Engineering: Obtain features from Sentinel2 TIFF
# -----------------------------------------------------------------------------

# try_data = map_satellite_data("S2_sample.tiff", ground_df.head(5).copy())

# Mapping satellite data with training data.
satellite_bands_df = map_satellite_data(TIFF_PATH, ground_df)

display(satellite_bands_df.head())

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
# See the Sentinel-2 sample notebook for more information about the NDVI index
satellite_bands_df['NDVI'] = (satellite_bands_df['B08'] - satellite_bands_df['B04']) / (satellite_bands_df['B08'] + satellite_bands_df['B04'])
satellite_bands_df['NDVI'] = satellite_bands_df['NDVI'].replace([np.inf, -np.inf], np.nan) 

satellite_bands_df['gNDBI'] = (satellite_bands_df['B08'] - satellite_bands_df['B03']) / (satellite_bands_df['B08'] + satellite_bands_df['B03'])
satellite_bands_df['gNDBI'] = satellite_bands_df['gNDBI'].replace([np.inf, -np.inf], np.nan) 


# -----------------------------------------------------------------------------
# Feature Engineering: Generate buffers around the coordinates and extract features
# -----------------------------------------------------------------------------

radius_list = [50, 150, 100, 200, 250]
bbox_dataset = get_bbox_radius(ground_df[['Longitude', 'Latitude']], radius_list)

for r in radius_list:
    bbox_dataset[f'buffer_{r}m_selection'] = bbox_dataset[f'buffer_{r}m_bbox_4326'].progress_apply(
        lambda bbox: get_bbox_selection(TIFF_PATH, bbox)
    )

buffer_radius_features = []
for r in radius_list:
    """
    According to the nomenclature of Sentinel2_GeoTIFF.ipynb, the bands are:
        dst.write(data_slice.B01, 1)
        dst.write(data_slice.B04, 2)
        dst.write(data_slice.B06, 3) 
        dst.write(data_slice.B08, 4)
    """
    bbox_dataset[f'ndvi_buffer_{r}m'] = bbox_dataset[f'buffer_{r}m_selection'].progress_apply(
        lambda bbox: (bbox.sel(band=4) - bbox.sel(band=2))/(bbox.sel(band=4) + bbox.sel(band=2))
        # get_ndvi(bbox)
    )

    bbox_dataset[f'ndvi_buffer_{r}m_mean'] = bbox_dataset[f'ndvi_buffer_{r}m'].progress_apply(
        lambda ndvi: np.nanmean(ndvi)
    )

    bbox_dataset[f'vegetation_ratio_ndvi_{r}m'] = bbox_dataset[f'ndvi_buffer_{r}m'].progress_apply(
        lambda ndvi: get_vegetation_ratio(ndvi)
    )

    buffer_radius_features.extend([f'ndvi_buffer_{r}m_mean', f'vegetation_ratio_ndvi_{r}m'])

display(bbox_dataset[buffer_radius_features].head())


# -----------------------------------------------------------------------------
# * Joining the predictor variables and response variables
# -----------------------------------------------------------------------------

# Combining ground data, focal radius data and satellite bands data into a single dataset.
uhi_data = combine_two_datasets(ground_df,satellite_bands_df)
uhi_data = combine_two_datasets(uhi_data, bbox_dataset[buffer_radius_features])

all_features = uhi_data.copy()

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = ['B01','B06','NDVI','UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'B11', 'B12', 'gNDBI'] + buffer_radius_features

for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')

# Resetting the index of the dataset
uhi_data = uhi_data.reset_index(drop=True)

# Saving the dataset to a parquet file
# !pip install pyarrow fastparquet
uhi_data.to_parquet('./data/train_data.parquet')

# Saving the extracted column names into a JSON file
with open('./data/columns.json', 'w') as f:
    json.dump({
        'focal_radius_features': buffer_radius_features,
    }, f)