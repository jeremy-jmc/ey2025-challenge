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
# !pip install pyarrow fastparquet pandarallel

import sys
sys.path.append('..')

from baseline.utilities import *

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

# from utilities import *

SENTINEL_TIFF_PATH = '../baseline/S2_sample.tiff' # './S2_sample_5res.tiff'
LANDSAT_TIFF_PATH = '../baseline/Landsat_LST.tiff'

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
satellite_bands_df = map_satellite_data(SENTINEL_TIFF_PATH, ground_df)

display(satellite_bands_df.head())

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
# See the Sentinel-2 sample notebook for more information about the NDVI index
satellite_bands_df['NDVI'] = (satellite_bands_df['B08'] - satellite_bands_df['B04']) / (satellite_bands_df['B08'] + satellite_bands_df['B04'])
satellite_bands_df['NDVI'] = satellite_bands_df['NDVI'].replace([np.inf, -np.inf], np.nan) 

satellite_bands_df['gNDBI'] = (satellite_bands_df['B08'] - satellite_bands_df['B03']) / (satellite_bands_df['B08'] + satellite_bands_df['B03'])
satellite_bands_df['gNDBI'] = satellite_bands_df['gNDBI'].replace([np.inf, -np.inf], np.nan) 


# -----------------------------------------------------------------------------
# Data Preprocessing: Generate geographic bounding boxes around the coordinates
# -----------------------------------------------------------------------------

radius_list = [50, 100, 150, 200, 250]
bbox_dataset = compute_geographic_bounding_boxes(ground_df[['Longitude', 'Latitude']], radius_list)
print(bbox_dataset.columns)

# -----------------------------------------------------------------------------
# Feature Engineering: Extract features from the bounding boxes extracted above using Landsat TIFF
# -----------------------------------------------------------------------------

landsat_features_df = bbox_dataset.copy()

sample = landsat_features_df['buffer_50m_bbox_4326'].iloc[0]
selection = get_bbox_selection(LANDSAT_TIFF_PATH, sample)
lwir11_arr = selection.sel(band=1).to_numpy()
print(f"{lwir11_arr.shape=}")

fig, ax = plt.subplots(figsize=(11,10))
selection.sel(band=1).plot.imshow(vmin=20.0, vmax=45.0, cmap="jet")
plt.title("Land Surface Temperature (LST)")
plt.axis('off')
plt.show()

print(np.nanmean(lwir11_arr), np.nanstd(lwir11_arr))
print(np.nanmean(selection.sel(band=1)), np.nanstd(selection.sel(band=1)))

landsat_feature_list = []
for r in radius_list:
    print(f"Processing {r}m radius")

    landsat_features_df[f'ldnst_buffer_{r}m_selection'] = landsat_features_df[f'buffer_{r}m_bbox_4326'].parallel_apply(
        lambda bbox: get_bbox_selection(LANDSAT_TIFF_PATH, bbox)
    )

    landsat_features_df[f'lndst_mean_lwir11_{r}m'] = landsat_features_df[f'ldnst_buffer_{r}m_selection'].parallel_apply(
        lambda patch: np.nanmean(patch.sel(band=1))
    )

    landsat_features_df[f'lndst_std_lwir11_{r}m'] = landsat_features_df[f'ldnst_buffer_{r}m_selection'].parallel_apply(
        lambda patch: np.nanstd(patch.sel(band=1))
    )
    
    landsat_feature_list.extend([f'lndst_mean_lwir11_{r}m', f'lndst_std_lwir11_{r}m'])


landsat_data = rxr.open_rasterio(LANDSAT_TIFF_PATH)

landsat_features_df['lndst_lwir_point'] = ground_df[['Latitude', 'Longitude']].progress_apply(
    lambda x: landsat_data.sel(x=x['Longitude'], y=x['Latitude'], method='nearest').values[0],
    axis=1
)
landsat_feature_list.append('lndst_lwir_point')

# -----------------------------------------------------------------------------
# Feature Engineering: Extract features from the bounding boxes extracted above using Sentinel 2 TIFF
# -----------------------------------------------------------------------------

sentinel_features_df = bbox_dataset.copy()

sentinel_data = rxr.open_rasterio(SENTINEL_TIFF_PATH)

for r in radius_list:
    sentinel_features_df[f'sntnl_buffer_{r}m_selection'] = sentinel_features_df[f'buffer_{r}m_bbox_4326'].parallel_apply(
        lambda bbox: get_bbox_selection(SENTINEL_TIFF_PATH, bbox)
    )

sentinel_focal_radius_ft = []

for r in radius_list:
    for b in sentinel_data.band.to_numpy():
        print(f"{r=}, {b=}")
        sentinel_features_df[f'sntnl_buffer_band_{b}_{r}_mean'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
            lambda patch: np.nanmean(patch.sel(band=b))
        )
        sentinel_features_df[f'sntnl_buffer_band_{b}_{r}_std'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
            lambda patch: np.nanstd(patch.sel(band=b))
        )
        sentinel_focal_radius_ft.extend([f'sntnl_buffer_band_{b}_{r}_mean', f'sntnl_buffer_band_{b}_{r}_std'])

for r in radius_list:
    """
    According to the nomenclature of Sentinel2_GeoTIFF.ipynb, the bands are:
        dst.write(data_slice.B01, 1)
        dst.write(data_slice.B04, 2)
        dst.write(data_slice.B06, 3) 
        dst.write(data_slice.B08, 4)
    """
    print(f"Processing {r}m radius")
    sentinel_features_df[f'sntnl_ndvi_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=4) - patch.sel(band=2))/(patch.sel(band=4) + patch.sel(band=2))
        # get_ndvi(bbox)
    )

    sentinel_features_df[f'sntnl_mean_ndvi_{r}m'] = sentinel_features_df[f'sntnl_ndvi_{r}m'].parallel_apply(
        lambda ndvi: np.nanmean(ndvi)
    )

    sentinel_features_df[f'sntnl_vegetation_ratio_ndvi_{r}m'] = sentinel_features_df[f'sntnl_ndvi_{r}m'].parallel_apply(
        lambda ndvi: get_vegetation_ratio(ndvi)
    )

    sentinel_focal_radius_ft.extend([f'sntnl_mean_ndvi_{r}m', f'sntnl_vegetation_ratio_ndvi_{r}m'])
    print()

display(sentinel_features_df[sentinel_focal_radius_ft].head())


# -----------------------------------------------------------------------------
# * Joining the predictor variables and response variables
# -----------------------------------------------------------------------------

# Combining ground data, focal radius data and satellite bands data into a single dataset.
uhi_data = combine_two_datasets(ground_df,satellite_bands_df)
uhi_data = combine_two_datasets(uhi_data, sentinel_features_df[sentinel_focal_radius_ft])
uhi_data = combine_two_datasets(uhi_data, landsat_features_df[landsat_feature_list])

all_features = uhi_data.copy()

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = ['B01', 'B06', 'NDVI', 'UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'B11', 'B12', 'gNDBI']

for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')

# Resetting the index of the dataset
uhi_data = uhi_data.reset_index(drop=True)

# Saving the dataset to a parquet file
uhi_data.to_parquet('./data/train_data.parquet')

# Saving the extracted column names into a JSON file
with open('./data/columns.json', 'w') as f:
    json.dump({
        'focal_radius_features': sentinel_focal_radius_ft + landsat_feature_list,
    }, f)

print(f"{list(uhi_data.columns)=}")

"""
INSIGHT:
    The mean and the std on each focal buffer using the B01 extracted from Sentinel2 TIFF improves the model
"""
