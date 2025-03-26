import sys
sys.path.append('../..')

from baseline.utilities import *

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=8)


SENTINEL_TIFF_PATH = '../../baseline/S2_sample.tiff' # './S2_sample_5res.tiff'
LANDSAT_TIFF_PATH = '../../baseline/Landsat_LST.tiff'
MODE = 'submission'  # 'submission' 'train'

# Load the training data from csv file and display the first few rows to inspect the data
if MODE == 'train':
    ground_df = pd.read_csv("../../baseline/Training_data_uhi_index.csv")
elif MODE == 'submission':
    ground_df = pd.read_csv("../../baseline/Submission_template.csv")
else:
    raise ValueError("MODE should be either 'train' or 'submission")

# `lower_left` and `upper_right` variables of the "Sentinel2_GeoTIFF" notebook
display(ground_df[['Longitude', 'Latitude']].describe())


# -----------------------------------------------------------------------------
# Feature Engineering: Obtain features from Sentinel2 TIFF
# -----------------------------------------------------------------------------

# * Sentinel-2 bands: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/
"""
- B01 (aerosol)
- B02 (blue)
- B03 (green)
- B04 (red)
- B05 (red edge)
- B06
- B07
- B08 (NIR)
- B8A
- B09
- B10
- B11 (SWIR 1)
- B12 (SWIR 2)
"""

# try_data = map_satellite_data("S2_sample.tiff", ground_df.head(5).copy())

# Mapping satellite data with training data.
sentinel2_bands_df = map_satellite_data(SENTINEL_TIFF_PATH, ground_df)

display(sentinel2_bands_df.head())

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
# NDVI = (NIR - Red) / (NIR + Red)
sentinel2_bands_df['NDVI'] = (sentinel2_bands_df['B08'] - sentinel2_bands_df['B04']) / (sentinel2_bands_df['B08'] + sentinel2_bands_df['B04'])
sentinel2_bands_df['NDVI'] = sentinel2_bands_df['NDVI'].replace([np.inf, -np.inf], np.nan) 

# gNDBI = (NIR - Green) / (NIR + Green)
sentinel2_bands_df['gNDBI'] = (sentinel2_bands_df['B08'] - sentinel2_bands_df['B03']) / (sentinel2_bands_df['B08'] + sentinel2_bands_df['B03'])
sentinel2_bands_df['gNDBI'] = sentinel2_bands_df['gNDBI'].replace([np.inf, -np.inf], np.nan) 

# UI (Urban Index) = (SWIR1 - NIR) / (SWIR1 + NIR)
sentinel2_bands_df['UI'] = (sentinel2_bands_df['B11'] - sentinel2_bands_df['B08']) / (sentinel2_bands_df['B11'] + sentinel2_bands_df['B08'])
sentinel2_bands_df['UI'] = sentinel2_bands_df['UI'].replace([np.inf, -np.inf], np.nan) 

# NDBI (Normalized Difference Built−up Index) = (SWIR2 - NIR) / (SWIR2 + NIR)
sentinel2_bands_df['NDBI'] = (sentinel2_bands_df['B12'] - sentinel2_bands_df['B08']) / (sentinel2_bands_df['B12'] + sentinel2_bands_df['B08'])
sentinel2_bands_df['NDBI'] = sentinel2_bands_df['NDBI'].replace([np.inf, -np.inf], np.nan) 

# NBI (New built-up index) = (Red * SWIR2) / NIR
sentinel2_bands_df['NBI'] = (sentinel2_bands_df['B04'] * sentinel2_bands_df['B12']) / (sentinel2_bands_df['B08'])
sentinel2_bands_df['NBI'] = sentinel2_bands_df['NBI'].replace([np.inf, -np.inf], np.nan) 

# BRBA (Band ratio for built-up area) = Red / SWIR2
sentinel2_bands_df['BRBA'] = (sentinel2_bands_df['B04']) / (sentinel2_bands_df['B12'])
sentinel2_bands_df['BRBA'] = sentinel2_bands_df['BRBA'].replace([np.inf, -np.inf], np.nan)

# NBAI (Normalized built-up area index) = (SWIR2 - SWIR1) / NIR / (SWIR2 + SWIR1) / NIR
sentinel2_bands_df['NBAI'] = ((sentinel2_bands_df['B12'] - sentinel2_bands_df['B11']) / sentinel2_bands_df['B03']) / ((sentinel2_bands_df['B12'] + sentinel2_bands_df['B11']) / sentinel2_bands_df['B03'])
sentinel2_bands_df['NBAI'] = sentinel2_bands_df['NBAI'].replace([np.inf, -np.inf], np.nan)

# MBI (Modified built-up index) = (SWIR1 * Red - NIR * NIR) / (Red + NIR + SWIR1)
sentinel2_bands_df['MBI'] = (sentinel2_bands_df['B11'] * sentinel2_bands_df['B04'] - (sentinel2_bands_df['B08'] * sentinel2_bands_df['B08'])) / (sentinel2_bands_df['B04'] + sentinel2_bands_df['B08'] + sentinel2_bands_df['B11'])
sentinel2_bands_df['MBI'] = sentinel2_bands_df['MBI'].replace([np.inf, -np.inf], np.nan)

# BAEI (Built-up area extraction index) = (Red + 0.3) / (Green + SWIR1)
sentinel2_bands_df['BAEI'] = (sentinel2_bands_df['B04'] + 0.3) / (sentinel2_bands_df['B03'] + sentinel2_bands_df['B11'])
sentinel2_bands_df['BAEI'] = sentinel2_bands_df['BAEI'].replace([np.inf, -np.inf], np.nan)

# gCI = NIR / Green - 1
sentinel2_bands_df['gCI'] = (sentinel2_bands_df['B08'] / sentinel2_bands_df['B03']) - 1
sentinel2_bands_df['gCI'] = sentinel2_bands_df['gCI'].replace([np.inf, -np.inf], np.nan)

# TODO: implement more indices. Source: https://www.sciencedirect.com/science/article/pii/S2667010022001251
sentinel_2_indices = ['NDVI', 'gNDBI', 'UI', 'NDBI', 'NBI', 'BRBA', 'NBAI', 'MBI', 'BAEI', 'gCI']

display(sentinel2_bands_df.head())
print(sentinel2_bands_df.dtypes)

for col in sentinel2_bands_df.columns:
    sentinel2_bands_df[col] = sentinel2_bands_df[col].astype(float)

print(sentinel2_bands_df.std())

# Save data
os.makedirs(f'../data/processed/{MODE}', exist_ok=True)
sentinel2_bands_df.to_parquet(f'../data/processed/{MODE}/sentinel2_bands.parquet')

# # Update column groups
# column_groups = json.loads(open("../data/column_groups.json").read())

# column_groups['sentinel2_bands'] = [col for col in sentinel2_bands_df.columns if col not in sentinel_2_indices]
# column_groups['sentinel2_band_indices'] = sentinel_2_indices

# with open('../data/column_groups.json', 'w') as f:
#     f.write(json.dumps(column_groups, indent=4))

