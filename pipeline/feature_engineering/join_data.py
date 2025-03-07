import sys
sys.path.append('../..')

from baseline.utilities import *

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=8)

SENTINEL_TIFF_PATH = '../../baseline/S2_sample.tiff' # './S2_sample_5res.tiff'
LANDSAT_TIFF_PATH = '../../baseline/Landsat_LST.tiff'
MODE = 'train'  # 'submission' 'train'

# Load the training data from csv file and display the first few rows to inspect the data
if MODE == 'train':
    ground_df = pd.read_csv("../../baseline/Training_data_uhi_index.csv")
elif MODE == 'submission':
    ground_df = pd.read_csv("../../baseline/Submission_template.csv")
else:
    raise ValueError("MODE should be either 'train' or 'submission")

sentinel_bands_df = pd.read_parquet(f'../data/processed/{MODE}/sentinel2_bands.parquet')
print(f"{sentinel_bands_df.columns=}")
sentinel_features_df = pd.read_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers.parquet')
print(f"{sentinel_features_df.columns=}")
sentinel_features_bands_df = pd.read_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers_bands.parquet')
print(f"{sentinel_features_bands_df.columns=}")
landsat_features_df = pd.read_parquet(f'../data/processed/{MODE}/landsat.parquet')
print(f"{landsat_features_df.columns=}")
ny_mesonet_features_df = pd.read_parquet(f'../data/processed/{MODE}/ny_mesonet_features.parquet')
print(f"{ny_mesonet_features_df.columns=}")
bldng_footprint = pd.read_parquet(f'../data/processed/{MODE}/building_footprint.parquet')
print(f"{bldng_footprint.columns=}")

# if MODE == 'train':
#     cluster_df = pd.read_parquet(f'../data/processed/{MODE}/cluster.parquet')

# -----------------------------------------------------------------------------
# * Joining the predictor variables and response variables
# -----------------------------------------------------------------------------

# Combining ground data, focal radius data and satellite bands data into a single dataset.
uhi_data = combine_two_datasets(ground_df,sentinel_bands_df)
uhi_data = combine_two_datasets(uhi_data, sentinel_features_df)
uhi_data = combine_two_datasets(uhi_data, landsat_features_df)
uhi_data = combine_two_datasets(uhi_data, sentinel_features_bands_df)
uhi_data = combine_two_datasets(uhi_data, ny_mesonet_features_df)
uhi_data = combine_two_datasets(uhi_data, bldng_footprint)

# if MODE == 'train':
#     uhi_data = combine_two_datasets(uhi_data, cluster_df)

all_features = uhi_data.copy()
for col in all_features.columns:
    print(col)

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = \
    ['B01', 'B06', 'UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'B11', 'B12'] # + \
    # ['NDVI', 'gNDBI', 'UI', 'NDBI', 'NBI', 'BRBA', 'NBAI', 'MBI', 'BAEI', 'gCI']

for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
if MODE == 'train':
    uhi_data.to_parquet(f'../data/processed/{MODE}/train_data.parquet')
    uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')

# Resetting the index of the dataset
uhi_data = uhi_data.reset_index(drop=True)
print(f"{uhi_data.shape=}")
print(uhi_data.isna().sum())

# Saving the dataset to a parquet file
if MODE == 'train':
    uhi_data.to_parquet(f'../data/processed/{MODE}/train_data.parquet')
elif MODE == 'submission':
    uhi_data.to_parquet(f'../data/processed/{MODE}/submission_data.parquet')
else:
    raise ValueError("MODE should be either 'train' or 'submission'")

print(f"{uhi_data.shape=}")

with open('../data/columns.json', mode='w') as f:
    json.dump({'features': [c for c in uhi_data.columns if c not in ['Longitude', 'Latitude', 'UHI Index', 'datetime']]}, f, indent=4)

print(f"{list(uhi_data.columns)=}")

import yaml
with open('../data/columns.yml', mode='w') as f:
    yaml.dump({'features': [c for c in uhi_data.columns if c not in ['Longitude', 'Latitude', 'UHI Index', 'datetime']]}, f)

# Open Yaml
feature_list = yaml.safe_load(open('../data/columns.yml', 'r'))['features']
print(len(feature_list))
print(feature_list)

# ls -lhR --block-size=M ./data/processed