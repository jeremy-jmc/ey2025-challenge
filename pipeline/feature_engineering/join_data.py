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
sentinel_features_df = pd.read_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers.parquet')
landsat_features_df = pd.read_parquet(f'../data/processed/{MODE}/landsat.parquet')
ny_mesonet_features_df = pd.read_parquet(f'../data/processed/{MODE}/ny_mesonet_features.parquet')

# -----------------------------------------------------------------------------
# * Joining the predictor variables and response variables
# -----------------------------------------------------------------------------

# Combining ground data, focal radius data and satellite bands data into a single dataset.
uhi_data = combine_two_datasets(ground_df,sentinel_bands_df)
uhi_data = combine_two_datasets(uhi_data, sentinel_features_df)
uhi_data = combine_two_datasets(uhi_data, landsat_features_df)
uhi_data = combine_two_datasets(uhi_data, ny_mesonet_features_df)

all_features = uhi_data.copy()
for col in all_features.columns:
    print(col)

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = ['B01', 'B06', 'NDVI', 'UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'B11', 'B12', 'gNDBI']

for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
if MODE == 'train':
    uhi_data.to_parquet(f'../data/processed/{MODE}/train_data.parquet')
    uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')

# Resetting the index of the dataset
uhi_data = uhi_data.reset_index(drop=True)
print(uhi_data.isna().sum())

# Saving the dataset to a parquet file
if MODE == 'train':
    uhi_data.to_parquet(f'../data/processed/{MODE}/train_data.parquet')
elif MODE == 'submission':
    uhi_data.to_parquet(f'../data/processed/{MODE}/submission_data.parquet')
else:
    raise ValueError("MODE should be either 'train' or 'submission'")

# Saving the extracted column names into a JSON file
column_groups = json.loads(open('../data/column_groups.json').read())
column_names = []
for k, v in column_groups.items():
    column_names += v

with open('../data/columns.json', 'w') as f:
    json.dump({'features': column_names}, f, indent=4)

print(f"{list(uhi_data.columns)=}")

