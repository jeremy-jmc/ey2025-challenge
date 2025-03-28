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


radius_list = json.loads(open("../data/radius_list.json", "r").read())['radius_list']
print(f"{radius_list=}")

bbox_dataset = pd.read_parquet(f'../data/processed/{MODE}/bbox_dataset.parquet')


# -----------------------------------------------------------------------------
# Feature Engineering: Extract features from the bounding boxes extracted above using Sentinel 2 TIFF
# -----------------------------------------------------------------------------

sentinel_features_df = bbox_dataset.copy()

sentinel_data = rxr.open_rasterio(SENTINEL_TIFF_PATH)

for r in radius_list:
    sentinel_features_df[f'sntnl_buffer_{r}m_selection'] = sentinel_features_df[f'buffer_{r}m_bbox_4326'].parallel_apply(
        lambda bbox: get_bbox_selection(SENTINEL_TIFF_PATH, bbox)
    )
    sentinel_features_df = sentinel_features_df.drop(columns=[f'buffer_{r}m_bbox_4326'])

sentinel_focal_radius_ft = []

for r in radius_list:   # tqdm(radius_list, total=len(radius_list), desc='Sentinel-2 Bands')
    for b in sentinel_data.band.to_numpy():
        print(f"{r=}, {b=}")
        sentinel_features_df[f'sntnl_buffer_band_{b}_{r}_mean'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
            lambda patch: np.nanmean(patch.sel(band=b))
        )
        sentinel_features_df[f'sntnl_buffer_band_{b}_{r}_std'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
            lambda patch: np.nanstd(patch.sel(band=b))
        )
        sentinel_focal_radius_ft.extend([f'sntnl_buffer_band_{b}_{r}_mean', f'sntnl_buffer_band_{b}_{r}_std'])

display(sentinel_features_df[sentinel_focal_radius_ft].head())

# Save data
sentinel_features_df[sentinel_focal_radius_ft].to_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers_bands.parquet')

# # Update column groups
# column_groups = json.loads(open("../data/column_groups.json").read())

# column_groups['sentinel2_focal_buffer_features_bands'] = sentinel_focal_radius_ft

# with open('../data/column_groups.json', 'w') as f:
#     f.write(json.dumps(column_groups, indent=4))

