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

# `lower_left` and `upper_right` variables of the "Sentinel2_GeoTIFF" notebook
display(ground_df[['Longitude', 'Latitude']].describe())


radius_list = json.loads(open("../data/radius_list.json", "r").read())['radius_list']
print(f"{radius_list=}")

bbox_dataset = pd.read_parquet(f'../data/processed/{MODE}/bbox_dataset.parquet')


# -----------------------------------------------------------------------------
# Feature Engineering: Extract features from the bounding boxes extracted above using Landsat TIFF
# -----------------------------------------------------------------------------

landsat_features_df = bbox_dataset.copy()

sample = landsat_features_df['buffer_50m_bbox_4326'].iloc[0]
selection = get_bbox_selection(LANDSAT_TIFF_PATH, sample)
lwir11_arr = selection.sel(band=1).to_numpy()
print(f"{lwir11_arr.shape=}")

fig, ax = plt.subplots(figsize=(5, 5))
selection.sel(band=1).plot.imshow(vmin=20.0, vmax=45.0, cmap="jet")
plt.title("Land Surface Temperature (LST)")
plt.axis('off')
plt.show()

print(np.nanmean(lwir11_arr), np.nanstd(lwir11_arr))
print(np.nanmean(selection.sel(band=1)), np.nanstd(selection.sel(band=1)))


landsat_feature_list = []
for r in tqdm(radius_list, total=len(radius_list), desc='Landsat LWIR11'):
    # print(f"Processing {r}m radius")

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

display(landsat_features_df[landsat_feature_list])

# Save data
landsat_features_df[landsat_feature_list].to_parquet(f'../data/processed/{MODE}/landsat.parquet')

# Update column groups
column_groups = json.loads(open("../data/column_groups.json").read())

column_groups['landsat_focal_buffer_features'] = landsat_feature_list

with open('../data/column_groups.json', 'w') as f:
    f.write(json.dumps(column_groups, indent=4))

