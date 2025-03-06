import sys
sys.path.append('../..')

from baseline.utilities import *

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=12)


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

for r in tqdm(radius_list, total=len(radius_list), desc='Sentinel-2 Indexes Mean-Std'):
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

    According to the nomenclature of Sentinel2_GeoTIFF.ipynb, the bands are:
        dst.write(data_slice.B01, 1)
        dst.write(data_slice.B04, 2)
        dst.write(data_slice.B06, 3) 
        dst.write(data_slice.B08, 4)
        dst.write(data_slice.B02, 5)
        dst.write(data_slice.B03, 6)
        dst.write(data_slice.B05, 7)
        dst.write(data_slice.B07, 8)
        dst.write(data_slice.B08A, 9)
        dst.write(data_slice.B11, 10)
        dst.write(data_slice.B12, 11)
    """
    # print(f"Processing {r}m radius")
    sentinel_features_df[f'sntnl_ndvi_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=4) - patch.sel(band=2))/(patch.sel(band=4) + patch.sel(band=2))
        # get_ndvi(bbox)
    )

    # NDVI = (NIR - Red) / (NIR + Red)
    sentinel_features_df[f'sntnl_mean_ndvi_{r}m'] = sentinel_features_df[f'sntnl_ndvi_{r}m'].parallel_apply(
        lambda ndvi: np.nanmean(ndvi)
    )
    sentinel_features_df[f'sntnl_std_ndvi_{r}m'] = sentinel_features_df[f'sntnl_ndvi_{r}m'].parallel_apply(
        lambda ndvi: np.nanstd(ndvi)
    )
    sentinel_features_df[f'sntnl_vegetation_ratio_ndvi_{r}m'] = sentinel_features_df[f'sntnl_ndvi_{r}m'].parallel_apply(
        lambda ndvi: get_vegetation_ratio(ndvi)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_ndvi_{r}m', f'sntnl_std_ndvi_{r}m'])
    sentinel_focal_radius_ft.extend([f'sntnl_vegetation_ratio_ndvi_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_ndvi_{r}m'])
    

    # gNDBI = (NIR - Green) / (NIR + Green)
    sentinel_features_df[f'sntnl_gndbi_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=4) - patch.sel(band=6))/(patch.sel(band=4) + patch.sel(band=6))
    )
    sentinel_features_df[f'sntnl_mean_gndbi_{r}m'] = sentinel_features_df[f'sntnl_gndbi_{r}m'].parallel_apply(
        lambda gndbi: np.nanmean(gndbi)
    )
    sentinel_features_df[f'sntnl_std_gndbi_{r}m'] = sentinel_features_df[f'sntnl_gndbi_{r}m'].parallel_apply(
        lambda gndbi: np.nanstd(gndbi)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_gndbi_{r}m', f'sntnl_std_gndbi_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_gndbi_{r}m'])


    # UI (Urban Index) = (SWIR1 - NIR) / (SWIR1 + NIR)
    sentinel_features_df[f'sntnl_ui_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=10) - patch.sel(band=4))/(patch.sel(band=10) + patch.sel(band=4))
    )
    sentinel_features_df[f'sntnl_mean_ui_{r}m'] = sentinel_features_df[f'sntnl_ui_{r}m'].parallel_apply(
        lambda ui: np.nanmean(ui)
    )
    sentinel_features_df[f'sntnl_std_ui_{r}m'] = sentinel_features_df[f'sntnl_ui_{r}m'].parallel_apply(
        lambda ui: np.nanstd(ui)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_ui_{r}m', f'sntnl_std_ui_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_ui_{r}m'])


    # NDBI (Normalized Difference Built−up Index) = (SWIR2 - NIR) / (SWIR2 + NIR)
    sentinel_features_df[f'sntnl_ndbi_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=11) - patch.sel(band=4))/(patch.sel(band=11) + patch.sel(band=4))
    )
    sentinel_features_df[f'sntnl_mean_ndbi_{r}m'] = sentinel_features_df[f'sntnl_ndbi_{r}m'].parallel_apply(
        lambda ndbi: np.nanmean(ndbi)
    )
    sentinel_features_df[f'sntnl_std_ndbi_{r}m'] = sentinel_features_df[f'sntnl_ndbi_{r}m'].parallel_apply(
        lambda ndbi: np.nanstd(ndbi)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_ndbi_{r}m', f'sntnl_std_ndbi_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_ndbi_{r}m'])


    # NBI (New built-up index) = (Red * SWIR2) / NIR
    sentinel_features_df[f'sntnl_nbi_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=2) * patch.sel(band=11))/(patch.sel(band=4))
    )
    sentinel_features_df[f'sntnl_mean_nbi_{r}m'] = sentinel_features_df[f'sntnl_nbi_{r}m'].parallel_apply(
        lambda nbi: np.nanmean(nbi)
    )
    sentinel_features_df[f'sntnl_std_nbi_{r}m'] = sentinel_features_df[f'sntnl_nbi_{r}m'].parallel_apply(
        lambda nbi: np.nanstd(nbi)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_nbi_{r}m', f'sntnl_std_nbi_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_nbi_{r}m'])


    # BRBA (Band ratio for built-up area) = Red / SWIR2
    sentinel_features_df[f'sntnl_brba_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: (patch.sel(band=2))/(patch.sel(band=11))
    )
    sentinel_features_df[f'sntnl_mean_brba_{r}m'] = sentinel_features_df[f'sntnl_brba_{r}m'].parallel_apply(
        lambda brba: np.nanmean(brba)
    )
    sentinel_features_df[f'sntnl_std_brba_{r}m'] = sentinel_features_df[f'sntnl_brba_{r}m'].parallel_apply(
        lambda brba: np.nanstd(brba)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_brba_{r}m', f'sntnl_std_brba_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_brba_{r}m'])


    # NBAI (Normalized built-up area index) = (SWIR2 - SWIR1) / NIR / (SWIR2 + SWIR1) / NIR
    sentinel_features_df[f'sntnl_nbai_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: ((patch.sel(band=11) - patch.sel(band=10))/patch.sel(band=4))/((patch.sel(band=11) + patch.sel(band=10))/patch.sel(band=4))
    )
    sentinel_features_df[f'sntnl_mean_nbai_{r}m'] = sentinel_features_df[f'sntnl_nbai_{r}m'].parallel_apply(
        lambda nbai: np.nanmean(nbai)
    )
    sentinel_features_df[f'sntnl_std_nbai_{r}m'] = sentinel_features_df[f'sntnl_nbai_{r}m'].parallel_apply(
        lambda nbai: np.nanstd(nbai)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_nbai_{r}m', f'sntnl_std_nbai_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_nbai_{r}m'])


    # MBI (Modified built-up index) = (SWIR1 * Red - NIR * NIR) / (Red + NIR + SWIR1)
    sentinel_features_df[f'sntnl_mbi_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: ((patch.sel(band=10) * patch.sel(band=2) - (patch.sel(band=4) * patch.sel(band=4)))/(patch.sel(band=2) + patch.sel(band=4) + patch.sel(band=10)))
    )
    sentinel_features_df[f'sntnl_mean_mbi_{r}m'] = sentinel_features_df[f'sntnl_mbi_{r}m'].parallel_apply(
        lambda mbi: np.nanmean(mbi)
    )
    sentinel_features_df[f'sntnl_std_mbi_{r}m'] = sentinel_features_df[f'sntnl_mbi_{r}m'].parallel_apply(
        lambda mbi: np.nanstd(mbi)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_mbi_{r}m', f'sntnl_std_mbi_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_mbi_{r}m'])


    # BAEI (Built-up area extraction index) = (Red + 0.3) / (Green + SWIR1)
    sentinel_features_df[f'sntnl_baei_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: ((patch.sel(band=2) + 0.3)/(patch.sel(band=6) + patch.sel(band=10)))
    )
    sentinel_features_df[f'sntnl_mean_baei_{r}m'] = sentinel_features_df[f'sntnl_baei_{r}m'].parallel_apply(
        lambda baei: np.nanmean(baei)
    )
    sentinel_features_df[f'sntnl_std_baei_{r}m'] = sentinel_features_df[f'sntnl_baei_{r}m'].parallel_apply(
        lambda baei: np.nanstd(baei)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_baei_{r}m', f'sntnl_std_baei_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_baei_{r}m'])


    # gCI = NIR / Green - 1
    sentinel_features_df[f'sntnl_gci_{r}m'] = sentinel_features_df[f'sntnl_buffer_{r}m_selection'].parallel_apply(
        lambda patch: ((patch.sel(band=4)/patch.sel(band=6)) - 1)
    )
    sentinel_features_df[f'sntnl_mean_gci_{r}m'] = sentinel_features_df[f'sntnl_gci_{r}m'].parallel_apply(
        lambda gci: np.nanmean(gci)
    )
    sentinel_features_df[f'sntnl_std_gci_{r}m'] = sentinel_features_df[f'sntnl_gci_{r}m'].parallel_apply(
        lambda gci: np.nanstd(gci)
    )
    sentinel_focal_radius_ft.extend([f'sntnl_mean_gci_{r}m', f'sntnl_std_gci_{r}m'])
    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_gci_{r}m'])

    sentinel_features_df = sentinel_features_df.drop(columns=[f'sntnl_buffer_{r}m_selection'])

display(sentinel_features_df[sentinel_focal_radius_ft].head())

# Save data
sentinel_features_df[sentinel_focal_radius_ft].to_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers.parquet')

# Update column groups
column_groups = json.loads(open("../data/column_groups.json").read())

column_groups['sentinel2_focal_buffer_features'] = sentinel_focal_radius_ft

with open('../data/column_groups.json', 'w') as f:
    f.write(json.dumps(column_groups, indent=4))

"""
https://www.uber.com/en-BR/blog/deepeta-how-uber-predicts-arrival-times/
    https://arxiv.org/pdf/2206.02127
"""
