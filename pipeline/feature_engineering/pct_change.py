import sys
sys.path.append('../..')

from baseline.utilities import *

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=8)

SENTINEL_TIFF_PATH = '../../baseline/S2_sample.tiff' # './S2_sample_5res.tiff'
LANDSAT_TIFF_PATH = '../../baseline/Landsat_LST.tiff'
MODE = 'submission'  # 'submission' 'train'

assert MODE in ['submission', 'train'], "MODE should be either 'train' or 'submission'"

radius_list = json.loads(open("../data/radius_list.json", "r").read())['radius_list']

sentinel_data = (
    # pd.read_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers_bands.parquet')
    pd.read_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers.parquet')
)
sentinel_data_filtered = sentinel_data[[c for c in sentinel_data.columns if 'mean' in c]]

display(sentinel_data_filtered)

consecutive_pairs = [(i, i+100) for i in range(100, 1000, 100)]

sentinel_ratio_pct_df = pd.DataFrame()

for curr, nxt in consecutive_pairs:
    for col in [c for c in sentinel_data_filtered.columns if f'_{curr}m' in c]:
        # sentinel_pct_changes_df[f"{col}_{nxt}m_pct_change"] = sentinel_data_filtered[col] / sentinel_data_filtered[col.replace(f'_{curr}m', f'_{nxt}m')]
        sentinel_ratio_pct_df[f"{col}_{nxt}m_pct_change"] = (
            (sentinel_data_filtered[col] - sentinel_data_filtered[col.replace(f'_{curr}m', f'_{nxt}m')]) / sentinel_data_filtered[col.replace(f'_{curr}m', f'_{nxt}m')]
        ) * 100

        sentinel_ratio_pct_df[f"{col}_{nxt}m_ratio_outside_to_inside"] = (
            sentinel_data_filtered[col] / sentinel_data_filtered[col.replace(f'_{curr}m', f'_{nxt}m')]
        )

sentinel_ratio_pct_df.to_parquet(f'../data/processed/{MODE}/sentinel2_focal_buffers_pct_change_100m.parquet')
print("File saved successfully!")