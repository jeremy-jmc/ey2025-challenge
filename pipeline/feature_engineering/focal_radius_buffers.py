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
# Data Visualization: Save dataset as GeoDataFrame
# -----------------------------------------------------------------------------

gdf = gpd.GeoDataFrame(
    ground_df, geometry=gpd.points_from_xy(ground_df['Longitude'], ground_df['Latitude']),
    crs='EPSG:4326' # Latitude-Longitude -> https://spatialreference.org/ref/epsg/4326/
)
gdf.to_file(f"../data/processed/{MODE}/ground_dataset.json", driver='GeoJSON')
print(gdf.shape)

# -----------------------------------------------------------------------------
# Data Preprocessing: Generate geographic bounding boxes around the coordinates
# -----------------------------------------------------------------------------

# radius_list = [50, 100, 150, 200, 250]        # LB: 0.9306
radius_list = json.loads(open('../data/radius_list.json', 'r').read())['radius_list']
bbox_dataset = compute_geographic_bounding_boxes(ground_df[['Longitude', 'Latitude']], radius_list)
print(bbox_dataset.columns)

with open('../data/radius_list.json', 'w') as f:
    f.write(json.dumps({"radius_list": radius_list}, indent=4))

bbox_dataset.to_parquet(f'../data/processed/{MODE}/bbox_dataset.parquet')

# with open('../data/column_groups.json', 'w') as f:
#     f.write(json.dumps({}, indent=4))
