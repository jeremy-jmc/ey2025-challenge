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


# -----------------------------------------------------------------------------
# Feature Engineering: Explore the NY MESONET Weather data
# -----------------------------------------------------------------------------

ny_bronx_point = (40.87248, -73.89352)
ny_manhattan_point = (40.76754, -73.96449)

ny_mesonet_bronx_df = pd.read_excel('../../baseline/NY_Mesonet_Weather.xlsx', sheet_name='Bronx')
ny_mesonet_manhattan_df = pd.read_excel('../../baseline/NY_Mesonet_Weather.xlsx', sheet_name='Manhattan')

ny_mesonet_bronx_df['Date / Time'] = pd.to_datetime(ny_mesonet_bronx_df['Date / Time'])
ny_mesonet_manhattan_df['Date / Time'] = pd.to_datetime(ny_mesonet_manhattan_df['Date / Time'])

ny_mesonet_bronx_df = ny_mesonet_bronx_df.sort_values('Date / Time')
ny_mesonet_manhattan_df = ny_mesonet_manhattan_df.sort_values('Date / Time')

print(ny_mesonet_bronx_df.dtypes)
print(ny_mesonet_manhattan_df.dtypes)

# Filter from 3pm to 4pm
ny_mesonet_bronx_df = ny_mesonet_bronx_df[
    (ny_mesonet_bronx_df['Date / Time'].dt.hour == 15) |
    ((ny_mesonet_bronx_df['Date / Time'].dt.hour == 16) & (ny_mesonet_bronx_df['Date / Time'].dt.minute == 0))
].reset_index(drop=True)
display(ny_mesonet_bronx_df)

ny_mesonet_manhattan_df = ny_mesonet_manhattan_df[
    (ny_mesonet_manhattan_df['Date / Time'].dt.hour == 15) |
    ((ny_mesonet_manhattan_df['Date / Time'].dt.hour == 16) & (ny_mesonet_manhattan_df['Date / Time'].dt.minute == 0))
].reset_index(drop=True)
display(ny_mesonet_manhattan_df)

ny_mesonet_features = ground_df[['Latitude', 'Longitude']].copy()
ny_mesonet_features.columns = ny_mesonet_features.columns.str.lower()

ny_mesonet_features['distance_bronx'] = ny_mesonet_features.parallel_apply(
    lambda x: distance_meters(x, ny_bronx_point),
    axis=1
)
ny_mesonet_features['distance_manhattan'] = ny_mesonet_features.parallel_apply(
    lambda x: distance_meters(x, ny_manhattan_point),
    axis=1
)

ny_mesonet_features['ratio_dist_bronx_manhattan'] = (
    ny_mesonet_features['distance_bronx'] / (
        ny_mesonet_features['distance_manhattan'] + ny_mesonet_features['distance_bronx']
    )
)
ny_mesonet_features['ratio_dist_manhattan_bronx'] = (
    ny_mesonet_features['distance_manhattan'] / (
        ny_mesonet_features['distance_manhattan'] + ny_mesonet_features['distance_bronx']
    )
)


ny_mesonet_features['bearing_bronx'] = ny_mesonet_features[['latitude', 'longitude']].parallel_apply(
    lambda x: compute_bearing(ny_bronx_point, (x['latitude'], x['longitude'])),
    axis=1
)

ny_mesonet_features['bearing_manhattan'] = ny_mesonet_features[['latitude', 'longitude']].parallel_apply(
    lambda x: compute_bearing(ny_manhattan_point, (x['latitude'], x['longitude'])),
    axis=1
)

ny_mesonet_bm_df = pd.concat([ny_mesonet_bronx_df.assign(location='bronx'), ny_mesonet_manhattan_df.assign(location='manhattan')], axis=0)

ny_mesonet_bm_wind_dir_pivot = ny_mesonet_bm_df.pivot(
    index="location", 
    columns="Date / Time", 
    values="Wind Direction [degrees]"
)
ny_mesonet_bm_wind_dir_pivot.columns = [f"Wind Direction [degrees] {col}" for col in ny_mesonet_bm_wind_dir_pivot.columns]
ny_mesonet_bm_wind_dir_pivot = ny_mesonet_bm_wind_dir_pivot.reset_index(drop=False)

nymesonet_bm_avg_wind_speed_pivot = ny_mesonet_bm_df.pivot(
    index="location",
    columns="Date / Time",
    values="Avg Wind Speed [m/s]"
)
nymesonet_bm_avg_wind_speed_pivot.columns = [f"Avg Wind Speed [m/s] {col}" for col in nymesonet_bm_avg_wind_speed_pivot.columns]
nymesonet_bm_avg_wind_speed_pivot = nymesonet_bm_avg_wind_speed_pivot.reset_index(drop=False)

ny_mesonet_bm_df_pivot = pd.merge(
    ny_mesonet_bm_wind_dir_pivot, nymesonet_bm_avg_wind_speed_pivot, on="location"
)

display(ny_mesonet_bm_df_pivot) # .to_dict(orient='tight')

ny_mesonet_bm_dict = {
    "location": {
        row["location"]: {
            col : row[col]  # f"Wind Direction {col.split()[-1]}"
            for col in ny_mesonet_bm_df_pivot.columns if col != "location"
        }
        for _, row in ny_mesonet_bm_df_pivot.iterrows()
    }
}
print(json.dumps(ny_mesonet_bm_dict, indent=2))

for loc in ny_mesonet_bm_dict['location']:
    print(loc)
    for k, v in ny_mesonet_bm_dict['location'][loc].items():
        print(f"{k}: {v}")
        if k.startswith('Wind Direction'):    
            ny_mesonet_features[f"Wind Influence {k.split()[-1]} {loc}"] = ny_mesonet_features[f'bearing_{loc}'].parallel_apply(
                lambda x: np.cos(np.radians(v - x))
            )
        # elif k.startswith('Avg Wind Speed'):
        #     ny_mesonet_features[f"Weighted Wind Influence {k.split()[-1]} {loc}"] = (
        #         ny_mesonet_features[f"Wind Influence {k.split()[-1]} {loc}"] * v
        #     )

ny_mesonet_features.columns = [
    col.replace(' ', '_').lower() for col in ny_mesonet_features.columns
]

display(ny_mesonet_features)

ny_mesonet_features = ny_mesonet_features.drop(
    columns=['latitude', 'longitude', 'distance_bronx', 'distance_manhattan', 'ratio_dist_bronx_manhattan', 'ratio_dist_manhattan_bronx']
    # 'latitude', 'longitude', 
)

# create a list of hours strings from 15 to 16 every 5 minutes
hours = [f"{h:02d}:{m:02d}:00" for h in range(15, 16) for m in range(0, 60, 5)] + ['16:00:00']

for place in ['bronx', 'manhattan']:
    for idx in range(1, len(hours)):
        ny_mesonet_features[f"diff_wind_influence_{hours[idx]}_{place}"] = ny_mesonet_features[f"wind_influence_{hours[idx]}_{place}"] - ny_mesonet_features[f"wind_influence_{hours[idx-1]}_{place}"]
        ny_mesonet_features[f"pct_change_wind_influence_{hours[idx]}_{place}"] = ny_mesonet_features[f"wind_influence_{hours[idx]}_{place}"] / ny_mesonet_features[f"wind_influence_{hours[idx-1]}_{place}"] - 1


for place in ['bronx', 'manhattan']:
    for idx in range(1, len(hours)):
        ny_mesonet_features = ny_mesonet_features.drop(columns=[f"diff_wind_influence_{hours[idx]}_{place}"])

display(ny_mesonet_features)


# Save data
ny_mesonet_features.to_parquet(f'../data/processed/{MODE}/ny_mesonet_features.parquet')

# Update column groups
column_groups = json.loads(open("../data/column_groups.json").read())

column_groups['ny_mesonet_features'] = ny_mesonet_features.columns.tolist()

with open('../data/column_groups.json', 'w') as f:
    f.write(json.dumps(column_groups, indent=4))
