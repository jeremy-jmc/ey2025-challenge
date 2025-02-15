import sys
sys.path.append('../..')

from baseline.utilities import *

from shapely.geometry import MultiPolygon, Polygon
from shapely.wkt import loads

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=8)


lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)


date_request = "20210624"
df = pd.read_csv(f"https://data.cityofnewyork.us/api/views/5zhs-2jue/rows.csv?date={date_request}&accessType=DOWNLOAD")
display(df)

df['the_geom'] = df['the_geom'].parallel_apply(loads)


gdf = gpd.GeoDataFrame(df.rename(columns={'the_geom': 'geom'}), geometry='geom')
# set crs
gdf.crs = 'EPSG:4326'
print(gdf.shape)

display(gdf)

# gdf.dtypes

# Filter the data to only include the polygons that are within the bounding box
gdf = gdf.cx[lower_left[1]:upper_right[1], lower_left[0]:upper_right[0]]
print(gdf.shape)

print(gdf.isna().sum())

display(gdf[gdf['NAME'].notna()])

gdf = gdf[gdf['BIN'].astype(str).str.startswith('1')]

gdf['LSTMODDATE'] = pd.to_datetime(gdf['LSTMODDATE'])

print(gdf['LSTMODDATE'].describe())


gdf.to_file('../data/nyc_opendata.geojson', driver='GeoJSON')


"""
Documentation and data dictionary for the NYC Open Data Building Footprints dataset:
    https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_BuildingFootprints.md

The dates are not according to the challenge's requirements. The dataset is up to date until 2025, but the challenge only capture the data from July 24, 2021
"""
