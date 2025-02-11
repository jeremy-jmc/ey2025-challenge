import sys
sys.path.append('../..')

from baseline.utilities import *

from shapely.geometry import MultiPolygon, Polygon
from shapely.wkt import loads

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=8)


lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)


df = pd.read_csv('https://data.cityofnewyork.us/api/views/5zhs-2jue/rows.csv?date=20250210&accessType=DOWNLOAD')
display(df)

df['the_geom'] = df['the_geom'].parallel_apply(loads)


gdf = gpd.GeoDataFrame(df, geometry='the_geom')
# set crs
gdf.crs = 'EPSG:4326'
print(gdf.shape)

display(gdf)

# gdf.dtypes

# Filter the data to only include the polygons that are within the bounding box
gdf = gdf.cx[lower_left[1]:upper_right[1], lower_left[0]:upper_right[0]]
print(gdf.shape)
