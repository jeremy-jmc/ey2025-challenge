
# !python3 -m pip install geopy

import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import matplotlib.ticker as ticker
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.metrics import silhouette_score
from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display
from pyproj import Transformer
import random
import stackstac
import pystac_client
import planetary_computer 
from odc.stac import stac_load

tqdm.pandas()

pd.set_option('display.max_colwidth', 100)
pd.options.display.float_format = '{:.4f}'.format

RANDOM_SEED = 42
TIME_WINDOW = "2021-06-01/2021-09-01"

# Define the pixel resolution for the final product
# Define the scale according to our selected crs, so we will use degrees
RESOLUTION = 0.25  # meters per pixel 
SCALE = RESOLUTION / 111320.0 # degrees per pixel for crs=4326 


df = (
    pd.read_csv('./baseline/Training_data_uhi_index.csv')
    # .sort_values(by=['datetime'])
)
df.columns = [col.replace(" ", '_') for col in df.columns.str.lower()]

# -----------------------------------------------------------------------------
# Perform Dispersion-based Clustering
# -----------------------------------------------------------------------------

# Good Eps for DBSCAN -> 0.0005

X = df[['latitude', 'longitude']].values
clustering = DBSCAN(eps=0.0005, min_samples=2).fit(X)    # MeanShift().fit(X)
print(clustering.labels_)

print(np.unique(clustering.labels_))

df['cluster'] = clustering.labels_

# Silhouette Score
print(f"{silhouette_score(X, clustering.labels_)=}")

# -----------------------------------------------------------------------------
# Compute the associativy along the clusters
# -----------------------------------------------------------------------------

# Standard Deviation
associativity = df.groupby('cluster')['uhi_index'].std()
display(associativity)

print(f"{associativity.mean()=}")
print(f"{associativity.std()=}")

# -----------------------------------------------------------------------------
# Nearest Neighbors and Pair Distance
# -----------------------------------------------------------------------------

neighbors = NearestNeighbors(n_neighbors=2, metric='haversine')
data_radians = df[['latitude', 'longitude']].apply(lambda x: np.radians(x))

neighbors.fit(data_radians)

distances, indices = neighbors.kneighbors(data_radians)

df['nearest_index'] = indices[:, 1]
df['nearest_distance'] = distances[:, 1]


def distance_meters(row):
    return geodesic(
        (row['latitude'], row['longitude']),
        (df.iloc[row['nearest_index']]['latitude'], df.iloc[row['nearest_index']]['longitude'])
    ).meters

df['distance_meters'] = df.progress_apply(distance_meters, axis=1)

"""
https://www.youtube.com/watch?v=vqPskrNH-Hg
"""

# -----------------------------------------------------------------------------
# 10m, 50m, 100m, 150m buffers
# -----------------------------------------------------------------------------

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs='EPSG:4326' # Latitude-Longitude -> https://spatialreference.org/ref/epsg/4326/
)

gdf.to_file('./baseline/uhi_vars.geojson', driver='GeoJSON')

def buffer_meters(row, meters):
    return row['geometry'].buffer(meters)

def get_square_from_polygon(polygon):
    # get square bounding box from polygon
    x, y = polygon.exterior.coords.xy

    return (
        (min(x), min(y)),
        (max(x), min(y)),
        (max(x), max(y)),
        (min(x), max(y))
    )
    

transformer = Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)

buffer_radius = [250]

for meters in buffer_radius:    # 10, 50, 100, 
    # EPSG:3395 is a projected CRS that works better with meters
    gdf = gdf.to_crs(epsg=3395) 

    # Calculate in EPSG:3395
    gdf[f'buffer_{meters}m'] = gdf.apply(lambda x: buffer_meters(x, meters), axis=1)
    gdf[f'buffer_{meters}m_bbox'] = gdf[f'buffer_{meters}m'].apply(lambda x: x.bounds)      # get_bbox_from_polygon(x)

    # Transform to EPSG:4326
    gdf[f'buffer_{meters}m_bbox_4326'] = gdf[f'buffer_{meters}m_bbox'].apply(lambda bbox: [
        transformer.transform(bbox[0], bbox[1]),  # minx, miny
        transformer.transform(bbox[2], bbox[3])   # maxx, maxy
    ])
    
    gdf[f'linestring_buffer_{meters}m'] = (
        gdf[f'buffer_{meters}m_bbox_4326'].apply(lambda x: LineString(x))
    )
        
    # Only update the 'geometry' column
    gdf = gdf.to_crs(epsg=4326)

    buffers_gdf = gpd.GeoDataFrame(    
        gdf[['longitude', 'latitude', 'datetime', 'uhi_index', f'linestring_buffer_{meters}m']],
        geometry=f'linestring_buffer_{meters}m',
        crs='EPSG:4326'
    ).to_file(f'./baseline/buffers_{meters}.geojson', driver='GeoJSON')


display(gdf.head())
# gdf.iloc[0]['buffer_150m']
# dir(gdf.iloc[0]['buffer_150m'])
# print(gdf[['buffer_150m_bbox_4326']])
# print(gdf.iloc[0].to_dict())


# -----------------------------------------------------------------------------
# Sentinel2 API request for a random point in the dataset
# -----------------------------------------------------------------------------

random_res = random.choice(buffer_radius)

stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

random_buffer = gdf.sample(1).iloc[0][f'buffer_{random_res}m_bbox_4326']

# bounds = (min_lon, min_lat, max_lon, max_lat)
random_bounds = (*random_buffer[0], *random_buffer[1])

search = stac.search(
    bbox=random_bounds, 
    datetime=TIME_WINDOW,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)

items = list(search.items())

signed_items = [planetary_computer.sign(item).to_dict() for item in items]

data = stac_load(
    items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=SCALE, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=random_bounds
)
"""
REFERENCES:
    https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/

"""
print(data)

# Plot the multiple satellital images
plot_data = data[["B04","B03","B02"]].to_array()
print(type(plot_data))

fig = plot_data.plot.imshow(col='time', col_wrap=4, robust=True, vmin=0, vmax=2500)

for ax in fig.axes.flat:
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

plt.show()


# Compute the median composite along the time-range and plot it
median = data.median(dim="time").compute()

fig, ax = plt.subplots(figsize=(6,6))
median[["B04", "B03", "B02"]].to_array().plot.imshow(robust=True, ax=ax, vmin=0, vmax=2500)
ax.set_title("RGB Median Composite")
ax.axis('off')
plt.show()


# -----------------------------------------------------------------------------
# Calculate indexes
# -----------------------------------------------------------------------------

# Calculate NDVI for the median mosaic
ndvi_median = (median['B08'] - median['B04'])/(median['B08'] + median['B04'])

fig, ax = plt.subplots(figsize=(7,6))
ndvi_median.plot.imshow(vmin=0.0, vmax=1.0, cmap="RdYlGn")
plt.title("Median NDVI")
plt.axis('off')
plt.show()

# Calculate the proportion of vegetation pixels
veg_threshold = 0.4
vegetation_pixels = np.sum(ndvi_median > veg_threshold).item()
total_pixels = np.sum(~np.isnan(ndvi_median)).item()

veg_proportion = vegetation_pixels / total_pixels if total_pixels > 0 else 0

print(f"{veg_proportion=}")
print(f"{veg_proportion:.2%} of pixels have an NDVI greater than {veg_threshold}")


"""
TODO:
    READ:
        https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/#remote-sensing-indices
REFERENCES:
    https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/

"""