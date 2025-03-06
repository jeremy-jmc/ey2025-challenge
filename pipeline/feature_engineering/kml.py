# !python3 -m pip install fastkml GDAL
import sys
sys.path.append('../..')

from baseline.utilities import *

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=8)

# Read .kml
import geopandas as gpd
from shapely.ops import unary_union
import fiona
fiona.drvsupport.supported_drivers['libkml'] = 'rw' 
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

kml_path = '../../baseline/Building_Footprint.kml'

# # -----------------------------------------------------------------------------
# # Read KML from base
# # -----------------------------------------------------------------------------
# from osgeo import gdal, ogr

# ds = ogr.Open(kml_path)
# layer = ds.GetLayer()

# features = []
# for feature in layer:
#     attributes = feature.items()
#     geometry = feature.geometry().Clone()
#     attributes["geometry"] = geometry.ExportToWkt()  # Convert to Well-Known Text (WKT)
#     features.append(attributes)

# gdf = (
#     gpd.GeoDataFrame(features, geometry=gpd.GeoSeries.from_wkt([f["geometry"] for f in features]), crs="EPSG:4326")
#     .dropna(axis=1, how='all')
# )
# print(f"{gdf.shape=}")
# display(gdf)


# -----------------------------------------------------------------------------
# * Read KML with GeoPandas
# -----------------------------------------------------------------------------

bld_footprint = (
    gpd.read_file(kml_path, driver='libkml')
    .replace('', np.nan)
)
print(bld_footprint.columns)

# Drop columns with all NaN values 
bld_footprint = bld_footprint.dropna(axis=1, how='all')
print(f"{bld_footprint.shape=}")
display(bld_footprint)

print(bld_footprint.crs)

bld_footprint['geometry'] = bld_footprint['geometry'].apply(lambda x: unary_union(x))
bld_footprint = bld_footprint.to_crs(epsg=3395)

bld_footprint['area'] = bld_footprint['geometry'].area

bld_footprint = bld_footprint.to_crs(epsg=4326)

bld_footprint.to_file('../data/other/bf.json', driver='GeoJSON')
display(bld_footprint)

# Get bbox from all GeoDataFrame
bbox = bld_footprint.total_bounds


# -----------------------------------------------------------------------------
# * NYC Buildings -> External Dataset
# -----------------------------------------------------------------------------

# Source: https://nycmaps-nyc.hub.arcgis.com/datasets/nyc::building/about
bv = gpd.read_file('../../baseline/BUILDING_view_-5690770882456580009.geojson')

bv = bv.to_crs(epsg=3395)
bv['area'] = bv['geometry'].area
bv = bv.to_crs(epsg=4326)

bv = bv.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)

bv_csv = (
    pd.read_csv('../../baseline/BUILDING_view_7607496916235021567.csv')
    [['DOITT ID', 'Area', 'Length']]
)

bv = bv.join(bv_csv.set_index('DOITT ID'), on='DOITT_ID')

bv.to_file('../data/other/bv.json', driver='GeoJSON')

# https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_BuildingFootprints.md

bv = bv[['CONSTRUCTION_YEAR', 'FEATURE_CODE', 'GROUND_ELEVATION', 'HEIGHT_ROOF', 'LAST_EDITED_DATE', 'LAST_STATUS_TYPE', 'geometry', 'area', 'Area', 'Length']]

bv = bv.loc[bv['CONSTRUCTION_YEAR'] < 2021].reset_index(drop=True)     # ! Check consistency of results

# ? Length variable is the perimeter?
# TODO: Filter all polygons/geometries than intersects with any of the bld_footprint geometries

# -----------------------------------------------------------------------------
# * Point and Radius/Buffer Queries
# -----------------------------------------------------------------------------

train_data = pd.read_csv('../../baseline/Training_data_uhi_index.csv')
train_data.columns = train_data.columns.str.lower()

dataset = train_data[['longitude', 'latitude']]
dataset['geometry'] = gpd.points_from_xy(dataset['longitude'], dataset['latitude'])

# ! TODO: Remove the .sample
geodataset = gpd.GeoDataFrame(dataset.sample(200, ignore_index=True), crs='EPSG:4326')     # , random_state=42

radius_list = [25] + json.loads(open('../data/radius_list.json', 'r').read())['radius_list']

for radius_meter in tqdm(radius_list, total=len(radius_list), desc='Radius Areas'):
    geodataset = geodataset.to_crs(epsg=3395)

    geodataset[f'buffer_{radius_meter}m'] = (
        geodataset.apply(lambda x: buffer_meters(x, radius_meter), axis=1)
        .set_crs(3395)
        .to_crs(epsg=4326)
    )

    # * EY .kml file Building Footprints
    intersecting_squares = gpd.sjoin(bld_footprint, 
                            gpd.GeoDataFrame(geometry=geodataset[f'buffer_{radius_meter}m'], crs=bld_footprint.crs), 
                            predicate="intersects",
                            how='inner'
                            ).drop_duplicates(subset=['index_right', 'geometry'])

    area_sums = intersecting_squares.groupby('index_right')['area'].sum()
    # TODO: mean and std of areas inside the buffer

    geodataset[f"sum_areas_{radius_meter}m"] = geodataset.index.map(area_sums).fillna(0)
    
    # * NYC Buildings
    
    # TODO: Linear combination of area and height of buildings inside the buffer with `bv` variable
    


    geodataset = geodataset.drop(columns=[f'buffer_{radius_meter}m'])
    display(geodataset)

geodataset = geodataset.to_crs(epsg=4326)

display(geodataset)

# print(8756.9 + 2676.0 + 2297.6 + 2323.4 + 8792.1)

"""
TODO ValDLaw: 
    Calculate area for each Multipolygon object
    Perform a spatial join between a random point and the building footprints (hint: sjoin)

        Answer the following question creating a new feature as a column: 
            What is the area of buildings within a 250m radius of the point? Extrapolate to multiple radius values
"""

radius_meter = 50

geodataset = geodataset.to_crs(epsg=3395)

geodataset[f'buffer_{radius_meter}m'] = (
    geodataset.apply(lambda x: buffer_meters(x, radius_meter), axis=1)
    .set_crs(3395)
    .to_crs(epsg=4326)
)

intersecting_buildings = gpd.sjoin(bv[['Area', 'Length', 'geometry', 'GROUND_ELEVATION', 'HEIGHT_ROOF']], 
    gpd.GeoDataFrame(geometry=geodataset[f'buffer_{radius_meter}m'], crs=bv.crs), 
    predicate="intersects",
    how='inner'
    ).drop_duplicates(subset=['index_right', 'geometry'])

display(intersecting_buildings)

geodataset = geodataset.to_crs(epsg=4326)


