# !python3 -m pip install fastkml GDAL
import sys
sys.path.append('../..')

from baseline.utilities import *

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=8)

# Read .kml
import geopandas as gpd
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


"""
TODO ValDLaw: 
    Calculate area for each Multipolygon object
    Perform a spatial join between a random point and the building footprints (hint: sjoin)

        Answer the following question creating a new feature as a column: 
            What is the area of buildings within a 250m radius of the point? Extrapolate to multiple radius values
"""
