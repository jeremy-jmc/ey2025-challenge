import sys
sys.path.append('../..')

from baseline.utilities import *

from sklearn.cluster import MeanShift, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

ground_df = pd.read_csv("../../baseline/Training_data_uhi_index.csv")
new_ground_df = pd.read_csv('../../baseline/Training_data_uhi_index_2025-02-18.csv')

# Compare the content of the two dataframes
print(f"{ground_df.equals(new_ground_df)=}")

display(ground_df)
ground_df.columns = ground_df.columns.str.lower()

# -----------------------------------------------------------------------------
# Perform Dispersion-based Clustering
# -----------------------------------------------------------------------------

# Good Eps for DBSCAN -> 0.0005

X = ground_df[['latitude', 'longitude']].values
clustering = DBSCAN(eps=0.00035, min_samples=2).fit(X)    # MeanShift().fit(X)
print(clustering.labels_)

print(np.unique(clustering.labels_))

ground_df['cluster'] = clustering.labels_

ground_df[['cluster']].to_parquet("../data/processed/train/cluster.parquet")

# Silhouette Score
print(f"{silhouette_score(X, clustering.labels_)=}")

# -----------------------------------------------------------------------------
# Compute the associativy along the clusters
# -----------------------------------------------------------------------------

# Standard Deviation
associativity = ground_df.groupby('cluster')['uhi index'].std()
display(associativity)

print(f"{associativity.mean()=}")
print(f"{associativity.std()=}")

# -----------------------------------------------------------------------------
# To GeoDataFrame
# -----------------------------------------------------------------------------

gdf = gpd.GeoDataFrame(
    ground_df, geometry=gpd.points_from_xy(ground_df['longitude'], ground_df['latitude']),
    crs='EPSG:4326' # Latitude-Longitude -> https://spatialreference.org/ref/epsg/4326/
)

gdf.to_file("../Training_data_uhi_index_clustered.json", driver='GeoJSON')

