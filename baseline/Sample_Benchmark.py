'''
Welcome to the EY Open Science AI & Data Challenge 2025!
The objective of this challenge is to build a machine learning model to
predict urban heat island (UHI) hotspots in a city. By the end of the
challenge, you will have developed a regression model capable of predicting
the intensity of the UHI effect.

Participants will be given ground-level air temperature data in an index
format, which was collected on 24th July 2021 on traverse points in the
Bronx and Manhattan regions of New York city. This dataset constitutes
traverse points (latitude and longitude) and their corresponding UHI
(Urban Heat Island) index values. Participants will use this dataset to
build a regression model to predict UHI index values for a given set of
locations. It is important to understand that the UHI Index at any given
location is indicative of the relative temperature elevation at that
specific point compared to the city's average temperature.

This challenge is designed for participants with varying skill levels in
data science and programming, offering a great opportunity to apply your
knowledge and enhance your capabilities in the field.
'''

'''
TIP 1:
Participants might explore other combinations of bands from the Sentinel-2
and from other satellite datasets as well. For example, you can use mathematical
combinations of bands to generate various indices </a> which can then be used as
features in your model. These bands or indices may provide insights into surface
characteristics, vegetation, or built-up areas that could influence UHI patterns.
'''

'''
TIP 2:
Rather than extracting the bands for a single day coincident with the ground-based
data collection, participants might explore other options to improve data quality.
For example, one could select a different single date with minimal or no cloud
cover or generate a median mosaic using several scenes within a time series.
See the Sentinel-2 sample notebook for examples.
'''

'''
TIP 3:
Instead of a single point data extraction, participants might explore the
approach of creating a focal buffer around the locations (e.g., 50 m, 100 m,
150 m etc). For example, if the specified distance was 50 m and the specified
band was “Band 2”, then the value of the output pixels from this analysis
would reflect the average values in band 2 within 50 meters of the specific
location. This approach might help reduction in error associated with spatial
autocorrelation. In this demonstration notebook, we are extracting the band
data for each of the locations without creating a buffer zone.
'''

'''
TIP 4:
There are many data preprocessing methods available, which might help to
improve the model performance. Participants should explore various
suitable preprocessing methods as well as different machine learning
algorithms to build a robust model.
'''

from utilities import *

SEED = 42
TIFF_PATH = './S2_sample.tiff' # './S2_sample_5res.tiff'

# Load the training data from csv file and display the first few rows to inspect the data
ground_df = pd.read_csv("Training_data_uhi_index.csv")
display(ground_df.head())

# ground_df['datetime'].value_counts()
display(
    ground_df.groupby(['Longitude', 'Latitude']).agg({'datetime': 'nunique'})
    .sort_values('datetime', ascending=False)
)
# .reset_index(name='counts')

# `lower_left` and `upper_right` variables of the "Sentinel2_GeoTIFF" notebook
display(ground_df[['Longitude', 'Latitude']].describe())


# -----------------------------------------------------------------------------
# Feature Engineering: Obtain features from Sentinel2 TIFF
# -----------------------------------------------------------------------------

# try_data = map_satellite_data("S2_sample.tiff", ground_df.head(5).copy())

# Mapping satellite data with training data.
final_data = map_satellite_data(TIFF_PATH, 'Training_data_uhi_index.csv')

display(final_data.head())

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
# See the Sentinel-2 sample notebook for more information about the NDVI index
final_data['NDVI'] = (final_data['B08'] - final_data['B04']) / (final_data['B08'] + final_data['B04'])
final_data['NDVI'] = final_data['NDVI'].replace([np.inf, -np.inf], np.nan) 

final_data['gNDBI'] = (final_data['B08'] - final_data['B03']) / (final_data['B08'] + final_data['B03'])
final_data['gNDBI'] = final_data['gNDBI'].replace([np.inf, -np.inf], np.nan) 

# * Joining the predictor variables and response variables
# Combining ground data and final data into a single dataset.
uhi_data = combine_two_datasets(ground_df,final_data)
display(uhi_data.head())

# -----------------------------------------------------------------------------
# Feature Engineering: Generate buffers around the coordinates and extract features
# -----------------------------------------------------------------------------

radius_list = [50, 150, 100, 200, 250]
bbox_dataset = compute_geographic_bounding_boxes(uhi_data, radius_list)

for r in radius_list:
    bbox_dataset[f'buffer_{r}m_selection'] = bbox_dataset[f'buffer_{r}m_bbox_4326'].progress_apply(
        lambda bbox: get_bbox_selection(TIFF_PATH, bbox)
    )

buffer_radius_features = []
for r in radius_list:
    """
    According to the nomenclature of Sentinel2_GeoTIFF.ipynb, the bands are:
        dst.write(data_slice.B01, 1)
        dst.write(data_slice.B04, 2)
        dst.write(data_slice.B06, 3) 
        dst.write(data_slice.B08, 4)
    """
    bbox_dataset[f'ndvi_buffer_{r}m'] = bbox_dataset[f'buffer_{r}m_selection'].progress_apply(
        lambda bbox: (bbox.sel(band=4) - bbox.sel(band=2))/(bbox.sel(band=4) + bbox.sel(band=2))
        # get_ndvi(bbox)
    )

    bbox_dataset[f'ndvi_buffer_{r}m_mean'] = bbox_dataset[f'ndvi_buffer_{r}m'].progress_apply(
        lambda ndvi: np.nanmean(ndvi)
    )

    bbox_dataset[f'vegetation_ratio_ndvi_{r}m'] = bbox_dataset[f'ndvi_buffer_{r}m'].progress_apply(
        lambda ndvi: get_vegetation_ratio(ndvi)
    )

    buffer_radius_features.extend([f'ndvi_buffer_{r}m_mean', f'vegetation_ratio_ndvi_{r}m'])

display(bbox_dataset[buffer_radius_features].head())

uhi_data = combine_two_datasets(uhi_data, bbox_dataset[buffer_radius_features])

all_features = uhi_data.copy()

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = ['B01','B06','NDVI','UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'B11', 'B12', 'gNDBI'] + buffer_radius_features

for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')

# Resetting the index of the dataset
uhi_data = uhi_data.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Model building and Data preparation/splitting
# -----------------------------------------------------------------------------

# Retaining only the columns for B01, B06, NDVI, and UHI Index in the dataset.
uhi_data = uhi_data[['B01', 'B06', 'B8A', 'NDVI', 'UHI Index'] + buffer_radius_features] # , 'B02', 'B03', 'B04', 'B05', 'B07', 'B08',  'B11', 'B12', 'gNDBI'
# [b for b in buffer_radius_features if 'ndvi_buffer' not in b]
display(uhi_data.head())

# Split the data into features (X) and target (y), and then into training and testing sets
X = uhi_data.drop(columns=['UHI Index']).values
y = uhi_data ['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

print(f"{X_train.shape}")

# Scale the training and test data using standardscaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------------------------------------------------------
# Model training
# -----------------------------------------------------------------------------

# * Train the Random Forest model on the training data
model = RandomForestRegressor(n_estimators=100, random_state=SEED)
# model = lgb.LGBMRegressor(n_estimators=100, boosting_type='rf', random_state=SEED, bagging_freq=1, bagging_fraction=0.8)
model.fit(X_train, y_train)

# * Make predictions on the training data
insample_predictions = model.predict(X_train)

# Calculate R-squared score for in-sample predictions
Y_train = y_train.tolist()
print(f"{r2_score(Y_train, insample_predictions)=}")

# * Make predictions on the test data
outsample_predictions = model.predict(X_test)

# Calculate R-squared score for out-sample predictions
Y_test = y_test.tolist()
print(f"{r2_score(Y_test, outsample_predictions)=}")

# * K-Fold cross-validation
print()
for fold in [3, 5]: # , 10
    kf = KFold(n_splits=fold, shuffle=True, random_state=SEED)
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

    print(f"{fold} - R² Scores: {r2_scores}")
    print(f"{fold} - Mean R² Score: {np.mean(r2_scores):.4f}")
    print(f"{fold} - Standard Deviation of R² Scores: {np.std(r2_scores):.4f}")
    print()


# -----------------------------------------------------------------------------
# ! No editar hasta tener un buen modelo. Editar de aca hacia arriba
# -----------------------------------------------------------------------------

''' SUBMISSION '''
#Reading the coordinates for the submission
test_file = pd.read_csv('Submission_template.csv')
test_file.head()

# Mapping satellite data for submission.
val_data = map_satellite_data(TIFF_PATH, 'Submission_template.csv')

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
val_data['NDVI'] = (val_data['B08'] - val_data['B04']) / (val_data['B08'] + val_data['B04'])
val_data['NDVI'] = val_data['NDVI'].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN

# Extracting specific columns (B01, B06, and NDVI) from the validation dataset
submission_val_data = val_data.loc[:,['B01','B06','NDVI']]

# Feature Scaling 
submission_val_data = submission_val_data.values
transformed_submission_data = sc.transform(submission_val_data)

# Making predictions
final_predictions = model.predict(transformed_submission_data)
final_prediction_series = pd.Series(final_predictions)

# Combining the results into dataframe
submission_df = pd.DataFrame({
    'Longitude': test_file['Longitude'].values, 
    'Latitude':test_file['Latitude'].values, 
    'UHI Index':final_prediction_series.values
})

# Dumping the predictions into a csv file.
submission_df.to_csv("submission.csv",index = False)
