
# TODO
# -----------------------------------------------------------------------------
# ! No editar hasta tener un buen modelo. Editar de aca hacia arriba
# -----------------------------------------------------------------------------
import sys
sys.path.append('..')

from baseline.utilities import *

''' SUBMISSION '''
#Reading the coordinates for the submission
test_file = pd.read_csv('../baseline/Submission_template.csv')
print(f"{test_file.shape}=")
display(test_file.head())
print(test_file.describe())


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

# * Load Model
model = joblib.load('./models/random_forest_model.pkl')
print(model.oob_score_)

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
