
# TODO
# -----------------------------------------------------------------------------
# ! No editar hasta tener un buen modelo. Editar de aca hacia arriba
# -----------------------------------------------------------------------------
import sys
sys.path.append('..')

from baseline.utilities import *

''' SUBMISSION '''
# * Reading the coordinates for the submission
test_file = pd.read_parquet('./data/submission_data.parquet')
# for col in test_file:
#     print(col)

train_cols = pd.read_parquet('./data/train_data.parquet').columns
submission_cols = test_file.columns
print(set.difference(set(train_cols), set(submission_cols)))

print(f"{test_file.shape}=")
display(test_file.head())
print(test_file.describe())

selected_features = json.loads(open("selected_features.json", "r").read())['selected_features']

# Scale the training and test data using standardscaler
sc = joblib.load('./models/scaler.pkl')

transformed_submission_data = sc.transform(test_file.drop(columns=['Latitude', 'Longitude', 'UHI Index']))# [selected_features]
transformed_submission_data = (
    pd.DataFrame(transformed_submission_data, 
                 columns=[col for col in test_file.columns if col not in ['Latitude', 'Longitude', 'UHI Index']]
    )
    [selected_features]
)

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
submission_df.to_csv("../submissions/RF_0,9553_CV10_7FT_0,2Test.csv", index=False)
