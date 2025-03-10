
# TODO
# -----------------------------------------------------------------------------
# ! No editar hasta tener un buen modelo. Editar de aca hacia arriba
# -----------------------------------------------------------------------------
import sys
sys.path.append('..')

from baseline.utilities import *
import yaml

''' SUBMISSION '''
# * Reading the coordinates for the submission
feature_list = yaml.safe_load(open('./data/columns.yml', 'r'))['features'] + ['Latitude', 'Longitude', 'UHI Index']

test_file = pd.read_parquet('./data/processed/submission/submission_data.parquet')[feature_list]
# test_file = test_file[[col for col in test_file.columns if 'diff_wind_influence_' not in col]]
# for col in test_file:
#     print(col)

train_cols = pd.read_parquet('./data/processed/train/train_data.parquet').columns
submission_cols = test_file.columns
print(set.difference(set(train_cols), set(submission_cols)))
print(set.difference(set(submission_cols), set(train_cols)))

print(f"{test_file.shape=}")
print(f"{len(train_cols)=}")

print(f"{test_file.shape}=")
display(test_file.head())
print(test_file.describe())

selected_features = json.loads(open("selected_features.json", "r").read())['selected_features']

# Scale the training and test data using standardscaler
sc = joblib.load('./models/scaler.pkl')

transformed_submission_data = sc.transform(test_file.drop(columns=['Latitude', 'Longitude', 'UHI Index']))  # [selected_features]
transformed_submission_data = (
    pd.DataFrame(transformed_submission_data, 
                 columns=[col for col in test_file.columns if col not in ['Latitude', 'Longitude', 'UHI Index']]
    )
    [selected_features]
)

# * Load Model
model = joblib.load('./models/random_forest_model.pkl')
try:
    print(f"{model.oob_score_=}")
except:
    pass

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
submission_df.to_csv("../submissions/RF_0,9618_CV10_13FT_0,20rfecv_all.csv", index=False)