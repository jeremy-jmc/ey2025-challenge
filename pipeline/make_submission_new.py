import sys
sys.path.append('..')

from baseline.utilities import *
import yaml

X_rfe = pd.read_parquet('./data/processed/train/X_selected.parquet')
test_file = pd.read_parquet('./data/processed/submission/submission_data.parquet')


# * Load Model
model = joblib.load('./models/stacking.pkl')

sc = joblib.load('./models/scaler.pkl')

# Making predictions
final_predictions = model.predict(sc.transform(test_file[X_rfe.columns]))
final_prediction_series = pd.Series(final_predictions)

# Combining the results into dataframe
submission_df = pd.DataFrame({
    'Longitude': test_file['Longitude'].values, 
    'Latitude':test_file['Latitude'].values, 
    'UHI Index':final_prediction_series.values
})

submission_df.to_csv(f"../submissions/RF_ElasticNet_0.9747_CV10_{len(X_rfe.columns)}FT_0,01_stacking.csv", index=False)

