import sys
sys.path.append('..')

from baseline.utilities import *

train_data = pd.read_parquet('./data/train_data.parquet')
column_dict = json.loads(open('./data/columns.json').read())

buffer_radius_features = column_dict['focal_radius_features']
print(f"{buffer_radius_features=}")

# Retaining only the columns for B01, B06, NDVI, and UHI Index in the dataset.
uhi_data = train_data[['B01', 'B06', 'B8A', 'NDVI', 'UHI Index'] + buffer_radius_features] # , 'B02', 'B03', 'B04', 'B05', 'B07', 'B08',  'B11', 'B12', 'gNDBI'
print(uhi_data.isna().sum())
display(uhi_data.head())


# Split the data into features (X) and target (y), and then into training and testing sets
X = uhi_data.drop(columns=['UHI Index']).values
y = uhi_data ['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

print(f"{X_train.shape}")

# TODO: @ValDLaw23 check if the scaling affects the training
# Scale the training and test data using standardscaler
sc = StandardScaler() # MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------------------------------------------------------
# Feature selection
# -----------------------------------------------------------------------------

# TODO: ValDLaw23 implement automatic RFE feature selection algorithm

# -----------------------------------------------------------------------------
# Model training
# -----------------------------------------------------------------------------

# * Train the Random Forest model on the training data
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=SEED)
# model = lgb.LGBMRegressor(n_estimators=100, boosting_type='rf', random_state=SEED, bagging_freq=1, bagging_fraction=0.8)
model.fit(X_train, y_train)

# * OOB Score
print(f"{model.oob_score_=}")

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
for fold in [3, 5, 10]:
    kf = KFold(n_splits=fold, shuffle=True, random_state=SEED)
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

    print(f"{fold} - R² Scores: {r2_scores}")
    print(f"{fold} - Mean R² Score: {np.mean(r2_scores):.4f}")
    print(f"{fold} - Standard Deviation of R² Scores: {np.std(r2_scores):.4f}")
    print()


# -----------------------------------------------------------------------------
# Model Feature Importances
# -----------------------------------------------------------------------------

# TODO: Research how to plot FE using the data proportined by the model and then, with SHAP Python library

