import sys
sys.path.append('..')

from baseline.utilities import *

MODE = 'train_model'  # 'train_model', 'feature_selection'

train_data = pd.read_parquet('./data/train_data.parquet')
try:
    train_data = train_data.drop(columns=['latitude', 'longitude'])
except:
    pass

for col in train_data.columns:
    print(col)

column_dict = json.loads(open('./data/columns.json').read())

buffer_radius_features = [col for col in column_dict['focal_radius_features']]  #  if 'diff_wind_influence_' not in col
print(f"({len(buffer_radius_features)}) -> {buffer_radius_features=}")

# -----------------------------------------------------------------------------
# Feature selection
# -----------------------------------------------------------------------------

# Retaining only the columns for B01, B06, NDVI, and UHI Index in the dataset.
print(f"{train_data.shape=}")
uhi_data = train_data[['B01', 'B06', 'B8A', 'NDVI', 'UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08',  'B11', 'B12', 'gNDBI'] + buffer_radius_features]
# .drop(columns=['Longitude', 'Latitude', 'datetime'])
print(f"{uhi_data.shape=}")

print(uhi_data.isna().sum())
# display(uhi_data.head())

X = uhi_data.drop(columns=['UHI Index'])    
y = uhi_data ['UHI Index'].values

# corr_matrix = X.corr()
# plt.figure()
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show()

if MODE == 'feature_selection':
    rfecv = RFECV(estimator=DecisionTreeRegressor(random_state=SEED), cv=KFold(n_splits=5, shuffle=True, random_state=SEED), scoring='r2', n_jobs=-1)

    X_selected = rfecv.fit_transform(X, y)
    # TODO: ValDLaw23 research the computational viability of BorutaRandomForest or another method to select features
    """
    https://www.kaggle.com/code/residentmario/automated-feature-selection-with-boruta
    https://amueller.github.io/aml/05-advanced-topics/12-feature-selection.html
    """

    print(f"{rfecv.ranking_=}")
    print(f"{rfecv.cv_results_.keys()=}")

    plt.figure()
    plt.plot(rfecv.cv_results_['n_features'][2:], rfecv.cv_results_['mean_test_score'][2:], color='blue')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validation Score')
    plt.title('RFECV - Number of Features vs. Cross-Validation Score')
    plt.show()

    # Print selected features
    selected_features = X.columns[rfecv.support_]
    print(f"Selected features ({len(selected_features)}): {list(selected_features)}")
    with open("selected_features.json", "w") as f:
        f.write(json.dumps({
            "selected_features" : list(selected_features)
        }))

    mask_selected_cols = rfecv.support_
elif MODE == 'train_model':
    selected_features = json.loads(open("selected_features.json", "r").read())['selected_features']
    mask_selected_cols = [True if col in selected_features else False for col in X.columns]

print(f"{mask_selected_cols=}")

# -----------------------------------------------------------------------------
# Train/Test Split
# -----------------------------------------------------------------------------

X = uhi_data.drop(columns=['UHI Index']).values
y = uhi_data ['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

print(f"{X_train.shape}")

# TODO: @ValDLaw23 check if the scaling affects the training
# Scale the training and test data using standardscaler
sc = StandardScaler() # MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Use selected features
X_train = X_train[:, mask_selected_cols]
X_test = X_test[:, mask_selected_cols]

# * Save the scaler
scaler_path = './models/scaler.pkl'
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)

# -----------------------------------------------------------------------------
# Train model with Selected Features
# -----------------------------------------------------------------------------

# * Train the Random Forest model on the training data
# TODO: @ValDLaw23 try boosting methods and another models like XGBoost, LightGBM, CatBoost, GradientBoosting, HistGradientBoosting, etc
print("Training model")
model = RandomForestRegressor(oob_score=True, random_state=SEED)
# model = LGBMRegressor(boosting_type='rf', random_state=SEED)
# model = XGBRegressor(grow_policy='lossguide', verbosity=3, random_state=SEED)
# model = GradientBoostingRegressor(loss='squared_error', random_state=SEED)
# model.fit(X_train, y_train)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
# }

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(oob_score=True, random_state=SEED),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='r2',  # Optimize for R² score
    n_jobs=-1,  # Use all available CPU cores
    verbose=2  # Show progress
)

# grid_search = GridSearchCV(
#     estimator=model, 
#     param_grid=param_grid, 
#     cv=5, 
#     scoring='r2', 
#     n_jobs=-1,
#     verbose=2
# )

# Fit Grid Search on training data
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.4f}")

# Train final model with the best parameters
model = grid_search.best_estimator_

# * OOB Score
try:
    print(f"{model.oob_score_=}")
except:
    pass

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

# * Save the model
os.makedirs('./models', exist_ok=True)
model_path = './models/random_forest_model.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved at {model_path}")

# -----------------------------------------------------------------------------
# Plot the distribution of the error in train and test predictions
# -----------------------------------------------------------------------------

train_errors = Y_train - insample_predictions
test_errors = Y_test - outsample_predictions

# Plot Distribution of Errors
plt.figure(figsize=(10, 5))
sns.histplot(train_errors, bins=30, kde=True, color="blue", label="Train Error", alpha=0.6)
sns.histplot(test_errors, bins=30, kde=True, color="red", label="Test Error", alpha=0.6)
plt.axvline(0, color="black", linestyle="dashed", linewidth=1)  # Reference line at 0
plt.legend()
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution: Train vs Test")
plt.show()


# -----------------------------------------------------------------------------
# Model Feature Importances
# -----------------------------------------------------------------------------

# TODO: @ValDLaw23 research how to plot FE using the data proportioned by the model and then, with SHAP Python library

"""
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
https://mljar.com/blog/feature-importance-in-random-forest/
https://github.com/search?type=code&q=%22RandomForestRegressor%28%22+AND+%22importance%22+language%3APython
https://github.com/search?type=code&q=%22RandomForestRegressor%28%22+AND+%22shap%22+language%3APython
"""
    
# TODO: Research if exists a way to "explain the decision path of random forest with LLMs"
# !python3.10 -m pip install pyarrow fastparquet
