import sys
sys.path.append('..')

import yaml
from baseline.utilities import *

MODE = 'feature_selection'  # 'train_model', 'feature_selection'
HT = True

train_data = pd.read_parquet('./data/processed/train/train_data.parquet')
print(train_data.shape)
# y_cluster = train_data['cluster']
# train_data = train_data.drop(columns=['cluster'])

feature_list = yaml.safe_load(open('./data/columns.yml', 'r'))['features']

# feature_list = [f for f in feature_list if not any(f.startswith(c) for c in ['sntnl_buffer_band'])]
# [f'sntnl_buffer_band_{v}' for v in range(1, 12) if v not in [1, 9]]
# feature_list = [f for f in feature_list if not any(f.startswith(c) for c in ['sntnl'])]
print(f"({len(feature_list)}) -> {feature_list=}")

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

print(f"{train_data.shape=}")
uhi_data = train_data.copy()
# Sorted Columns Alphabetically
uhi_data = uhi_data[sorted(feature_list + ['UHI Index', 'Latitude', 'Longitude'])]

# [['B01', 'B06', 'B8A', 'NDVI', 'UHI Index', 'B02', 'B03', 'B04', 'B05', 'B07', 'B08',  'B11', 'B12', 'gNDBI'] + feature_list]
for col in uhi_data.columns:
    print(col)

print(f"{uhi_data.shape=}")
print(uhi_data.isna().sum())
# display(uhi_data.head())

X = uhi_data.drop(columns=['UHI Index'])
y = uhi_data['UHI Index'].values

# -----------------------------------------------------------------------------
# Train/Test Split
# -----------------------------------------------------------------------------

X = uhi_data.drop(columns=['UHI Index'])
y = uhi_data ['UHI Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    shuffle=True, 
                                                    random_state=SEED, 
                                                    # stratify=y_cluster
                                                    )

geodf_split = pd.concat([X_train.assign(partition='train'), X_test.assign(partition='test')], axis=0)[['Latitude', 'Longitude', 'partition']].reset_index(drop=True)
geojson_to_check_split = gpd.GeoDataFrame(
    geodf_split, 
    geometry=gpd.points_from_xy(geodf_split['Longitude'], geodf_split['Latitude']),
    crs='EPSG:4326'
)
geojson_to_check_split.to_file('./split_data.geojson', driver='GeoJSON')

try:
    X_train = X_train.drop(columns=['partition'])
    X_test = X_test.drop(columns=['partition'])
except:
    pass

try:
    X = X.drop(columns=['Latitude', 'Longitude'])
    X_train = X_train.drop(columns=['Latitude', 'Longitude'])
    X_test = X_test.drop(columns=['Latitude', 'Longitude'])

    X = X.drop(columns=['datetime'])
    X_train = X_train.drop(columns=['datetime'])
    X_test = X_test.drop(columns=['datetime'])

    X = X.drop(columns=['latitude', 'longitude'])
    X_train = X_train.drop(columns=['latitude', 'longitude'])
    X_test = X_test.drop(columns=['latitude', 'longitude'])
except:
    pass

display(X_train)
display(X_test)

# TODO: Save the X_train, X_test in GeoJSON format in order to check how the split is done.
X_train = X_train.values
X_test = X_test.values

# TODO: Check how well the data splitting is done
print(f"{X_train.shape=}")
print(f"{X_test.shape=}")

# TODO: @ValDLaw23 check if the scaling affects the training
# Scale the training and test data using standardscaler
sc = StandardScaler() # MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------------------------------------------------------
# Baseline Model
# -----------------------------------------------------------------------------

print("BASELINE MODEL")
baseline_model = RandomForestRegressor(oob_score=True, random_state=SEED, n_jobs=-1)
cv_scores = cross_val_score(baseline_model, X, y, cv=5, scoring='r2')
print(f"{cv_scores.mean()=}")

# -----------------------------------------------------------------------------
# Feature Selection
# -----------------------------------------------------------------------------

if MODE == 'feature_selection':
    # ! DecisionTreeRegressor is sensitive to feature order
    """
    https://stackoverflow.com/questions/43941163/does-feature-order-impact-decision-tree-algorithm-in-sklearn
    https://github.com/scikit-learn/scikit-learn/issues/5394
    """
    # DecisionTreeRegressor(random_state=SEED)
    # GradientBoostingRegressor(random_state=42)
    rfe_fs = RFECV(estimator=DecisionTreeRegressor(random_state=SEED), cv=KFold(n_splits=3, shuffle=True, random_state=SEED), scoring='r2', step=10, n_jobs=-1, verbose=1)
    # rfe_fs = RFE(estimator=RandomForestRegressor(n_estimators=100, oob_score=True, random_state=SEED, n_jobs=-1), n_features_to_select=20, step=1, verbose=1)

    X_selected = rfe_fs.fit_transform(X, y)
    # TODO: ValDLaw23 research the computational viability of BorutaRandomForest or another method to select features
    """
    https://www.kaggle.com/code/residentmario/automated-feature-selection-with-boruta
    https://amueller.github.io/aml/05-advanced-topics/12-feature-selection.html

    https://www.kaggle.com/code/attackgnome/basic-feature-benchmark-rfecv-xgboost
    https://gist.github.com/Tejas-Deo/fb193565ffcaba4e5ee1b5d5a0852e66
    """

    print(f"{rfe_fs.ranking_=}")
    try:
        print(f"{rfe_fs.cv_results_.keys()=}")

        # TODO: plot RFECV with error bar. Source: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
        plt.figure()
        plt.plot(rfe_fs.cv_results_['n_features'][2:], rfe_fs.cv_results_['mean_test_score'][2:], color='blue')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Number of Features Selected')
        plt.ylabel('Cross-Validation Score')
        plt.title('RFECV - Number of Features vs. Cross-Validation Score')
        plt.show()
    except:
        pass

    # Print selected features
    selected_features = X.columns[rfe_fs.support_]
    print(f"Selected features ({len(selected_features)}): {list(selected_features)}")
    with open("selected_features.json", "w") as f:
        f.write(json.dumps({
            "selected_features" : list(selected_features)
        }, indent=4))

    mask_selected_cols = rfe_fs.support_
elif MODE == 'train_model':
    selected_features = json.loads(open("selected_features.json", "r").read())['selected_features']
    mask_selected_cols = [True if col in selected_features else False for col in X.columns]

print(f"{mask_selected_cols=}")

X_train = X_train[:, mask_selected_cols]
X_test = X_test[:, mask_selected_cols]

print(f"{X_train.shape=}")
print(f"{X_test.shape=}")

# * Correlation matrix of selected features
corr_matrix = pd.DataFrame(X_train, columns=selected_features).corrwith(pd.Series(y_train, name='UHI Index'))
plt.figure()
sns.heatmap(corr_matrix.to_frame(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix between Features and Target (UHI Index)")
plt.show()

# * Save the X_train and X_test
np.save('./data/X_train.npy', X_train)
np.save('./data/X_test.npy', X_test)

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

if HT:
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    #     'subsample': [0.8, 1.0],
    # }

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None,8, 10, 20],
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

    # Fit Grid Search on training data
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R² score: {grid_search.best_score_:.4f}")

    # Save Best Params
    with open("best_params.json", "w") as f:
        f.write(json.dumps({
            "best_params" : grid_search.best_params_
        }, indent=4))

    # Train final model with the best parameters
    model = grid_search.best_estimator_

    cv_scores = cross_val_score(RandomForestRegressor(**grid_search.best_params_, random_state=SEED), X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"{cv_scores.mean()=}")

else:
    model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=SEED, n_jobs=-1)
    
    cv_scores = cross_val_score(copy.deepcopy(model), X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"{cv_scores.mean()=}")

    # model = LGBMRegressor(boosting_type='rf', random_state=SEED)
    # model = XGBRegressor(grow_policy='lossguide', verbosity=3, random_state=SEED)
    # model = GradientBoostingRegressor(loss='squared_error', random_state=SEED)
    model.fit(X_train, y_train)
    
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
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)

    print(f"{fold} - R² Scores: {r2_scores}")
    print(f"{fold} - Mean R² Score: {np.mean(r2_scores):.4f}")
    print(f"{fold} - Standard Deviation of R² Scores: {np.std(r2_scores):.4f}")
    print()

# -----------------------------------------------------------------------------
# Save the model
# -----------------------------------------------------------------------------

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
# Overfitting detection
# -----------------------------------------------------------------------------

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    baseline_model, X, y, cv=5, scoring="r2", train_sizes=[0.7, 0.8, 0.9], n_jobs=4, verbose=1
)   # np.linspace(0.1, 1.0, 10)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train R²", marker="o")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation R²", marker="s")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# Model Feature Importances
# -----------------------------------------------------------------------------

# TODO: @ValDLaw23 research how to plot FE using the data proportioned by the model and then, with SHAP Python library
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

if hasattr(model, 'feature_importances_'):  # isinstance(model, RandomForestRegressor)
    feature_names = selected_features
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI (Mean Decrease in Impurity)")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Model Decision Path
# -----------------------------------------------------------------------------

(node_indicator, n_nodes_ptr) = model.decision_path(X_train[[0]])
print(f"{node_indicator.shape=}")
print(f"{n_nodes_ptr.shape=}")

"""
https://stackoverflow.com/questions/48869343/decision-path-for-a-random-forest-classifier
"""

# !python3.10 -m pip install pyarrow fastparquet