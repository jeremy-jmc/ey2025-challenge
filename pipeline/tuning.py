import sys
sys.path.append('..')

import yaml
from baseline.utilities import *
import copy
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, ElasticNetCV


TARGET = 'UHI Index'
X = pd.read_parquet('./data/processed/train/X_selected.parquet')
y = pd.read_parquet('./data/processed/train/y_selected.parquet')[TARGET]

# 0.9738 -> 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, 
                                                    shuffle=True, 
                                                    random_state=SEED, 
                                                    # stratify=y_cluster
                                                    )

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"{X_train.shape=}")
print(f"{X_test.shape=}")


# -----------------------------------------------------------------------------
# Hyperparameter tuning Random Forest
# -----------------------------------------------------------------------------

rf_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 8, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    # 'bootstrap': [True, False]
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestRegressor(oob_score=True, random_state=SEED),
    param_grid=rf_param_grid,
    cv=10,
    scoring='r2',  # Optimize for R² score
    n_jobs=-1,  # Use all available CPU cores
    verbose=2  # Show progress
)
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best R² score: {grid_search_rf.best_score_:.4f}")

rf_model = grid_search_rf.best_estimator_

with open('./models/rf_model.pkl', 'wb') as rf_file:
    pickle.dump(rf_model, rf_file)

# -----------------------------------------------------------------------------
# Hyperparameter tuning XGBoost
# -----------------------------------------------------------------------------

xgb_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 7, 9, 20],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.7, 1],
    'colsample_bytree': [0.5, 0.7, 1],
    # 'gamma': [0, 0.1, 0.2],
}

grid_search_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=SEED),
    param_grid=xgb_param_grid,
    cv=10,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
grid_search_xgb.fit(X_train, y_train)

print(f"Best parameters: {grid_search_xgb.best_params_}")
print(f"Best R² score: {grid_search_xgb.best_score_:.4f}")

xgb_model = grid_search_xgb.best_estimator_

with open('./models/xgb_model.pkl', 'wb') as xgb_file:
    pickle.dump(xgb_model, xgb_file)

# -----------------------------------------------------------------------------
# Hyperparameter tuning ExtraTrees
# -----------------------------------------------------------------------------

extratrees_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 8, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['log2', 'sqrt', None],
    # 'bootstrap': [True, False]
}

grid_search_extratrees = GridSearchCV(
    estimator=ExtraTreesRegressor(random_state=SEED),
    param_grid=extratrees_param_grid,
    cv=10,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
grid_search_extratrees.fit(X_train, y_train)

print(f"Best parameters: {grid_search_extratrees.best_params_}")
print(f"Best R² score: {grid_search_extratrees.best_score_:.4f}")

extratrees_model = grid_search_extratrees.best_estimator_

with open('./models/extratrees_model.pkl', 'wb') as extratrees_file:
    pickle.dump(extratrees_model, extratrees_file)

# -----------------------------------------------------------------------------
# Tuning ElasticNetCV with Stacking
# -----------------------------------------------------------------------------

stacking_param_grid = {
    'l1_ratio': [0.1, 0.5, 0.7, 0.9],
    'eps': [0.001, 0.01, 0.1],
    'n_alphas': [100, 200, 300],
    'max_iter': [1000, 2000, 3000],
    'cv': [10]
}

grid_search_stacking = GridSearchCV(
    estimator=ElasticNetCV(random_state=SEED),
    param_grid=stacking_param_grid,
    cv=10,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

xgb_model = pickle.load(open('./models/xgb_model.pkl', 'rb'))
extratrees_model = pickle.load(open('./models/extratrees_model.pkl', 'rb'))

# Train
extra_trees_pred = extratrees_model.predict(X_train)
xgb_pred = xgb_model.predict(X_train)

X_train_stacking = np.column_stack((extra_trees_pred, xgb_pred))

grid_search_stacking.fit(X_train_stacking, y_train)

print(f"Best parameters: {grid_search_stacking.best_params_}")
print(f"Best R² score: {grid_search_stacking.best_score_:.4f}")

# Test
extra_trees_pred = extratrees_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

X_test_stacking = np.column_stack((extra_trees_pred, xgb_pred))

preds = grid_search_stacking.best_estimator_.predict(X_test_stacking)
test_accuracy = r2_score(y_test, preds)
print(f"Test accuracy: {test_accuracy:.4f}")
