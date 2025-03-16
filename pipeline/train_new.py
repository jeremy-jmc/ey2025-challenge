import sys
sys.path.append('..')

import yaml
from baseline.utilities import *
import copy

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

scaler_path = './models/scaler.pkl'
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)


# -----------------------------------------------------------------------------
# Compare models
# -----------------------------------------------------------------------------

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, ElasticNetCV

models = {
    # 'Linear Regression': LinearRegression(),
    # 'Ridge Regression': Ridge(),
    # 'Lasso Regression': Lasso(),
    # 'Elastic Net': ElasticNet(),

    'Random Forest': RandomForestRegressor(random_state=SEED),
    'Gradient Boosting': GradientBoostingRegressor(random_state=SEED),
    'XGBoost': XGBRegressor(random_state=SEED),
    'LightGBM': LGBMRegressor(verbosity=0, random_state=SEED),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=SEED),

    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=SEED),
    'Extra Trees': ExtraTreesRegressor(random_state=SEED),
    'AdaBoost': AdaBoostRegressor(random_state=SEED),
    'Bagging': BaggingRegressor(random_state=SEED),
    'Decision Tree': DecisionTreeRegressor(random_state=SEED),

    # 'Stacking': StackingRegressor(
    #     estimators=[
    #         ('et', ExtraTreesRegressor(n_jobs=-2, random_state=SEED)),
    #         # ('rf', RandomForestRegressor(n_jobs=-2, random_state=SEED)), 
    #         # ('bg', BaggingRegressor(n_jobs=-2, random_state=SEED)),
    #         ('xgb', XGBRegressor(n_jobs=-2, random_state=SEED)),
    #         # ('ctb', CatBoostRegressor(verbose=0, random_state=SEED))
    #     ],
    #     final_estimator=ElasticNetCV(cv=10, n_jobs=-2, random_state=SEED),
    #     verbose=2
    # )
}

for name, model in models.items():
    print(f"{name=}")
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} test accuracy: {r2_score(y_test, y_pred):.4f}")

"""
Random Forest: 0.9572 ± 0.0020
Random Forest test accuracy: 0.9711

Gradient Boosting: 0.7899 ± 0.0053
Gradient Boosting test accuracy: 0.7728

XGBoost: 0.9468 ± 0.0010
XGBoost test accuracy: 0.9590

LightGBM: 0.9288 ± 0.0015
LightGBM test accuracy: 0.9404

CatBoost: 0.9442 ± 0.0018
CatBoost test accuracy: 0.9596


Stacking: 0.9659 ± 0.0029
Stacking test accuracy: 0.9742
"""

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

# -----------------------------------------------------------------------------
# Stacking
# -----------------------------------------------------------------------------

rf_best_params_ = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
xgb_best_params_ = {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 20, 'n_estimators': 100, 'subsample': 1}
extratrees_best_params_ = {'bootstrap': False, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

model = StackingRegressor(
    estimators=[
        # ('rf', RandomForestRegressor(**rf_best_params_, n_jobs=-2, random_state=SEED)), 
        ('et', ExtraTreesRegressor(**extratrees_best_params_, n_jobs=-2, random_state=SEED)),
        ('xgb', XGBRegressor(**xgb_best_params_, n_jobs=-2, random_state=SEED)),
        # ('ctb', CatBoostRegressor(verbose=0, random_state=SEED))
    ],
    final_estimator=ElasticNetCV(cv=10, n_jobs=-2, random_state=SEED),
    verbose=2
)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Model Generalization Capability Evaluation
# -----------------------------------------------------------------------------

# * Make predictions on the training data (in-sample predictions)
insample_predictions = model.predict(X_train)
Y_train = y_train.tolist()
print(f"{r2_score(Y_train, insample_predictions)=}")

# * Make predictions on the test data (out-sample predictions)
outsample_predictions = model.predict(X_test)
Y_test = y_test.tolist()
print(f"{r2_score(Y_test, outsample_predictions)=}")

# -----------------------------------------------------------------------------
# Save the model
# -----------------------------------------------------------------------------

os.makedirs('./models', exist_ok=True)
model_path = './models/stacking.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved at {model_path}")
