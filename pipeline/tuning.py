import sys
sys.path.append('..')

from baseline.utilities import *
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor


TARGET = 'UHI Index'
X = pd.read_parquet('./data/processed/train/X_selected.parquet')
y = pd.read_parquet('./data/processed/train/y_selected.parquet')[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, 
                                                    shuffle=True, 
                                                    random_state=SEED,
                                                    )

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"{X_train.shape=}")
print(f"{X_test.shape=}")


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

print(f"Best R² score: {grid_search_xgb.best_score_:.4f}")
print(f"Best parameters: {grid_search_xgb.best_params_}")

# xgb_model = grid_search_xgb.best_estimator_
# with open('./models/xgb_model.pkl', 'wb') as xgb_file:
#     pickle.dump(xgb_model, xgb_file)


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

print(f"Best R² score: {grid_search_extratrees.best_score_:.4f}")
print(f"Best parameters: {grid_search_extratrees.best_params_}")

# extratrees_model = grid_search_extratrees.best_estimator_
# with open('./models/extratrees_model.pkl', 'wb') as extratrees_file:
#     pickle.dump(extratrees_model, extratrees_file)
