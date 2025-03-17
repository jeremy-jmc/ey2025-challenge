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

results = []
for name, model in models.items():
    print(f"{name=}")
    scores = cross_val_score(model, X_train, y_train, cv=10)
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"{name}: {mean_score:.4f} ± {std_score:.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = r2_score(y_test, y_pred)
    print(f"{name} test accuracy: {test_accuracy:.4f}")

    results.append({
        'Model': name,
        'Mean CV': mean_score,
        'Std CV': std_score,
        'Test Accuracy': test_accuracy
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['Test Accuracy', 'Mean CV'], ascending=False)

display(results_df)

# -----------------------------------------------------------------------------
# Stacking
# -----------------------------------------------------------------------------

rf_best_params_ = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
xgb_best_params_ = {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 20, 'n_estimators': 200, 'subsample': 0.7}
extratrees_best_params_ = {'bootstrap': False, 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
elastic_net_best_params_ =  {'eps': 0.001, 'l1_ratio': 0.1, 'max_iter': 1000, 'n_alphas': 100}

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

with open('./models/extratrees_model.pkl', 'wb') as extratrees_file:
    pickle.dump(model.estimators_[0], extratrees_file)

with open('./models/xgb_model.pkl', 'wb') as xgb_file:
    pickle.dump(model.estimators_[1], xgb_file)

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
