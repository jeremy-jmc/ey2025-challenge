import sys
sys.path.append('..')

import yaml
from baseline.utilities import *
import copy

TARGET = 'UHI Index'
X = pd.read_parquet('./data/processed/train/X_selected.parquet')
y = pd.read_parquet('./data/processed/train/y_selected.parquet')[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, 
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
# Hyperparameter tuning
# -----------------------------------------------------------------------------

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 8, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    # 'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(oob_score=True, random_state=SEED),
    param_grid=param_grid,
    cv=10,
    scoring='r2',  # Optimize for R² score
    n_jobs=-1,  # Use all available CPU cores
    verbose=2  # Show progress
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

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
model_path = './models/random_forest_model.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved at {model_path}")
