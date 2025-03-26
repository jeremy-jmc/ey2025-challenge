import sys
sys.path.append('..')

from baseline.utilities import *

TARGET = 'UHI Index'
X = pd.read_parquet('./data/processed/train/X_selected.parquet')
y = pd.read_parquet('./data/processed/train/y_selected.parquet')[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, 
                                                    shuffle=True, 
                                                    random_state=SEED
                                                    )
test_file = pd.read_parquet('./data/processed/submission/submission_data.parquet')

# * Load Model
model = joblib.load('./models/stacking.pkl')
sc = joblib.load('./models/scaler.pkl')

import shap
import matplotlib.pyplot as plt

# Load individual models
extratrees_model = model.estimators_[0]
xgb_model = model.estimators_[1]


# Ensure SHAP values can be computed
for model, name in [(extratrees_model, "ExtraTrees"), (xgb_model, "XGBoost")]:
    print(f"Generating SHAP values for {name}...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Plot SHAP summary
    plt.title(f"SHAP Summary Plot for {name}")
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.show()
