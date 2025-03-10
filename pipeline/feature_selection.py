import sys
sys.path.append('..')

import yaml
from baseline.utilities import *

np.set_printoptions(suppress=True)

TARGET = 'UHI Index'

train_old = pd.read_parquet('./data/train_data.parquet')
train_new = pd.read_parquet('./data/processed/train/train_data.parquet')

intersect_columns = train_old.columns.intersection(train_new.columns)
print(f"{train_old[intersect_columns].equals(train_new[intersect_columns])=}")

feature_list = yaml.safe_load(open('./data/columns.yml', 'r'))['features']

X = train_new.drop(columns=['Longitude', 'Latitude', 'datetime', TARGET])[feature_list]
y = train_new[TARGET]

# X = train_old.drop(columns=['Longitude', 'Latitude', 'datetime', TARGET])
# y = train_old[TARGET]

print(f"Original column length -> {len(X.columns)=}")

# -----------------------------------------------------------------------------
# Drop columns with 0 correlation with the target
# -----------------------------------------------------------------------------

correlations = train_new[feature_list + [TARGET]].corr()

correlations = correlations[TARGET].sort_values(ascending=False)

threshold = 0.05
columns_to_keep = correlations[abs(correlations) >= threshold].index

X = train_new[columns_to_keep].drop(columns=TARGET)
y = train_new[TARGET]
print(f"After correlation threshold removal -> {len(X.columns)=}")

# -----------------------------------------------------------------------------
# Collinearity problem
# -----------------------------------------------------------------------------
from scipy.stats import pearsonr
import numpy as np

corr_matrix = X.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(f"{to_drop=}")

X = X.drop(columns=to_drop)
print(f"After collinearity removal -> {len(X.columns)=}")

# -----------------------------------------------------------------------------
# RFECV with RandomForest
# -----------------------------------------------------------------------------

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

rfe_fs = RFECV(estimator=RandomForestRegressor(oob_score=True, random_state=SEED), 
               cv=KFold(n_splits=5, shuffle=True, random_state=SEED), 
               scoring='r2', step=2, n_jobs=-1, verbose=1)
X_selected = rfe_fs.fit_transform(X, y)

print(f"{rfe_fs.ranking_=}")
try:
    print(f"{rfe_fs.cv_results_.keys()=}")

    plt.figure()
    # plt.plot(rfe_fs.cv_results_['n_features'][2:], rfe_fs.cv_results_['mean_test_score'][2:], color='blue')
    plt.errorbar(
        x=rfe_fs.cv_results_["n_features"][2:],
        y=rfe_fs.cv_results_["mean_test_score"][2:],
        yerr=rfe_fs.cv_results_["std_test_score"][2:],
    )
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validation Score (R2)')
    plt.title('RFECV - Number of Features vs. Cross-Validation Score')
    plt.show()
except:
    pass

X_rfe = X.loc[:, rfe_fs.support_]

print(f"After RFECV Feature Selection-> {X_selected.shape[1]=}")
print(X_selected.var(axis=0))

display(X_rfe.head())

X_rfe.to_parquet('./data/processed/train/X_selected.parquet')
pd.DataFrame(y).to_parquet('./data/processed/train/y_selected.parquet')


# # -----------------------------------------------------------------------------
# # Drop columns with 0 variance
# # -----------------------------------------------------------------------------

# from sklearn.feature_selection import VarianceThreshold

# selector = VarianceThreshold(0.2)
# selector.fit(X_selected)

# print(len(X_selected.loc[:, selector.get_support()].columns))
# # X = X.loc[:, selector.get_support()]


# -----------------------------------------------------------------------------
# Variance Threshold
# -----------------------------------------------------------------------------

# !pip install statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF             = pd.DataFrame()
VIF['feature']  = X_rfe.columns
VIF['VIF']      = [variance_inflation_factor(X_rfe.values, i) 
                   for i in tqdm(range(X_rfe.shape[1]), total=X_rfe.shape[1], desc='Calculating VIF')]

display(VIF.sort_values('VIF', ascending=False))


# -----------------------------------------------------------------------------
# Lasso Feature Selection
# -----------------------------------------------------------------------------

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

selector = SelectFromModel(RandomForestRegressor().fit(X_rfe, y), prefit=True)
selector.fit(X_rfe, y)

print(selector.estimator_.score(X_rfe, y))

selected_features = X_rfe.columns[selector.get_support()]

print(f"{selected_features=}")

