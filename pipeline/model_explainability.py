import sys
sys.path.append('..')

from baseline.utilities import *

# ! pip install shap
import shap
shap.initjs()


selected_features = json.loads(open("selected_features.json", "r").read())['selected_features']
model = joblib.load('./models/random_forest_model.pkl')

X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')

# -----------------------------------------------------------------------------
# Model Explainability
# -----------------------------------------------------------------------------

# * SHAP global explainability plots
shap_tree_explainer = shap.TreeExplainer(model, feature_names=selected_features)
shap_values = shap_tree_explainer(X_train)


# Compare the SHAP values data with X_train
comparison = np.allclose(shap_values.data, X_train)
print(f"Comparison result: {comparison}")
print(f"{shap_values.values.shape=}")

shap.plots.beeswarm(shap_values, max_display=25)
shap.plots.bar(shap_values, max_display=25)

# shap_interaction_values = shap_tree_explainer.shap_interaction_values(X_train)
# shap.summary_plot(shap_interaction_values, X_train, max_display=25)

print(f"{selected_features=}")
for feature in selected_features:
    shap.dependence_plot(feature, shap_values.values, pd.DataFrame(X_train, columns=selected_features))


# * SHAP local explainability plots
sample_idx = 0

# Force plot
shap.force_plot(shap_tree_explainer.expected_value, shap_values.values[sample_idx, :], X_train[sample_idx, :], 
                feature_names=selected_features, matplotlib=True, text_rotation=45)

sample_shap_values = shap_values.values[sample_idx]
sample_features = X_train[sample_idx, :]

contributions = sorted(
    zip(selected_features, sample_features, sample_shap_values),
    key=lambda x: -abs(x[2])  # Sort by absolute impact
)
explanation = "The model's prediction was influenced by:\n"
for feature, value, shap_value in contributions:
    impact = "increased" if shap_value > 0 else "decreased"
    explanation += f"- {feature} ({value:.5f}) {impact} the prediction by {abs(shap_value):.5f}\n"
print(explanation)

# Decision plot
shap.decision_plot(shap_tree_explainer.expected_value, sample_shap_values, sample_features, feature_names=selected_features)

"""
https://medium.com/towards-data-science/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
https://m.mage.ai/how-to-interpret-and-explain-your-machine-learning-models-using-shap-values-471c2635b78e


https://github.com/Luson045/all_teacher_web_of_science/blob/e681d8e49ae47bbbb2c6455728c968bf43851eaa/FinalVizCodes/shap.py#L4


https://github.com/search?type=code&q=%22RandomForestRegressor%22+AND+%22import+shap%22+AND+%22shap.%22+language%3APython
https://medium.com/biased-algorithms/introduction-to-model-explainability-in-regression-26bc8f4ddeb4

https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Fitting%20a%20Linear%20Simulation%20with%20XGBoost.html

https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

https://mljar.com/blog/feature-importance-in-random-forest/
https://github.com/search?type=code&q=%22RandomForestRegressor%28%22+AND+%22importance%22+language%3APython
https://github.com/search?type=code&q=%22RandomForestRegressor%28%22+AND+%22shap%22+language%3APython

https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py
"""
    
# TODO: Research if exists a way to "explain the decision path of random forest with LLMs"


