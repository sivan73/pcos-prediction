import pickle
import numpy as np
import lime
import lime.lime_tabular
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load required model components
with open(os.path.join(MODELS_DIR, 'pcos_training_data.pkl'), 'rb') as f:
    training_data = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'pcos_feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'pcos_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'pcos_ensemble_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Set up the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(training_data),
    feature_names=feature_names,
    class_names=['No PCOS', 'PCOS'],
    discretize_continuous=True,
    mode='classification'
)

def explain_instance(instance_values):
    """
    Explain a single prediction instance using LIME.

    Parameters:
    - instance_values: List or NumPy array of feature values before scaling.

    Returns:
    - HTML string of LIME explanation.
    """
    instance_scaled = scaler.transform([instance_values])
    explanation = explainer.explain_instance(
        data_row=instance_scaled[0],
        predict_fn=model.predict_proba,
        num_features=10
    )
    return explanation.as_html()
