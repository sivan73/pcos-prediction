from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('E:/Project/PCOS/pcos-prediction/models/pcos_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('E:/Project/PCOS/pcos-prediction/models/pcos_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('E:/Project/PCOS/pcos-prediction/models/pcos_training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)
with open('E:/Project/PCOS/pcos-prediction/models/pcos_feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Set up the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(training_data),
    feature_names=feature_names,
    class_names=['No PCOS', 'PCOS'],
    discretize_continuous=True,
    mode='classification'
)

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Initialize an empty list for user input
    user_input = []

    # Blood group mapping
    blood_group_mapping = {
        'A-': 11,
        'A+': 12,
        'B-': 13,
        'B+': 14,
        'O-': 15,
        'O+': 16,
        'AB-': 17,
        'AB+': 18
    }

    # Collect user input from the form
    for feature in feature_names:
        value = request.form[feature]
    if feature == 'Blood_Group':
        # Map blood group to its corresponding number
        user_input.append(blood_group_mapping.get(value, 0))  # Default to 0 if not found
    elif feature in ['CycleRI', 'Hair_lossYN', 'hair_growthYN', 'Skin_darkening_YN', 'PimplesYN', 'Weight_gainYN', 'Fast_food_YN', 'RegExerciseYN']:
        # For categorical features, append the string value directly
        user_input.append(value)
    else:
        # For numerical features, convert to float
        user_input.append(float(value) if value else 0)

    # Create DataFrame for input
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict using the ensemble model
    prediction = model.predict(input_scaled)[0]
    risk_score = model.predict_proba(input_scaled)[0][1]  # Probability of PCOS
    result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"

    # LIME explanation
    explanation = explainer.explain_instance(
        data_row=input_scaled[0],
        predict_fn=model.predict_proba,
        num_features=5  # Show top 5 contributing features
    )
    lime_html = explanation.as_html()

    return render_template('result.html', result=result, risk_score=risk_score, lime_html=lime_html)

if __name__ == '__main__':
    app.run(debug=True)