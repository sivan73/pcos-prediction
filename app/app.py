from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import os

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # for session tracking

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

# Blood group mapping
blood_group_mapping = {
    'A-': 11, 'A+': 12, 'B-': 13, 'B+': 14,
    'O-': 15, 'O+': 16, 'AB-': 17, 'AB+': 18
}

# Step 1: Choose manual or upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['input_method'] = request.form.get('input_method')
        if session['input_method'] == 'upload':
            return redirect(url_for('upload_report'))
        return redirect(url_for('step2'))  # Continue to Step 2 for manual entry
    return render_template('manual/step1.html')  # Render Step 1 form

# Step 2: Basic Info - Name, Age
@app.route('/step2', methods=['GET', 'POST'])
def step2():
    if request.method == 'POST':
        session['name'] = request.form.get('name')
        session['age'] = request.form.get('age')
        return redirect(url_for('step3'))  # Proceed to Step 3
    return render_template('manual/step2.html')  # Render Step 2 form

# Step 3: Height, Weight
@app.route('/step3', methods=['GET', 'POST'])
def step3():
    if request.method == 'POST':
        session['height'] = request.form.get('height')
        session['weight'] = request.form.get('weight')
        return redirect(url_for('step4'))  # Proceed to Step 4
    return render_template('manual/step3.html')  # Render Step 3 form

# Step 4: Cycle Info (Menstrual History, etc.)
@app.route('/step4', methods=['GET', 'POST'])
def step4():
    if request.method == 'POST':
        session['cycle_length'] = request.form.get('cycle_length')
        session['pregnancy_status'] = request.form.get('pregnancy_status')
        session['abortions'] = request.form.get('abortions')
        return redirect(url_for('step5'))  # Proceed to Step 5
    return render_template('manual/step4.html')  # Render Step 4 form

# Step 5: Other details (Lifestyle, Symptoms, etc.)
@app.route('/step5', methods=['GET', 'POST'])
def step5():
    if request.method == 'POST':
        session['weight_gain'] = request.form.get('weight_gain')
        session['hair_growth'] = request.form.get('hair_growth')
        session['skin_darkening'] = request.form.get('skin_darkening')
        session['pimples'] = request.form.get('pimples')
        session['fast_food'] = request.form.get('fast_food')
        session['exercise'] = request.form.get('exercise')
        return redirect(url_for('step6'))  # Proceed to Step 6
    return render_template('manual/step5.html')  # Render Step 5 form

# Step 6: Medical Test Results (Optional)
@app.route('/step6', methods=['GET', 'POST'])
def step6():
    if request.method == 'POST':
        session['blood_group'] = request.form.get('blood_group')
        session['hb'] = request.form.get('hb')
        session['fs_hl'] = request.form.get('fs_hl')  # Collect additional data as needed
        return redirect(url_for('step7'))  # Proceed to step 7
    return render_template('manual/step6.html')  # Render Step 6 form

# Step 7: Blood Pressure & Glucose
@app.route('/step7', methods=['GET', 'POST'])
def step7():
    if request.method == 'POST':
        session['blood_glucose'] = request.form.get('blood_glucose')
        session['bp_systolic'] = request.form.get('bp_systolic')
        session['bp_diastolic'] = request.form.get('bp_diastolic')
        session['vitamin_d3'] = request.form.get('vitamin_d3')
        session['pulse_rate'] = request.form.get('pulse_rate')
        return redirect(url_for('step8'))  # Proceed to step 8
    return render_template('manual/step7.html')

# Step 8: Hair Loss, Skin Darkening, Weight Gain
@app.route('/step8', methods=['GET', 'POST'])
def step8():
    if request.method == 'POST':
        session['hair_loss'] = request.form.get('hair_loss')
        session['skin_darkening'] = request.form.get('skin_darkening')
        session['weight_gain'] = request.form.get('weight_gain')
        return redirect(url_for('summary'))  # Proceed to summary page
    return render_template('manual/step8.html')

# Step for Uploading Medical Report (Placeholder)
@app.route('/upload', methods=['GET', 'POST'])
def upload_report():
    if request.method == 'POST':
        # Handle file upload and parsing logic here
        session['file_data'] = request.files['report_file']
        return redirect(url_for('summary'))  # Redirect to summary after processing the file
    return render_template('upload_report.html')  # Render upload page

# Summary of all collected session data
@app.route('/summary')
def summary():
    return render_template('summary.html', feature_names=feature_names)  # Pass feature_names to summary

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    user_input = []

    for feature in feature_names:
        value = session.get(feature)

        if feature == 'Blood_Group':
            user_input.append(blood_group_mapping.get(value, 0))
        elif value is not None:
            try:
                user_input.append(float(value))
            except ValueError:
                user_input.append(0.0)  # Fallback to 0 if conversion fails
        else:
            user_input.append(0.0)  # Default value if feature is missing

    input_df = pd.DataFrame([user_input], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    risk_score = model.predict_proba(input_scaled)[0][1]
    result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"

    explanation = explainer.explain_instance(
        data_row=input_scaled[0],
        predict_fn=model.predict_proba,
        num_features=5
    )
    lime_html = explanation.as_html()

    return render_template('result.html', result=result, risk_score=risk_score, lime_html=lime_html)

if __name__ == '__main__':
    app.run(debug=True)
