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
        return redirect(url_for('summary'))  # Proceed to summary page
    return render_template('manual/step6.html')  # Render Step 6 form

#Step 7 
@app.route('/step7', methods=['GET', 'POST'])
def step7():
    if request.method == 'POST':
        blood_glucose = request.form.get('blood_glucose')
        bp_systolic = request.form.get('bp_systolic')
        bp_diastolic = request.form.get('bp_diastolic')
        vitamin_d3 = request.form.get('vitamin_d3')
        pulse_rate = request.form.get('pulse_rate')
        hb = request.form.get('hb')
        # Store the data as needed, e.g., in session or a database
        return redirect(url_for('step8'))  # Proceed to step 8

    return render_template('step7.html')

#Step 8
@app.route('/step8', methods=['GET', 'POST'])
def step8():
    if request.method == 'POST':
        pimples = request.form.get('pimples')
        hair_loss = request.form.get('hair_loss')
        skin_darkening = request.form.get('skin_darkening')
        weight_gain = request.form.get('weight_gain')
        # Store the data as needed
        return redirect(url_for('summary'))  # Proceed to summary page

    return render_template('step8.html')

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
    return render_template('summary.html', data=session)  # Display all collected data

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Initialize an empty list for user input
    user_input = []

    # Blood group mapping
    blood_group_mapping = {
        'A-': 11, 'A+': 12, 'B-': 13, 'B+': 14,
        'O-': 15, 'O+': 16, 'AB-': 17, 'AB+': 18
    }

    # Collect user input from the form
    for feature in feature_names:
        value = session.get(feature, None)
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
