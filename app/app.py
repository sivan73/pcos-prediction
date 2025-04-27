from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '12345687'

# Load the trained model and scaler
with open('E:/Project/PCOS/pcos-prediction/models/pcos_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('E:/Project/PCOS/pcos-prediction/models/pcos_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('E:/Project/PCOS/pcos-prediction/models/pcos_training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)
with open('E:/Project/PCOS/pcos-prediction/models/pcos_feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Set up LIME explainer
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

@app.route('/')
def home():
    return redirect(url_for('welcome'))

@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        method = request.form.get('input_method')
        session['input_method'] = method
        if method == 'upload':
            return redirect(url_for('upload_report'))
        return redirect(url_for('step0'))
    return render_template('welcome.html')

# Step 0: Patient Information (Full Name, Contact details)
@app.route('/step0', methods=['GET', 'POST'])
def step0():
    if request.method == 'POST':
        # Collecting the details from the form
        session['Full_Name'] = request.form.get('full_name')
        session['Phone'] = request.form.get('phone')
        session['Email'] = request.form.get('email')
        
        # Redirect to Step 1
        return redirect(url_for('step1'))
    return render_template('manual/step0.html', current_step=0, total_steps=8)


@app.route('/step1', methods=['GET', 'POST'])
def step1():
    if request.method == 'POST':
        age = request.form.get('age')
        marriage_years = request.form.get('marriage_years')
        if age and marriage_years:
            try:
                session['Age_yrs'] = float(age)
                session['Marraige_Status_Yrs'] = float(marriage_years)
                return redirect(url_for('step2'))
            except ValueError:
                return "Invalid input. Please enter valid numbers."
        else:
            return "Please fill all the fields."
    return render_template('manual/step1.html', current_step=1, total_steps=8)

# Step 2: Physical Measurements
@app.route('/step2', methods=['GET', 'POST'])
def step2():
    if request.method == 'POST':
        session['HeightCm'] = float(request.form.get('height'))
        session['Weight_Kg'] = float(request.form.get('weight'))
        session['Waistinch'] = float(request.form.get('waist'))
        session['Hipinch'] = float(request.form.get('hip'))

        # Calculate BMI and Waist-Hip Ratio
        height_m = session['HeightCm'] / 100
        session['BMI'] = round(session['Weight_Kg'] / (height_m ** 2), 2)
        session['WaistHip_Ratio'] = round(session['Waistinch'] / session['Hipinch'], 2)

        return redirect(url_for('step3'))
    return render_template('manual/step2.html', current_step=2, total_steps=8)

# Step 3: Menstrual & Reproductive History
@app.route('/step3', methods=['GET', 'POST'])
def step3():
    if request.method == 'POST':
        pregnant = request.form.get('pregnant')
        if pregnant == 'Yes':
            session['PregnantYN'] = 1
        elif pregnant == 'No':
            session['PregnantYN'] = 0
        else:
            flash('Please select Yes or No for the pregnancy question', 'error')
            return render_template('manual/step3.html', current_step=3, total_steps=8)
        
        session['Cycle_lengthdays'] = float(request.form.get('cycle_length'))
        session['No_of_abortions'] = int(request.form.get('abortions'))
        return redirect(url_for('step4'))
    
    return render_template('manual/step3.html', current_step=3, total_steps=8)


# Step 4: Lifestyle & Symptoms
@app.route('/step4', methods=['GET', 'POST'])
def step4():
    if request.method == 'POST':
        # Handle 'Yes' and 'No' responses for each question, converting them to 1 and 0 respectively
        pimples = request.form.get('pimples')
        session['PimplesYN'] = 1 if pimples == 'Yes' else 0
        
        fast_food = request.form.get('fast_food')
        session['Fast_food_YN'] = 1 if fast_food == 'Yes' else 0
        
        skin_darkening = request.form.get('skin_darkening')
        session['Skin_darkening_YN'] = 1 if skin_darkening == 'Yes' else 0
        
        hair_growth = request.form.get('hair_growth')
        session['hair_growthYN'] = 1 if hair_growth == 'Yes' else 0
        
        weight_gain = request.form.get('weight_gain')
        session['Weight_gainYN'] = 1 if weight_gain == 'Yes' else 0
        
        hair_loss = request.form.get('hair_loss')
        session['Hair_lossYN'] = 1 if hair_loss == 'Yes' else 0
        
        exercise = request.form.get('exercise')
        session['RegExerciseYN'] = 1 if exercise == 'Yes' else 0 
        
        # Redirect to the next step if all fields are valid
        return redirect(url_for('step5'))
    
    # Render the step4 template if it's a GET request
    return render_template('manual/step4.html', current_step=4, total_steps=8)


# Step 5: Vital Signs
@app.route('/step5', methods=['GET', 'POST'])
def step5():
    if request.method == 'POST':
        session['Pulse_ratebpm'] = float(request.form.get('pulse_rate'))
        session['RR_breathsmin'] = float(request.form.get('respiratory_rate'))
        session['BP__Systolic_mmHg'] = float(request.form.get('bp_systolic'))
        session['BP__Diastolic_mmHg'] = float(request.form.get('bp_diastolic'))
        session['RBSmgdl'] = float(request.form.get('blood_glucose'))
        session['Blood_Group'] = request.form.get('blood_group')
        session['Vit_D3_ngmL'] = float(request.form.get('vitamin_d3'))
        return redirect(url_for('step6'))
    return render_template('manual/step5.html', current_step=5, total_steps=8)

# Step 6: Hormonal Test Results
@app.route('/step6', methods=['GET', 'POST'])
def step6():
    if request.method == 'POST':
        session['AMHngmL'] = float(request.form.get('amh'))
        session['Hbgdl'] = float(request.form.get('hb'))
        session['PRGngmL'] = float(request.form.get('prg'))
        session['PRLngmL'] = float(request.form.get('prl'))
        session['FSHmIUmL'] = float(request.form.get('fsh'))
        session['LHmIUmL'] = float(request.form.get('lh'))
        session['FSHLH'] = float(request.form.get('fsh_lh'))
        session['TSH_mIUL'] = float(request.form.get('tsh'))
        session['I___betaHCGmIUmL'] = float(request.form.get('i_beta_hcg'))
        session['II____betaHCGmIUmL'] = float(request.form.get('ii_beta_hcg'))
        return redirect(url_for('step7'))
    return render_template('manual/step6.html', current_step=6, total_steps=8)

# Step 7: Follicular & Ultrasound Details
@app.route('/step7', methods=['GET', 'POST'])
def step7():
    if request.method == 'POST':
        session['Follicle_No_R'] = int(request.form.get('follicle_no_r'))
        session['Follicle_No_L'] = int(request.form.get('follicle_no_l'))
        session['Avg_F_size_R_mm'] = float(request.form.get('avg_f_size_r'))
        session['Avg_F_size_L_mm'] = float(request.form.get('avg_f_size_l'))
        session['CycleRI'] = float(request.form.get('cycle_ri'))
        session['Endometrium_mm'] = float(request.form.get('endometrium'))
        return redirect(url_for('step8'))
    return render_template('manual/step7.html', current_step=7, total_steps=8)

# Step 8: Review & Confirm
@app.route('/step8', methods=['GET', 'POST'])
def step8():
    if request.method == 'POST':
        return redirect(url_for('summary'))
    return render_template('manual/step8.html', current_step=8, total_steps=8)

# Summary page
@app.route('/summary')
def summary():
    return render_template('summary.html', session=session, feature_names=feature_names)

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
                user_input.append(0.0)
        else:
            user_input.append(0.0)

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

# Upload Report (Placeholder)
@app.route('/upload', methods=['GET', 'POST'])
def upload_report():
    if request.method == 'POST':
        session['file_data'] = request.files['report_file']
        return redirect(url_for('summary'))
    return render_template('upload_report.html')

if __name__ == '__main__':
    app.run(debug=True)
