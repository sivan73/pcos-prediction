from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
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

# step 1: Age and maritial details
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
        session['Pulse_ratebpm'] = int(request.form.get('pulse_rate'))
        session['RR_breathsmin'] = int(request.form.get('respiratory_rate'))
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
    
    return render_template('manual/step8.html', current_step=8, total_steps=8)

ensemble_model = pickle.load(open('../models/pcos_ensemble_model.pkl', 'rb'))
# Load model and other components
model = pickle.load(open('../models/pcos_ensemble_model.pkl', 'rb'))
scaler = pickle.load(open('../models/pcos_scaler.pkl', 'rb'))
feature_names = pickle.load(open('../models/pcos_feature_names.pkl', 'rb'))
training_data = pickle.load(open('../models/pcos_training_data.pkl', 'rb'))

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('step8'))

    user_input = []
    blood_group_mapping = {
        'A-': 11, 'A+': 12, 'B-': 13, 'B+': 14,
        'O-': 15, 'O+': 16, 'AB-': 17, 'AB+': 18
    }


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
    risk_score = round(model.predict_proba(input_scaled)[0][1] * 100, 2)
    result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"

    # Dynamic phrase based on risk
    if risk_score >= 80:
        dynamic_phrase = "You may be at high risk. Please consult a gynecologist for detailed screening."
    elif risk_score >= 50:
        dynamic_phrase = "You may be at moderate risk. It is advised to monitor your health and consult a doctor."
    else:
        dynamic_phrase = "Your risk appears to be low. Maintain a healthy lifestyle and stay aware."

    # LIME explanation (image)
    explanation = explainer.explain_instance(
        data_row=input_scaled[0],
        predict_fn=model.predict_proba,
        num_features=5
    )
    lime_plot_path = 'lime_plot.png'
    explanation.save_to_file(os.path.join('static', lime_plot_path))
    os.makedirs(os.path.join('static'), exist_ok=True)
    static_dir = os.path.join('static')
    explanation.save_to_file(os.path.join(static_dir, lime_plot_path))

    user_input = [float(x) for x in request.form.values()]
    input_scaled = scaler.transform([user_input])
    # Predict
    prediction = ensemble_model.predict(input_scaled)[0]
    prediction_proba = ensemble_model.predict_proba(input_scaled)[0][1]
    risk_score = round(prediction_proba * 100, 2)

    # Dynamic phrase logic
    if risk_score >= 80:
        dynamic_phrase = "High risk. Please consult a doctor soon."
    elif risk_score >= 50:
        dynamic_phrase = "Moderate risk. Monitoring is advised."
    else:
        dynamic_phrase = "Low risk. Stay healthy!"


    input_data = [float(request.form.get(f)) for f in feature_names]
    user_df = pd.DataFrame([input_data], columns=feature_names)
    scaled_input = scaler.transform(user_df)
    
    # Predict
    prediction_proba = model.predict_proba(scaled_input)[0][1]
    prediction_label = model.predict(scaled_input)[0]
    session['input_data'] = input_data
    session['prediction_proba'] = prediction_proba
    session['prediction_label'] = int(prediction_label)
    

    input_data = session.get('input_data')
    prediction_proba = session.get('prediction_proba')
    prediction_label = session.get('prediction_label')

    if not input_data:
        return redirect(url_for('index'))

    user_df = pd.DataFrame([input_data], columns=feature_names)
    scaled_input = scaler.transform(user_df)

    # SHAP pie chart
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(training_data)
    shap_input_value = explainer.shap_values(scaled_input)[1][0]
    top_indices = np.argsort(np.abs(shap_input_value))[::-1][:5]
    top_features = [feature_names[i] for i in top_indices]
    top_contributions = [shap_input_value[i] for i in top_indices]

    fig, ax = plt.subplots()
    ax.pie(np.abs(top_contributions), labels=top_features, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    pie_chart_path = 'static/images/shap_pie.png'
    os.makedirs('static/images', exist_ok=True)
    plt.savefig(pie_chart_path)
    plt.close()

    # SHAP summary
    shap.summary_plot(shap_values[1], training_data, feature_names=feature_names, show=False)
    summary_path = 'static/images/shap_summary.png'
    plt.savefig(summary_path)
    plt.close()

    # LIME explanation
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data.values, feature_names=feature_names, class_names=['No PCOS', 'PCOS'], discretize_continuous=True)
    lime_exp = lime_explainer.explain_instance(scaled_input[0], model.predict_proba, num_features=5)
    lime_html_path = 'static/lime_explanation.html'
    lime_exp.save_to_file(lime_html_path)

    return render_template('result.html',
                           prediction_proba=round(prediction_proba * 100, 2),
                           prediction_label=prediction_label,
                           pie_chart=pie_chart_path,
                           summary_plot=summary_path,
                           lime_html=lime_html_path)




# Upload Report (Placeholder)
@app.route('/upload', methods=['GET', 'POST'])
def upload_report():
    if request.method == 'POST':
        session['file_data'] = request.files['report_file']
        return redirect(url_for('summary'))
    return render_template('upload_report.html')

if __name__ == '__main__':
    app.run(debug=True)
