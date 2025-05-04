#  Femura: PCOS Risk Prediction Using Machine Learning & Explainable AI

**Femura** is an intuitive, web-based predictive tool for identifying early-stage risk of Polycystic Ovary Syndrome (PCOS) using machine learning. It combines a robust ensemble model with human-centered, explainable AI to help users understand their health better and take proactive steps. Whether you're a curious individual, healthcare professional, or researcher — Femura is built with care to make medical AI understandable, responsible, and supportive.

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)
![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Backend-Flask-orange)
![ML](https://img.shields.io/badge/Model-CatBoost%20%7C%20RandomForest-brightgreen)
![Explainable AI](https://img.shields.io/badge/Explainability-LIME%20%7C%20SHAP-purple)
![Open Source](https://img.shields.io/badge/Open--Source-Yes-success)

---

##  Key Features

-  **Risk Prediction**: Uses ensemble learning (CatBoost + Random Forest) to predict PCOS likelihood from user-submitted data.
-  **Human-Friendly Explanations**: Automatically generates simple pie charts and friendly summaries of key contributing factors using LIME.
-  **Advanced Insights Toggle**: Users can switch to see SHAP visualizations (currently under development and testing) and full LIME HTML outputs for transparency.
-  **Preprocessing Pipeline**: Handles categorical encoding, medical NaN imputation, SMOTE balancing, and feature scaling.
-  **Hybrid Feature Selection**: Features chosen based on both medical relevance and ML-based importance scores.
-  **Designed for All**: Designed to Support both everyday users and advanced users like students, doctors, and ML practitioners.
-  **Open Source & Collaborative**: Actively maintained on GitHub and available under a permissive research-friendly license.

---

##  Machine Learning Pipeline

- **Models Used**:  
  - CatBoostClassifier (optimized for tabular data with categorical features)  
  - RandomForestClassifier (robust ensemble baseline)

- **Ensemble Strategy**:  
  - Soft voting (averages the predicted probabilities)

- **Data Preprocessing Steps**:
  1. Drop identifiers, constant, or irrelevant columns
  2. Clean and standardize column names
  3. Encode 'Y'/'N' fields into binary
  4. Convert 0s to `NaN` for medically implausible values
  5. Median imputation of missing values
  6. SMOTE for class balancing
  7. StandardScaler for numerical normalization

- **Explainable AI**:
  - **LIME**: Generates pie chart of top features + natural language summary
  - **SHAP**: Advanced toggle shows SHAP summary plots and LIME HTML output for transparency

---

##  Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla + Chart.js)
- **ML Libraries**: CatBoost, Scikit-learn, imbalanced-learn
- **XAI Tools**: LIME, SHAP
- **Data Handling**: Pandas, NumPy, SMOTE
- **Visualization**: Chart.js (frontend pie chart), Matplotlib/SHAP (backend)
- **Model Deployment**: Locally with Flask (cloud-ready)
- **Development Notebook**: Google Colab

---

##  Setup Instructions

Clone the repository and run it locally:

```bash
git clone https://github.com/sivan73/pcos-prediction.git
cd pcos-prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Flask app:

```bash
cd app
python app.py
```

Open your browser at `http://localhost:5000` to try it out!

---

##  Example Outputs

- Users are shown a simple risk score (e.g., 68% PCOS likelihood).
- A pie chart visualizes top contributing features in layman's terms.
- Sample message:  
  _“Your menstrual irregularities and elevated weight have significantly contributed to your PCOS risk score. Lifestyle changes can still help you reduce risk—consult a professional for further guidance.”_
- Advanced users can toggle to see:
  - SHAP summary bar chart (under development)
  - LIME's full explanation HTML

---

##  Dataset Details

The model was trained on a clinically annotated PCOS dataset featuring hormonal, physiological, and lifestyle parameters. The dataset underwent careful preprocessing and was split using stratified sampling to preserve class balance. Training was done with SMOTE-enhanced data to avoid model bias toward the majority class. This dataset was dowmloaded from Kaggle and sourced orginally from a hospital in Kerala, India.

---

##  Collaboration & License

We used **GitHub** for version control and collaboration throughout this project. The repository is open-source so others can learn from, build upon, or contribute to this tool.

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  
See the LICENSE file for more information.

---

##  Author

**Sivanesan R R**  
📧 [sivan.offcl@gmail.com](mailto:sivan.offcl@gmail.com)  
🔗 [GitHub](https://github.com/sivan73)

---

##  Future Enhancements

- Smartwatch integration (Google Health Connect API)
- Mobile-friendly version of Femura
- Doctor dashboard to view and annotate patient reports
- Expand to cover other early-stage women’s health conditions

---

> _“We believe healthcare AI should be interpretable, inclusive, affordable and empowering. Femura is a small step toward that vision.”_
