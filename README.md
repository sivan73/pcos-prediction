#  PCOS Prediction using Machine Learning and Explainable AI

A predictive web-based tool for identifying early-stage risk of Polycystic Ovary Syndrome (PCOS) using CatBoost and XGBoost models, enhanced with SHAP explainability. Built to support proactive health decisions before the onset of symptoms.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey)
![Flask](https://img.shields.io/badge/Backend-Flask-orange)
![ML](https://img.shields.io/badge/Model-CatBoost%20%7C%20XGBoost-brightgreen)
![Explainable AI](https://img.shields.io/badge/Explainability-SHAP-purple)

---

##  Features

- Web app for PCOS risk prediction
- User questionnaire input
- Machine learning models: CatBoost & XGBoost
- Explainable AI (SHAP) for transparency
- Hybrid feature selection (SHAP + clinical signs)
- Future support for wearable device integration

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JS (Vanilla)
- **ML Models**: CatBoost, XGBoost
- **Explainability**: SHAP (Summary & Waterfall plots)
- **Notebook Interface**: Google Colab (for training)
- **Deployment-ready** for local & cloud

---

---

## ⚙️ Setup Instructions

1. **Clone the repo**
   git clone https://github.com/yourusername/pcos-prediction.git
   cd pcos-prediction
2. **Set up virtual environment**
    python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. **Install dependencies**
    pip install -r requirements.txt
4. **Run the app**
    cd app
python app.py

## About the Models
CatBoost & XGBoost: Trained on preprocessed PCOS dataset with hybrid feature selection.

Explainability: Each prediction is explained for the risk score with the major contributing factors and their weightage

Synthetic wearable data (to be added) simulates future integration with real devices.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
See the LICENSE file for details.
## Author

Sivanesan R R
sivan.offcl@gmail.com
[Github](https://github.com/sivan73)

## Future Enhancements

Integration with smartwatches via Google Health Connect API

Support for other early-stage disorder predictions

Doctor interface for receiving and reviewing reports

