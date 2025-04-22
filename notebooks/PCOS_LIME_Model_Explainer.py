
# PCOS Prediction and Explanation using Ensemble Model (CatBoost + RandomForest) with LIME

## 📦 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import joblib

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --------------------------------------------
## 📂 Load and Preprocess the Dataset
df = pd.read_csv('../data/PCOS_data.csv')

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace(r'[^\w]', '', regex=True)
)

# Drop unnecessary columns
df.drop(columns=['Sl_No', 'Patient_File_No', 'Unnamed_44'], errors='ignore', inplace=True)
df['II____betaHCGmIUmL'] = pd.to_numeric(df['II____betaHCGmIUmL'], errors='coerce')

# Convert all other columns to numeric
for col in df.columns:
    if col != 'PCOS_YN':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace Y/N with binary
yn_cols = ['PregnantYN', 'Weight_gainYN', 'hair_growthYN', 'Skin_darkening_YN',
           'Hair_lossYN', 'PimplesYN', 'Fast_food_YN', 'RegExerciseYN']
df[yn_cols] = df[yn_cols].replace({'Y': 1, 'N': 0})

# Replace 0s with NaN for numeric columns, then fill with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('PCOS_YN')
df[num_cols] = df[num_cols].replace(0, np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

# Split features and target
X = df.drop('PCOS_YN', axis=1)
y = df['PCOS_YN']

# --------------------------------------------
## ✂️ Train-Test Split and Resampling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE before scaling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
## 🤖 Model Training: CatBoost + Random Forest Ensemble

catboost_model = CatBoostClassifier(
    iterations=141,
    depth=8,
    learning_rate=0.0533,
    l2_leaf_reg=9,
    border_count=72,
    loss_function='Logloss',
    verbose=0
)
catboost_model.fit(X_train_resampled, y_train_resampled)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

ensemble_model = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('random_forest', rf_model)
    ],
    voting='soft',
    weights=[2, 1]
)
ensemble_model.fit(X_train_resampled, y_train_resampled)

# --------------------------------------------
## 💾 Save Models and Scaler for App Integration

joblib.dump(ensemble_model, 'pcos_ensemble_model.pkl')
joblib.dump(scaler, 'pcos_scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

# --------------------------------------------
## 🧠 LIME Explanation Setup

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_resampled,
    feature_names=X.columns.tolist(),
    class_names=['No PCOS', 'PCOS'],
    mode='classification',
    discretize_continuous=True
)

def explain_pcos_risk(input_array, model, explainer, feature_names, top_n=5):
    prediction_proba = model.predict_proba([input_array])[0]
    predicted_class = int(np.argmax(prediction_proba))
    predicted_score = prediction_proba[1] * 100

    explanation = explainer.explain_instance(input_array, lambda x: model.predict_proba(x), num_features=top_n)
    feature_weights = explanation.as_list(label=1)
    total = sum(abs(w) for _, w in feature_weights)
    contributions = [(f, round(abs(w) / total * 100, 2)) for f, w in feature_weights]
    contributions = sorted(contributions, key=lambda x: x[1], reverse=True)[:top_n]

    return predicted_class, round(predicted_score, 2), contributions

def plot_feature_impact(contributions):
    features, impacts = zip(*contributions)
    plt.figure(figsize=(8, 5))
    plt.barh(features, impacts, color='teal')
    plt.xlabel('% Contribution')
    plt.title('Top Contributing Features (LIME)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --------------------------------------------
## 🔍 Example Explanation for a Test Instance

instance = X_test_scaled[0]
predicted_class, risk_score, top_features = explain_pcos_risk(instance, ensemble_model, explainer, X.columns.tolist())

print(f"Prediction: {'PCOS' if predicted_class == 1 else 'No PCOS'} ({risk_score}%)")
print("Top Contributing Features:")
for feature, weight in top_features:
    print(f"- {feature}: {weight}%")

plot_feature_impact(top_features)
