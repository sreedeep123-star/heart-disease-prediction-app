import streamlit as st
import joblib
import numpy as np
from tensorflow import keras

st.set_page_config(page_title="❤️ Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("Heart Disease Prediction using ML & Deep Learning")

# Load preprocessing and models
imp, scaler, num_cols = joblib.load("models/preprocess.joblib")
lr_model = joblib.load("models/lr_model.joblib")
rf_model = joblib.load("models/rf_model.joblib")
nn_model = keras.models.load_model("models/nn_model.keras")

st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Logistic Regression", "Random Forest", "Neural Network"])

st.header("🔢 Enter Patient Details")
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0=normal, 1=fixed defect, 2=reversible defect)", [0, 1, 2])

X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]])
X = imp.transform(X)
X = scaler.transform(X)

if st.button("Predict"):
    if model_choice == "Logistic Regression":
        pred = lr_model.predict(X)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(X)[0]
    else:
        pred = (nn_model.predict(X) > 0.5).astype("int32")[0][0]

    st.subheader("✅ Result:")
    st.write("Heart Disease Detected 💔" if pred == 1 else "No Heart Disease ❤️")
