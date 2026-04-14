import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# Load model and scaler
model = keras.models.load_model("heart_model.keras",compile=False)
scaler = joblib.load("scaler.pkl")

# Title
st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# ----------- Input Fields -----------

age = st.number_input("Age", 20, 100)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex == "Female" else 1

cp_list = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
cp = st.selectbox("Chest Pain Type", cp_list)
cp = cp_list.index(cp)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)")

chol = st.number_input("Cholesterol (mg/dl)")

fbs = st.selectbox("Fasting Blood Sugar", ["Normal (<=120)", "High (>120)"])
fbs = 0 if "Normal" in fbs else 1

restecg_list = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
restecg = st.selectbox("Rest ECG", restecg_list)
restecg = restecg_list.index(restecg)

thalach = st.number_input("Maximum Heart Rate Achieved")

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 0 if exang == "No" else 1

oldpeak = st.number_input("Oldpeak (ST depression)")

slope_list = ["Upsloping", "Flat", "Downsloping"]
slope = st.selectbox("Slope", slope_list)
slope = slope_list.index(slope)

ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])

thal_list = ["Normal", "Fixed Defect", "Reversible Defect", "Other"]
thal = st.selectbox("Thal", thal_list)
thal = thal_list.index(thal)

# ----------- Prediction -----------

if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

    # Scale input
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)

    # Output
    if prediction[0][0] > 0.5:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")