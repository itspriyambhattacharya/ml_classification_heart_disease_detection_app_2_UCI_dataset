import streamlit as st
import joblib
import pandas as pd

st.title("Heart Disease Prediction App")
st.write("UCI Dataset")
st.markdown("---")

res = None

# model load


@st.cache_resource
def load_model():
    pipeline = joblib.load('pipeline.pkl')
    col_names = joblib.load('col_names.pkl')
    return pipeline, col_names


pipeline, col_names = load_model()

# app ui

st.markdown("### Enter your data to check your heart condition")

age = st.slider(
    "Age:",
    min_value=18,
    max_value=100,
    value=26
)

col1, col2 = st.columns(2, border=True)

with col1:
    sex = st.selectbox(
        "Sex:",
        ['Male', 'Female']
    )

    cp = st.selectbox(
        "Chest Pain Type", [1, 2, 3, 4]
    )
    trestbps = st.number_input(
        "Resting Blood Pressure in mm Hg:",
        min_value=75
    )
    chol = st.number_input(
        "Serum Cholesterol:",
        min_value=90
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar:",
        ['Greater than 120 mg/dl', 'Less than 120 mg/dl']
    )
    restecg = st.selectbox(
        "Resting ECG Results:",
        ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy']
    )

with col2:
    thalach = st.slider(
        "Maximum Heart Rate achiever:",
        min_value=70,
        max_value=250,
        value=100
    )
    exang = st.selectbox(
        "Exercise Induced Angina:",
        ['Yes', 'No']
    )
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise:",
        min_value=1.4
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment:",
        ['Upsloping', 'Flat', 'Downslopping']
    )
    ca = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy:",
        [0, 1, 2, 3]
    )
    thal = st.selectbox(
        "Thalassemia:",
        ['Normal', 'Fixed Defect', 'Reversal Defect']
    )

_, center, _ = st.columns(3)

with center:
    if st.button("Predict", width='stretch', type='primary'):
        with st.spinner("Predicting ...."):
            sex = 1 if sex == 'Male' else 0

            fbs = 1 if fbs == 'Greater than 120 mg/dl' else 0

            if restecg == 'Normal':
                restecg = 0
            elif restecg == 'ST-T wave abnormality':
                restecg = 1
            else:
                restecg = 2

            exang = 1 if exang == 'Yes' else 0

            if slope == 'Upsloping':
                slope = 1
            elif slope == 'Flat':
                slope = 2
            else:
                slope = 3

            if thal == 'Normal':
                thal = 3
            elif thal == 'Fixed Defect':
                thal = 6
            else:
                thal = 7

            # Prediction
            ns = pd.DataFrame({
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            },
                index=[0]
            )
            ns = ns[col_names]
            output = pipeline.predict(ns)
            res = output[0]
if res == 0:
    st.success("Congrats!! You're healthy")
elif res == 1:
    st.error("You might have heart disease. Consult doctor ASAP.")
