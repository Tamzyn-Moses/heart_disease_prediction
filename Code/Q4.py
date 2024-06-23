import streamlit as st
import pandas as pd
from joblib import load
import os

model_file = 'best_model_heart_disease_prediction.joblib'

if not os.path.exists(model_file):
    st.error(f"File '{model_file}' not found")
else:
    model = load(model_file)

    st.title('Heart Disease Prediction')

    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex 1: Male; 0: Female', [0, 1])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=100)
    chol = st.number_input('Serum Cholesterol in mg/dl', min_value=100, max_value=600, value=250)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results (0-2)', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=100)
    exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment (0-2)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', [1, 2, 3])

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    if st.button('Predict'):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write('Likely to have heart disease.')
        else:
            st.write('Not likely to have heart disease.')
