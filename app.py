import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="AI Drug Recommendation System", page_icon="🧬")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
encoders_path = os.path.join(BASE_DIR, "encoders.pkl")

st.write("Current Directory:", os.getcwd())
st.write("App Directory:", BASE_DIR)
st.write("Files in app folder:", os.listdir(BASE_DIR))

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoders_path, "rb") as f:
    le_sex, le_bp, le_chol, le_drug = pickle.load(f)

st.title("🧬 AI-based Drug Recommendation System")
st.write("Enter patient details to predict the most suitable drug type.")

age = st.slider("Age", min_value=1, max_value=100, value=30)
sex_input = st.selectbox("Sex", ["F", "M"])
bp_input = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
chol_input = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.number_input("Na_to_K Ratio", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

if st.button("Predict Drug"):
    sex = le_sex.transform([sex_input])[0]
    bp = le_bp.transform([bp_input])[0]
    chol = le_chol.transform([chol_input])[0]

    sample_df = pd.DataFrame(
        [[age, sex, bp, chol, na_to_k]],
        columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    )

    prediction = model.predict(sample_df)
    predicted_drug = le_drug.inverse_transform(prediction)[0]

    st.success(f"Predicted Drug Type: {predicted_drug}")