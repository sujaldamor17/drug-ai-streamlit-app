import streamlit as st
import pandas as pd
import pickle
import os

# Optional debug
st.write("Current Directory:", os.getcwd())
st.write("Files in folder:", os.listdir())

# Load model and encoders
with open(r"C:\Users\Sujal Damor\OneDrive\Desktop\My career\8 sem project\model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\Sujal Damor\OneDrive\Desktop\My career\8 sem project\encoders.pkl", "rb") as f:
    le_sex, le_bp, le_chol, le_drug = pickle.load(f)

st.set_page_config(page_title="AI Drug Recommendation System", page_icon="🧬")

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