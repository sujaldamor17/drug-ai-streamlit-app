import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="AI Drug Recommendation System", page_icon="🧬")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🧬 AI-based Drug Recommendation System")
st.write("Enter patient details to predict the most suitable drug type.")

age = st.slider("Age", min_value=1, max_value=100, value=30)
sex_input = st.selectbox("Sex", ["F", "M"])
bp_input = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
chol_input = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.number_input("Na_to_K Ratio", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

# manual mappings
sex_map = {"F": 0, "M": 1}
bp_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
chol_map = {"HIGH": 0, "NORMAL": 1}

# IMPORTANT:
# Check your drug class order from training
drug_map_reverse = {
    0: "DrugA",
    1: "DrugB",
    2: "DrugC",
    3: "DrugX",
    4: "DrugY"
}

if st.button("Predict Drug"):
    sex = sex_map[sex_input]
    bp = bp_map[bp_input]
    chol = chol_map[chol_input]

    sample_df = pd.DataFrame(
        [[age, sex, bp, chol, na_to_k]],
        columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    )

    prediction = model.predict(sample_df)[0]
    predicted_drug = drug_map_reverse[prediction]

    st.success(f"Predicted Drug Type: {predicted_drug}")