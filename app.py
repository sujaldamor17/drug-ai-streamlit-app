import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Clinical Drug Recommendation System",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ---------------- MANUAL MAPPINGS ----------------
sex_map = {"F": 0, "M": 1}
bp_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
chol_map = {"HIGH": 0, "NORMAL": 1}

drug_map_reverse = {
    0: "DrugA",
    1: "DrugB",
    2: "DrugC",
    3: "DrugX",
    4: "DrugY"
}

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title-box {
    padding: 1.5rem;
    border-radius: 18px;
    background: linear-gradient(90deg, #1e293b, #0f766e);
    color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #111827;
    padding: 1rem;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    border: 1px solid #1f2937;
}
.output-card {
    background: linear-gradient(90deg, #064e3b, #065f46);
    padding: 1.2rem;
    border-radius: 18px;
    color: white;
    font-size: 1.1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.info-card {
    background: #0f172a;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #1e293b;
}
.small-text {
    font-size: 0.9rem;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="title-box">
    <h1>🧬 AI Clinical Drug Recommendation System</h1>
    <p style="font-size:18px; margin-bottom:0;">
        A machine learning-based prototype for predicting the most suitable drug type 
        using patient health parameters.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Patient Parameters")
st.sidebar.write("Enter the patient details below:")

age = st.sidebar.slider("Age", min_value=1, max_value=100, value=30)
sex_input = st.sidebar.selectbox("Sex", ["F", "M"])
bp_input = st.sidebar.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
chol_input = st.sidebar.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.sidebar.slider("Na_to_K Ratio", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

predict_btn = st.sidebar.button("🚀 Predict Drug Recommendation")

# ---------------- TOP METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Age</h4>
        <h2>{age}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Sex</h4>
        <h2>{sex_input}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Blood Pressure</h4>
        <h2>{bp_input}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Na_to_K Ratio</h4>
        <h2>{na_to_k}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- MAIN CONTENT ----------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📋 Patient Summary")
    summary_df = pd.DataFrame({
        "Parameter": ["Age", "Sex", "Blood Pressure", "Cholesterol", "Na_to_K"],
        "Value": [age, sex_input, bp_input, chol_input, na_to_k]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("🧠 Model Overview")
    st.markdown("""
    <div class="info-card">
        This prototype uses a <b>Random Forest Classifier</b> trained on patient clinical parameters
        to recommend the most suitable drug type. The system simulates a basic
        <b>AI-powered clinical decision support tool</b>.
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.subheader("ℹ️ About This Prototype")
    st.markdown("""
    <div class="info-card">
        <b>Project Type:</b> Bioinformatics + AI<br><br>
        <b>Use Case:</b> Drug type recommendation<br><br>
        <b>ML Algorithm:</b> Random Forest<br><br>
        <b>Goal:</b> Demonstrate how AI can assist in personalized treatment decision support.
    </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:
    sex = sex_map[sex_input]
    bp = bp_map[bp_input]
    chol = chol_map[chol_input]

    sample_df = pd.DataFrame(
        [[age, sex, bp, chol, na_to_k]],
        columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    )

    prediction = model.predict(sample_df)[0]
    predicted_drug = drug_map_reverse[prediction]

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("✅ Prediction Result")
    st.markdown(f"""
    <div class="output-card">
        <b>Recommended Drug Type:</b> {predicted_drug}
    </div>
    """, unsafe_allow_html=True)

    # Probability if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample_df)[0]
        prob_df = pd.DataFrame({
            "Drug Type": [drug_map_reverse[i] for i in range(len(probabilities))],
            "Confidence": probabilities
        }).sort_values("Confidence", ascending=False)

        st.subheader("📊 Prediction Confidence")
        st.bar_chart(prob_df.set_index("Drug Type"))

        top_conf = prob_df.iloc[0]["Confidence"] * 100
        st.info(f"Top prediction confidence: **{top_conf:.2f}%**")

    st.subheader("🩺 Clinical Interpretation")
    st.markdown(f"""
    <div class="info-card">
        Based on the entered patient parameters, the model predicts <b>{predicted_drug}</b> 
        as the most suitable drug category. This result is generated using learned patterns
        from the training dataset and serves as a demonstration of AI-assisted decision support.
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="small-text">
<b>Disclaimer:</b> This is an academic machine learning prototype developed for demonstration purposes only. 
It is not intended for real clinical diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)