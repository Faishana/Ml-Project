import streamlit as st
import pickle
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Marine Fish Catch Prediction",
    page_icon="🐟",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        text-align: center;
        color: #003366;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #0066cc;
        font-size: 22px;
        font-weight: bold;
        color: #003366;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = pickle.load(open("rf_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------------- Title ----------------
st.markdown("<div class='title'>🐟 Marine Fish Catch Prediction System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict marine fish catch quantities in Sri Lanka using Machine Learning</div>",
    unsafe_allow_html=True
)

# ---------------- Input Section ----------------
st.subheader("🔢 Input Details")

year = st.number_input(
    "📅 Select Year",
    min_value=2000,
    max_value=2035,
    step=1
)

district_name = st.selectbox(
    "📍 Select Fisheries District",
    encoder.classes_
)

# Encode district
district_encoded = encoder.transform([district_name])[0]

# ---------------- Prediction ----------------
st.markdown("---")

if st.button("🔍 Predict Fish Catch"):
    input_data = pd.DataFrame({
        "Year": [year],
        "District_encoded": [district_encoded]
    })

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"<div class='prediction-box'>"
        f"Predicted Marine Fish Catch<br><br>"
        f"{prediction:,.0f} kg"
        f"</div>",
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<center><small>Developed as part of a Machine Learning Academic Project</small></center>",
    unsafe_allow_html=True
)
