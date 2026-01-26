import streamlit as st
import pickle
import pandas as pd

# Load trained model and encoder
model = pickle.load(open("rf_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# App title
st.title("Marine Fish Catch Prediction System")
st.write("Predict marine fish catch quantities in Sri Lanka using machine learning.")

# User inputs
year = st.number_input(
    "Select Year",
    min_value=2000,
    max_value=2035,
    step=1
)

district_name = st.selectbox(
    "Select Fisheries District",
    encoder.classes_
)

# Encode district
district_encoded = encoder.transform([district_name])[0]

# Prediction
if st.button("Predict Fish Catch"):
    input_data = pd.DataFrame({
        "Year": [year],
        "District_encoded": [district_encoded]
    })


    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Marine Fish Catch: {prediction:,.0f} kg")
