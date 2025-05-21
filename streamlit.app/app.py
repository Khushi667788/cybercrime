import streamlit as st
import pandas as pd
import joblib

st.title("🔐 Cyber Crime Predictor")

model = joblib.load("../models/rf_model.pkl")
df = pd.read_csv("../data/cybercrime_data.csv")

user_input = st.text_input("Enter Crime Description:")
# Simple NLP preprocessing logic or manual input mapping

if st.button("Predict Crime Type"):
    # Placeholder prediction
    st.success("Predicted Crime Type: Cyber Fraud")
