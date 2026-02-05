import streamlit as st
import joblib
import numpy as np
import os

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model = joblib.load(os.path.join(BASE_DIR, "loan_model.pkl"))

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to check loan eligibility")

st.divider()

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

app_income = st.number_input("Applicant Income", min_value=0.0)
coapp_income = st.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Amount Term (months)", min_value=0.0)

credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Encoding inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Yes" else 0

property_semiurban = 1 if property_area == "Semiurban" else 0
property_urban = 1 if property_area == "Urban" else 0

# Prediction button
if st.button("Predict Loan Eligibility"):
    user_data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        app_income,
        coapp_income,
        loan_amount,
        loan_term,
        credit_history,
        property_semiurban,
        property_urban
    ]])

    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)

    st.divider()

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved\n\nApproval Probability: {probability[0][1]*100:.2f}%")
    else:
        st.error(f"❌ Loan Rejected\n\nRejection Probability: {probability[0][0]*100:.2f}%")