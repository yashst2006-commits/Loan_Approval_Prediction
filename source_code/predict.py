import joblib
import os
import numpy as np

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load trained model and scaler
model = joblib.load(os.path.join(BASE_DIR, "loan_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

print("=== Loan Approval Prediction System ===\n")

# Take user input
gender = int(input("Gender (Male=1, Female=0): "))
married = int(input("Married (Yes=1, No=0): "))
dependents = int(input("Dependents (0 / 1 / 2 / 3): "))
education = int(input("Education (Graduate=1, Not Graduate=0): "))
self_employed = int(input("Self Employed (Yes=1, No=0): "))
app_income = float(input("Applicant Income: "))
coapp_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Term (in months): "))
credit_history = int(input("Credit History (1=Yes, 0=No): "))
property_semiurban = int(input("Property Area Semiurban (Yes=1, No=0): "))
property_urban = int(input("Property Area Urban (Yes=1, No=0): "))

# Convert input to numpy array (ORDER MUST MATCH TRAINING)
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

# Scale input
user_data_scaled = scaler.transform(user_data)

# Prediction
prediction = model.predict(user_data_scaled)
probability = model.predict_proba(user_data_scaled)

# Output result
print("\n=== Prediction Result ===")
if prediction[0] == 1:
    print("✅ Loan Approved")
    print(f"Approval Probability: {probability[0][1]*100:.2f}%")
else:
    print("❌ Loan Rejected")
    print(f"Rejection Probability: {probability[0][0]*100:.2f}%")