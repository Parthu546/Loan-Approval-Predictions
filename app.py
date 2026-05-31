import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Loan Prediction App", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏦 Loan Approval System</h1>", unsafe_allow_html=True)

# -------------------------------
# Instructions
# -------------------------------
st.markdown("""
### 📌 Instructions
- Enter all values in **Indian Rupees (₹)**
- Loan Amount should be total amount (e.g., ₹5,00,000)
- Select Loan Term in **Years or Months**
- Credit History:  
  - 1 → Good history  
  - 0 → Poor history  
""")

st.markdown("---")

# -------------------------------
# Load Data
# -------------------------------
data = pd.read_csv("train.csv")

data.ffill(inplace=True)
data.bfill(inplace=True)

data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# -------------------------------
# Applicant Name
# -------------------------------
st.subheader("👤 Applicant Information")

name = st.text_input("Enter Applicant Name")

st.markdown("---")

# -------------------------------
# Input Fields
# -------------------------------
st.subheader("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    app_income = st.number_input("Applicant Income (₹)", min_value=0)
    co_income = st.number_input("Coapplicant Income (₹)", min_value=0)
    loan_amount_rupees = st.number_input("Loan Amount (₹)", min_value=0)

    term_type = st.radio("Loan Term Type", ["Years", "Months"])

    if term_type == "Years":
        loan_term_input = st.number_input("Loan Term (Years)", min_value=0)
        loan_term = loan_term_input * 12
    else:
        loan_term = st.number_input("Loan Term (Months)", min_value=0)

    credit_history = st.selectbox("Credit History", [0, 1])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# -------------------------------
# Convert Inputs
# -------------------------------
loan_amount = loan_amount_rupees / 1000

input_data = pd.DataFrame([[
    1 if gender == "Male" else 0,
    1 if married == "Yes" else 0,
    dependents,
    1 if education == "Graduate" else 0,
    1 if self_employed == "Yes" else 0,
    app_income,
    co_income,
    loan_amount,
    loan_term,
    credit_history,
    {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
]], columns=X.columns)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Loan Status"):

    if name == "":
        st.warning("⚠️ Please enter applicant name")
    else:
        input_scaled = scaler.transform(input_data)

        result = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")

        st.subheader(f"Result for {name}")

        if result[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.info(f"Approval Probability: {round(prob * 100, 2)}%")

        if term_type == "Years":
            st.info(f"Loan Term: {loan_term_input} years ({loan_term} months)")
        else:
            st.info(f"Loan Term: {loan_term} months")

        # EMI Calculation
        try:
            P = loan_amount_rupees
            r = 0.08 / 12
            n = loan_term

            emi = (P * r * (1 + r)**n) / ((1 + r)**n - 1)

            st.markdown("### 💰 Estimated EMI")
            st.success(f"₹ {round(emi, 2)} per month")

        except:
            st.warning("Enter valid details for EMI calculation")
