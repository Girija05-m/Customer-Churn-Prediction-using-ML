import streamlit as st
import pandas as pd
import joblib

# ================= Page Config =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# ================= Simple Gradient Background =================
st.markdown("""
<style>
/* Background Gradient: Purple -> Blue */
.stApp {
    background: linear-gradient(to right, #00d2ff, #3a7bd5);

}

/* Button Styling */
div.stButton > button {
    background-color: #6a11cb;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #2575fc;
    color: white;
}

/* Input labels color */
label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ================= Load Model =================
model = joblib.load("rf_churn_model.pkl")

# ================= Title =================
st.title("📊 Customer Churn Prediction")
st.markdown("### Predict if a telecom customer is likely to churn")
st.markdown("---")

# ================= Customer Profile =================
st.subheader("👤 Customer Profile")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", 0, 72)
    monthly_charges = st.number_input("Monthly Charges", 0.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])

with col2:
    total_charges = st.number_input("Total Charges", 0.0)
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

# ================= Services =================
st.subheader("🌐 Services")
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ================= Billing =================
st.subheader("💳 Billing")
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ================= Feature Engineering =================
avg_charges = total_charges / (tenure + 1)
is_long_term = 1 if tenure > 24 else 0

# ================= Input DataFrame =================
input_data = pd.DataFrame({
    'tenure':[tenure],
    'MonthlyCharges':[monthly_charges],
    'TotalCharges':[total_charges],
    'gender':[gender],
    'SeniorCitizen':[1 if senior=="Yes" else 0],
    'Partner':[partner],
    'Dependents':[dependents],
    'PhoneService':[phone_service],
    'MultipleLines':[multiple_lines],
    'InternetService':[internet],
    'OnlineSecurity':[online_security],
    'OnlineBackup':[online_backup],
    'DeviceProtection':[device_protection],
    'TechSupport':[tech_support],
    'StreamingTV':[streaming_tv],
    'StreamingMovies':[streaming_movies],
    'Contract':[contract],
    'PaperlessBilling':[paperless_billing],
    'PaymentMethod':[payment],
    'AvgCharges':[avg_charges],
    'Is_Long_Term':[is_long_term]
})

# ================= Prediction =================
st.markdown("## 🔍 Prediction")
if st.button("Predict Customer Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ Likely to CHURN (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Not likely to churn (Probability: {probability:.2f})")
