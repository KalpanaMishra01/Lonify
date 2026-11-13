import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
model = joblib.load("loan_modell.pkl")
preprocessing = joblib.load("preprocessing.pkl")

# Page configuration
st.set_page_config(
    page_title="Lonify - Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("Lonify 🏦")
st.subheader("Loan Approval Prediction System")

st.write(
    "Enter applicant details to predict loan approval."
)

# Applicant Details Form

st.header("Applicant Information")

with st.form("loan_form"):

    st.subheader("Financial Details")

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=50000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=18, value=25)
    dependents = st.number_input("Dependents", min_value=0, value=0)
    credit_score = st.number_input("Credit Score", min_value=0, value=650)
    existing_loans = st.number_input("Existing Loans", min_value=0, value=0)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.3)
    savings = st.number_input("Savings", min_value=0.0, value=50000.0)
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
    loan_term = st.number_input("Loan Term", min_value=1, value=12)

    st.subheader("Personal Details")

    education = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Business", "Car", "Education", "Home", "Personal"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    employment = st.selectbox(
        "Employment Status",
        ["Contract", "Salaried", "Self-employed", "Unemployed"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Business", "Government", "MNC", "Private", "Unemployed"]
    )

    submit = st.form_submit_button("Predict Loan Approval")


if submit:

    input_data = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education,
        "Loan_Purpose": loan_purpose,
        "Marital_Status": marital_status,
        "Employment_Status": employment,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category
    }])

    # Encode Education Level
    input_data["Education_Level"] = (
        preprocessing["education_encoder"]
        .transform(input_data["Education_Level"])
    )
    
    # One-Hot Encoding
    hot_cols = [
        "Loan_Purpose",
        "Marital_Status",
        "Employment_Status",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]
    
    encoded_values = preprocessing["onehot_encoder"].transform(
        input_data[hot_cols]
    )
    
    encoded_df = pd.DataFrame(
        encoded_values,
        columns=preprocessing["onehot_encoder"].get_feature_names_out(hot_cols)
    )
    
    # Combine encoded data
    input_data = pd.concat(
        [input_data.drop(columns=hot_cols), encoded_df],
        axis=1
    )
    
    # Arrange features in training order
    input_data = input_data[preprocessing["feature_columns"]]
    
    # Scale input data
    input_scaled = preprocessing["scaler"].transform(input_data)
    
    st.write("Applicant details preprocessed successfully.")

    # Make Prediction
    prediction = model.predict(input_scaled)

    # Display Result
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected!")


