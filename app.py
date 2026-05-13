import pandas as pd
import streamlit as st
import os
import pickle
nb_model = pickle.load(open("nb_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
nb_acc = pickle.load(open("nb_acc.pkl", "rb"))
dt_acc = pickle.load(open("dt_acc.pkl", "rb"))
svm_acc = pickle.load(open("svm_acc.pkl", "rb"))

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Bank Customer Churn Predictor")
st.markdown("Enter customer details and get predictions from 3 ML models")

credit_score = st.number_input("Credit Score", 0, 1000, 600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, value=0.0)
num_products = st.number_input("Number of Products", 1, 10, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, value=50000.0)

input_data = pd.DataFrame([[
    credit_score, geography, gender, age, tenure,
    balance, num_products, has_cr_card, is_active, salary
]], columns=[
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary"
])

if st.button("🔮 Predict Churn"):

    nb_pred = nb_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Naive Bayes")
        st.success("Not Churn ✅" if nb_pred == 0 else "Churn ❌")
        st.info(f"Accuracy: {nb_acc*100:.2f}%")

    with col2:
        st.subheader("Decision Tree")
        st.success("Not Churn ✅" if dt_pred == 0 else "Churn ❌")
        st.info(f"Accuracy: {dt_acc*100:.2f}%")

    with col3:
        st.subheader("SVM")
        st.success("Not Churn ✅" if svm_pred == 0 else "Churn ❌")
        st.info(f"Accuracy: {svm_acc*100:.2f}%")



