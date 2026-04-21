import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="RandomForest Realty", layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("🏡 RandomForest Realty Dashboard")
st.markdown("Predict real estate prices using Machine Learning")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Dragon.joblib")
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("❌ Model file not found or corrupted. Please upload model.joblib")
    st.stop()

# -------------------------------
# LOAD DATA (OPTIONAL)
# -------------------------------
@st.cache_data
def load_data():
    if os.path.exists("data.csv"):
        return pd.read_csv("data.csv")
    return None

data = load_data()

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🏠 Property Features")

def user_input():

    features = {
        "CRIM": st.sidebar.number_input(
            "CRIM: Crime Rate (per capita)", 0.0, 100.0, 0.1
        ),

        "ZN": st.sidebar.number_input(
            "ZN: Residential Land % (large plots > 25,000 sq.ft.)", 0.0, 100.0, 0.0
        ),

        "INDUS": st.sidebar.number_input(
            "INDUS: Non-Retail Business Area (%)", 0.0, 30.0, 5.0
        ),

        "CHAS": st.sidebar.selectbox(
            "CHAS: Near Charles River?", ["No", "Yes"]
        ),

        "NOX": st.sidebar.number_input(
            "NOX: Air Pollution Level (NOX concentration)", 0.0, 1.0, 0.5
        ),

        "RM": st.sidebar.number_input(
            "RM: Average Number of Rooms", 0.0, 10.0, 5.0
        ),

        "AGE": st.sidebar.number_input(
            "AGE: Old Houses (%) (built before 1940)", 0.0, 100.0, 50.0
        ),

        "DIS": st.sidebar.number_input(
            "DIS: Distance to Employment Centers", 0.0, 20.0, 5.0
        ),

        "RAD": st.sidebar.number_input(
            "RAD: Highway Accessibility Index", 0.0, 25.0, 5.0
        ),

        "TAX": st.sidebar.number_input(
            "TAX: Property Tax Rate", 0.0, 1000.0, 300.0
        ),

        "PTRATIO": st.sidebar.number_input(
            "PTRATI: Student-Teacher Ratio", 0.0, 30.0, 15.0
        ),

        "B": st.sidebar.number_input(
            "B: Diversity Index (B)", 0.0, 400.0, 300.0
        ),

        "LSTAT": st.sidebar.number_input(
            "LSTAT: Lower Income Population (%)", 0.0, 50.0, 10.0
        )
    }

    # Convert CHAS to numeric
    features["CHAS"] = 1 if features["CHAS"] == "Yes" else 0

    feature_order = [
        "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
        "DIS","RAD","TAX","PTRATIO","B","LSTAT"
    ]

    return pd.DataFrame([features])[feature_order]
    
input_df = user_input()

# -------------------------------
# PREPROCESSING
# -------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@st.cache_resource
def get_preprocessing(data):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if data is not None and "MEDV" in data.columns:
        X = data.drop("MEDV", axis=1)
        imputer.fit(X)
        scaler.fit(imputer.transform(X))
        return imputer, scaler
    else:
        return None, None

imputer, scaler = get_preprocessing(data)

# -------------------------------
# PREDICTION
# -------------------------------

st.subheader("📊 Prediction")

if st.button("Predict Price"):
    with st.spinner("Predicting..."):
        try:
            if imputer is None or scaler is None:
                st.error("⚠️ data.csv required for preprocessing")
            else:
                input_processed = imputer.transform(input_df)
                input_scaled = scaler.transform(input_processed)
                prediction = model.predict(input_scaled)[0]
                st.success(f"💰 Estimated Price: ${prediction:,.2f}K")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            

# -------------------------------
# DASHBOARD SECTION
# -------------------------------
st.markdown("---")
st.subheader("📈 Dashboard Insights")

if data is not None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Min Price", f"${data.iloc[:, -1].min():,.0f}K")
    with col2:
        st.metric("Max Price", f"${data.iloc[:, -1].max():,.0f}K")
    with col3:
        st.metric("Avg Price", f"${data.iloc[:, -1].mean():,.0f}K")
    # -------------------------------
    # PRICE DISTRIBUTION
    # -------------------------------
    st.subheader("Price Distribution")

   fig, ax = plt.subplots()

    ax.hist(data.iloc[:, -1], bins=30)

    # Title
    ax.set_title("Distribution of House Prices")

    # X-axis label
    ax.set_xlabel("House Price ($1000)")

    # Y-axis label
    ax.set_ylabel("Number of Houses")

    st.caption("All prices are shown in thousands of USD ($1000)")

    st.pyplot(fig)

   

