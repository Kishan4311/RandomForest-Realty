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
    # 👉 MODIFY THESE FEATURES BASED ON YOUR MODEL
    area = st.sidebar.number_input("Area (sq ft)", 100, 10000, 1000)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
    
    location = st.sidebar.selectbox(
        "Location",
        ["Location_A", "Location_B", "Location_C"]
    )

    # Example encoding (modify as needed)
    location_map = {
        "Location_A": 0,
        "Location_B": 1,
        "Location_C": 2
    }

    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location_map[location]
    }

    return pd.DataFrame([input_dict])

input_df = user_input()

# -------------------------------
# PREDICTION
# -------------------------------
st.subheader("📊 Prediction")

if st.button("Predict Price"):
    with st.spinner("Predicting..."):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")
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
        st.metric("Min Price", f"₹ {data.iloc[:, -1].min():,.0f}")
    with col2:
        st.metric("Max Price", f"₹ {data.iloc[:, -1].max():,.0f}")
    with col3:
        st.metric("Avg Price", f"₹ {data.iloc[:, -1].mean():,.0f}")

    # -------------------------------
    # PRICE DISTRIBUTION
    # -------------------------------
    st.subheader("Price Distribution")

    fig, ax = plt.subplots()
    ax.hist(data.iloc[:, -1], bins=30)
    ax.set_title("Price Distribution")
    st.pyplot(fig)

    # -------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")

        importances = model.feature_importances_
        features = input_df.columns

        fig2, ax2 = plt.subplots()
        ax2.barh(features, importances)
        ax2.set_title("Feature Importance")
        st.pyplot(fig2)

else:
    st.info("Upload a dataset to see insights")

# -------------------------------
# CSV UPLOAD FOR BULK PREDICTION
# -------------------------------
st.markdown("---")
st.subheader("📂 Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Predict Bulk Data"):
            preds = model.predict(df)
            df["Predicted Price"] = preds

            st.success("Predictions Completed")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
