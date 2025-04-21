import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# --- Load model and scaler ---
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Load dataset (sample input for demo) ---
data = pd.read_csv("telco_data.csv")  # This should be your cleaned dataset (same columns as training)
original_data = data.copy()

# --- UI ---
st.title("📉 Telco Customer Churn Predictor")
st.markdown("Predict whether a customer will churn, and understand **why** with SHAP explanations.")

# --- Select a customer ---
st.sidebar.header("Customer Selection")
customer_index = st.sidebar.selectbox("Choose Customer", data.index)
customer_raw = data.loc[customer_index].copy()
customer_raw = customer_raw.values.reshape(1, -1)

print(customer_raw.shape)  # This should be (1, n_features)


# --- Prepare input for model ---
customer_scaled = scaler.transform(customer_raw)
  # shape: (1, n_features)

# --- Predict ---
prediction = model.predict(customer_scaled)[0]
probability = model.predict_proba(customer_scaled)[0][1]

# --- Display result ---
st.subheader("🔍 Prediction Result")
st.write(f"Prediction: **{'🔴 Will Churn' if prediction == 1 else '🟢 Will Stay'}**")
st.write(f"Probability of Churn: `{probability:.2f}`")

# --- SHAP Explanation ---
st.subheader("🧠 Model Explanation (SHAP)")

# Refit explainer on a subset for performance
explainer = shap.Explainer(model.predict_proba, data, feature_names=data.columns)
shap_values = explainer(customer_raw)

# Plot
st.write("SHAP Values Type:", type(shap_values))
st.write("SHAP Values:", shap_values)

#st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.waterfall(shap_values[0, :, 1], max_display=10)

st.pyplot(bbox_inches="tight")
