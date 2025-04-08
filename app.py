import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("ML Model Deployment with Streamlit")

# Example input — adjust depending on your model's input features
st.write("### Enter input for prediction")

# For demo, we’ll assume a single numeric input
input_value = st.number_input("Enter a value", step=0.01)

if st.button("Predict"):
    prediction = model.predict([[input_value]])
    st.success(f"Prediction: {prediction[0]}")
