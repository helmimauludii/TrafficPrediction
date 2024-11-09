# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('Data_Trafik_Cicadas_Hourly.csv')
    # Convert 'Date' and 'Time' to datetime if needed
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data = data.drop(['Date', 'Time', 'eNodeB Name', 'Cell Name', 'Integrity'], axis=1)
    return data

data = load_data()

st.title("4G Total Traffic Prediction")

# Data Preprocessing
st.write("Data Overview:")
st.write(data.head())

# Prepare the data for modeling
X = data.drop(['4G Total Traffic (GB)', 'DateTime'], axis=1)
y = data['4G Total Traffic (GB)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# Input form for predictions
st.header("Make a Prediction")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"Input {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Predict
if st.button("Predict 4G Total Traffic"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.write(f"Predicted 4G Total Traffic (GB): {prediction:.2f}")
