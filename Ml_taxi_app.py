import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Title
st.title("PragyanAI Taxi Fare Prediction App (End-to-End ML)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("taxis.csv")
    df = df.convert_dtypes()
    st.write(df.head())
    return df

df = load_data()

st.subheader("PragyanAI Dataset Preview")

# Data cleaning
df = df[['distance', 'fare']].dropna()
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')

# Features & target
X = df[['distance']]
y = df['fare']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Performance")
st.write(f"R2 Score: {r2:.2f}")
st.write(f"RMSE: {rmse:.2f}")

# User input
st.subheader("🚖 Enter Trip Details")

distance = st.number_input("Step 1: Enter Distance (km)", min_value=0.0, value=5.0)

passengers = st.number_input("Step 2: Number of Passengers", min_value=1, value=1)

hour = st.number_input("Step 3: Hour of Day (0–23)", min_value=0, max_value=23, value=12)

if st.button("Predict Fare"):
    prediction = model.predict([[distance]])
    st.success(f"🚖 Estimated Fare: {prediction[0]:.2f}")

# Built-in chart (instead of matplotlib)
st.subheader("📈 Distance vs Fare")
st.scatter_chart(df, x='distance', y='fare')
