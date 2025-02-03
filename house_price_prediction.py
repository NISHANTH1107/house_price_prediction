import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("india_housing_prices.csv")
df = pd.DataFrame(data)

# Encode categorical variables
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le_dict[col] = LabelEncoder()
        df[col] = le_dict[col].fit_transform(df[col])

# Define input (X) and target (y) variables
X = df.drop("Price_in_Lakhs", axis=1)
y = df["Price_in_Lakhs"]

# Train model
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, 'linear_regression_model.pkl')