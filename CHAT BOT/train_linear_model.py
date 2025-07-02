import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('retail_store_inventory.csv')
df = df.dropna()

# Extract date features
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday

# Encode categorical variables
encoders = {}
for col in ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f'{col}_encoder.pkl')

print("\n📘 Encoded label mappings:")
for col, encoder in encoders.items():
    classes = encoder.classes_
    print(f"\n➡️ {col} Encoding:")
    for idx, label in enumerate(classes):
        print(f"   {label} ➜ {idx}")

# Features and target
features = ['Store ID', 'Product ID', 'Category', 'Region', 'Price', 'Discount',
            'Competitor Pricing', 'Weather Condition', 'Holiday/Promotion',
            'Seasonality', 'Day', 'Month', 'Weekday']
X = df[features]
y = df['Demand Forecast']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'demand_forecast_linear.pkl')
print("✅ Linear Regression model saved as 'demand_forecast_linear.pkl'")

# Print the equation
coefficients = model.coef_
intercept = model.intercept_

print("\n📘 Demand Forecast Equation:")
equation = f"Demand Forecast = {intercept:.2f}"
for feature, coef in zip(features, coefficients):
    equation += f" + ({coef:.2f} * {feature})"
print(equation)
