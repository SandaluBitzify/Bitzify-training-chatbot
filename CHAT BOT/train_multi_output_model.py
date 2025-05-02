import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv('retail_store_inventory.csv')
df = df.dropna()


df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday


encoders = {}
for col in ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f'{col}_encoder.pkl')


features = ['Store ID', 'Product ID', 'Category', 'Region', 'Price', 'Discount',
            'Competitor Pricing', 'Weather Condition', 'Holiday/Promotion',
            'Seasonality', 'Day', 'Month', 'Weekday']
X = df[features]
y = df['Demand Forecast']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, 'demand_forecast_predictor.pkl')
print("✅ Model trained and saved as 'demand_forecast_predictor.pkl'")
