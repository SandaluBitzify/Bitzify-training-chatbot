from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
from llm_extractor_transformers import extract_structured_data

app = Flask(__name__)

# ✅ Load Linear Regression model & encoders
forecast_model = joblib.load('demand_forecast_linear.pkl')
encoders = {name: joblib.load(f'{name}_encoder.pkl') for name in [
    'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'
]}

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_text = data.get('question')

        if not user_text:
            return jsonify({'error': 'No input provided'}), 400

        user_input = extract_structured_data(user_text)

        # Fill missing fields with defaults
        for key, default in {
            'store_id': 'S001',
            'product_id': 'P0001',
            'competitor_price': 45,
            'holiday_promotion': 0,
            'seasonality': 'Summer'
        }.items():
            if user_input.get(key) is None:
                user_input[key] = default

        date = pd.to_datetime(user_input['date'])
        features = [
            encoders['Store ID'].transform([user_input['store_id']])[0],
            encoders['Product ID'].transform([user_input['product_id']])[0],
            encoders['Category'].transform([user_input['category']])[0],
            encoders['Region'].transform([user_input['region']])[0],
            float(user_input['price']),
            float(user_input['discount']),
            float(user_input['competitor_price']),
            encoders['Weather Condition'].transform([user_input['weather']])[0],
            int(user_input['holiday_promotion']),
            encoders['Seasonality'].transform([user_input['seasonality']])[0],
            date.day, date.month, date.weekday()
        ]

        prediction = forecast_model.predict([features])[0]
        return jsonify({'predicted_demand_forecast': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
