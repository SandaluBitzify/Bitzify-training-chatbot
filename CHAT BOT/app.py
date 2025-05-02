# app.py

from flask import Flask, request, jsonify, send_file
from chatbot_search import search_answer as basic_search
from smarter_search import search_answer as smart_search
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load demand forecast model
forecast_model = joblib.load('demand_forecast_predictor.pkl')

# Load encoders for prediction
encoders = {
    name: joblib.load(f'{name}_encoder.pkl') for name in [
        'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'
    ]
}

# ✅ Route 1: Chatbot Search + Demand Forecast Prediction (unified)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    mode = data.get('mode', 'smart')  # use 'smart' by default

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # 🔮 If it's a prediction-style question
    if any(word in question.lower() for word in ['predict', 'forecast', 'estimate']):
        try:
            user_input = data.get('inputs', {})  # user must send these fields

            # Extract and encode input features
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
            return jsonify({'error': f'Prediction failed: {str(e)}'})

    # 🔍 Else, it's a dataset search question
    try:
        results = smart_search(question) if mode == 'smart' else basic_search(question)
        if not results:
            return jsonify({'message': '❌ No matching answer found.'})
        return jsonify({'answers': results})
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'})

# ✅ Route 2: Serve chatbot frontend UI
@app.route('/')
def index():
    return send_file('chat.html')

# ✅ Start the app
if __name__ == '__main__':
    app.run(port=5000, debug=True)
