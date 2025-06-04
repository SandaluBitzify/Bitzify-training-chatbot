from flask import Flask, request, jsonify, send_file
from chatbot_search import search_answer as basic_search
from smarter_search import search_answer as smart_search
import joblib
import numpy as np
import pandas as pd
import re
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

# ✅ Text parser for natural language input
def parse_prediction_input(text):
    pattern_map = {
        'date': r'date is (\d{4}-\d{2}-\d{2})',
        'store_id': r'store id is (\w+)',
        'product_id': r'product_id is (\w+)',
        'category': r'category is (\w+)',
        'region': r'region is (\w+)',
        'price': r'price is ([\d.]+)',
        'discount': r'discount is (\d+)',
        'competitor_price': r'competitor price is ([\d.]+)',
        'weather': r'weather is (\w+)',
        'holiday_promotion': r'holiday promotion is (\d+)',
        'seasonality': r'seasonality is (\w+)',
    }

    extracted = {}
    for key, pattern in pattern_map.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1)

    # Convert numeric fields
    for num_key in ['price', 'discount', 'competitor_price', 'holiday_promotion']:
        if num_key in extracted:
            extracted[num_key] = float(extracted[num_key]) if '.' in extracted[num_key] else int(extracted[num_key])

    return extracted

# ✅ Main chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    mode = data.get('mode', 'smart')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # 🔮 Prediction section
    if any(word in question.lower() for word in ['predict', 'forecast', 'estimate']):
        try:
            user_input = data.get('inputs')
            if not user_input:
                user_input = parse_prediction_input(question)
            if not user_input:
                return jsonify({'error': '❌ Could not extract prediction inputs from question.'})

            # ✅ Check for missing fields
            required_keys = ['date', 'store_id', 'product_id', 'category', 'region',
                             'price', 'discount', 'competitor_price', 'weather',
                             'holiday_promotion', 'seasonality']
            missing_keys = [k for k in required_keys if k not in user_input]
            if missing_keys:
                return jsonify({'error': f'❌ Missing input fields: {", ".join(missing_keys)}'})

            # Encode and prepare features
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

    # 🔍 Search section
    try:
        results = smart_search(question) if mode == 'smart' else basic_search(question)
        if not results:
            return jsonify({'message': '❌ No matching answer found.'})
        return jsonify({'answers': results})
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'})

# ✅ Serve chatbot frontend (optional)
@app.route('/')
def index():
    return send_file('chat.html')

# ✅ Run the server
if __name__ == '__main__':
    app.run(port=5000, debug=True)
