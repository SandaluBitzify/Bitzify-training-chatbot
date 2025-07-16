# gemini_helper.py

import google.generativeai as genai
import os

# Replace this with your actual key (keep secret in .env for real apps)
genai.configure(api_key="AIzaSyA9WnVoF44HjrnkaEHu5INDCddGSr6fKdc")

model = genai.GenerativeModel("gemini-pro")

def extract_prediction_inputs_from_text(prompt):
    full_prompt = f"""
You are a smart assistant. Extract the following fields from the user's message and respond ONLY with a JSON object containing these fields:
- date
- store_id
- product_id
- category
- region
- price
- discount
- competitor_price
- weather
- holiday_promotion
- seasonality.

Input:
\"\"\"{prompt}\"\"\"

Respond ONLY with a JSON object. Do not include any explanations or additional text.
"""
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return {"error": str(e)}
