import google.generativeai as genai
import json

# ✅ Setup API Key
genai.configure(api_key="AIzaSyBNWY0isNHJhRSEtV-kdF3gHJ7ciGKYDck")

# ✅ Use correct full model name
MODEL_NAME = "models/gemini-1.5-flash-latest"  # or "models/gemini-1.5-pro-latest" if you have access

def extract_structured_data(user_text):
    system_prompt = """
You are a smart data extractor. Convert user's sentence into valid JSON using this format:

{
  "date": "YYYY-MM-DD",
  "store_id": "S001",
  "product_id": "P0001",
  "category": "Electronics",
  "region": "South",
  "price": 45,
  "discount": 10,
  "competitor_price": 43,
  "weather": "Rainy",
  "holiday_promotion": 0,
  "seasonality": "Summer"
}

If any field is missing, set it as null. ONLY return valid JSON. No explanation.
"""

    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(system_prompt + "\n\nUser input: " + user_text)

    extracted_json = response.text

    extracted_json = extracted_json.strip().replace("```json", "").replace("```", "").strip()

    parsed = json.loads(extracted_json)
    return parsed
