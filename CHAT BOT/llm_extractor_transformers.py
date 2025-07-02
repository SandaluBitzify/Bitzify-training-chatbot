from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import json
from datetime import datetime
import spacy

# Download: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

class LocalDataExtractor:
    def __init__(self):
        # Initialize NER pipeline for extracting entities
        self.ner_pipeline = pipeline("ner", 
                                   model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                   aggregation_strategy="simple")
        
        # Pre-defined mappings
        self.categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports', 'Books']
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        self.weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Snowy', 'Windy']
        self.seasonality = ['Spring', 'Summer', 'Fall', 'Winter']
        
    def extract_numbers(self, text):
        """Extract numbers from text"""
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        return [float(n) for n in numbers]
    
    def extract_date(self, text):
        """Extract date from text"""
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                try:
                    # Try to parse the date
                    date_str = ent.text
                    # Handle various date formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y']:
                        try:
                            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                        except:
                            continue
                except:
                    pass
        return datetime.now().strftime('%Y-%m-%d')  # Default to today
    
    def find_best_match(self, text, options):
        """Find best matching option from a list"""
        text_lower = text.lower()
        for option in options:
            if option.lower() in text_lower:
                return option
        return options[0]  # Default to first option
    
    def extract_structured_data(self, user_text):
        """Extract structured data from user text"""
        
        # Extract entities using NER
        entities = self.ner_pipeline(user_text)
        
        # Extract numbers (prices, discounts, etc.)
        numbers = self.extract_numbers(user_text)
        
        # Extract date
        date = self.extract_date(user_text)
        
        # Extract category
        category = self.find_best_match(user_text, self.categories)
        
        # Extract region
        region = self.find_best_match(user_text, self.regions)
        
        # Extract weather
        weather = self.find_best_match(user_text, self.weather_conditions)
        
        # Extract seasonality
        seasonality = self.find_best_match(user_text, self.seasonality)
        
        # Extract prices and discounts from numbers
        price = numbers[0] if len(numbers) > 0 else 45
        discount = numbers[1] if len(numbers) > 1 else 10
        competitor_price = numbers[2] if len(numbers) > 2 else 43
        
        # Check for promotion/holiday keywords
        holiday_promotion = 1 if any(word in user_text.lower() for word in 
                                   ['holiday', 'promotion', 'sale', 'offer', 'deal']) else 0
        
        # Extract store and product IDs from entities or use defaults
        store_id = "S001"
        product_id = "P0001"
        
        for entity in entities:
            if entity['entity_group'] == 'MISC' and 'S' in entity['word']:
                store_id = entity['word']
            elif entity['entity_group'] == 'MISC' and 'P' in entity['word']:
                product_id = entity['word']
        
        return {
            "date": date,
            "store_id": store_id,
            "product_id": product_id,
            "category": category,
            "region": region,
            "price": price,
            "discount": discount,
            "competitor_price": competitor_price,
            "weather": weather,
            "holiday_promotion": holiday_promotion,
            "seasonality": seasonality
        }

# Usage function for your app.py
def extract_structured_data(user_text):
    extractor = LocalDataExtractor()
    return extractor.extract_structured_data(user_text)