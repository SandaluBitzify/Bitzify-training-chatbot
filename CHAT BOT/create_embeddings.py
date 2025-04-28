# 1_create_embeddings.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


df = pd.read_csv('retail_store_inventory.csv') 

def row_to_text(row):
    return (
        f"On {row['Date']}, Store {row['Store ID']} had Product {row['Product ID']} "
        f"in {row['Category']} category located in {row['Region']} region. "
        f"The inventory level was {row['Inventory Level']}, units sold were {row['Units Sold']}, "
        f"units ordered were {row['Units Ordered']}, and demand forecast was {row['Demand Forecast']}. "
        f"The price was {row['Price']} with a discount of {row['Discount']}%. "
        f"Weather condition was {row['Weather Condition']}, holiday or promotion: {row['Holiday/Promotion']}, "
        f"competitor pricing: {row['Competitor Pricing']}, during {row['Seasonality']} season."
    )

sentences = df.apply(row_to_text, axis=1).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(sentences, show_progress_bar=True)

with open('sentences.pkl', 'wb') as f:
    pickle.dump(sentences, f)

np.save('embeddings.npy', embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, 'index.faiss')

print("✅ Embeddings and FAISS index saved successfully!")
