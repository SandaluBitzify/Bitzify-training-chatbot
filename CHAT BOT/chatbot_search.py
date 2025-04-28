# 2_chatbot_search.py

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load everything
sentences = pickle.load(open('sentences.pkl', 'rb'))
embeddings = np.load('embeddings.npy')
index = faiss.read_index('index.faiss')
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_answer(question, top_k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    
    answers = []
    for idx in indices[0]:
        answers.append(sentences[idx])
    return answers

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk your question: ")
        results = search_answer(user_query)
        for i, res in enumerate(results):
            print(f"\nAnswer {i+1}: {res}")
