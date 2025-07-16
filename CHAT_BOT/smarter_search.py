# smarter_search.py

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models and data once
sentences = pickle.load(open('sentences.pkl', 'rb'))
embeddings = np.load('embeddings.npy')
index = faiss.read_index('index.faiss')
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_answer(question, top_k=5, min_similarity=0.50):
    # Encode user question
    question_embedding = model.encode([question])
    
    # Search top_k candidates
    distances, indices = index.search(np.array(question_embedding), top_k)
    
    # Now rerank by cosine similarity
    candidate_embeddings = embeddings[indices[0]]
    similarities = cosine_similarity(question_embedding, candidate_embeddings)[0]
    
    results = []
    for idx, score in zip(indices[0], similarities):
        if score >= min_similarity:
            results.append({
                "sentence": sentences[idx],
                "similarity": float(score)
            })

    # Sort by similarity score
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    return results

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk your question: ")
        results = search_answer(user_query)
        if not results:
            print("❌ Sorry, no matching answer found. Try again!")
        else:
            for i, res in enumerate(results):
                print(f"\nAnswer {i+1} (Score: {res['similarity']:.2f}): {res['sentence']}")
