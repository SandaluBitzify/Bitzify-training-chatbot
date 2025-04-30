# app.py

from flask import Flask, request, jsonify
from smarter_search import search_answer  # ✅ Importing smart search function

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    results = search_answer(question)

    if not results:
        return jsonify({'message': '❌ No good answer found. Try a different question.'})
    
    return jsonify({'answers': results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
