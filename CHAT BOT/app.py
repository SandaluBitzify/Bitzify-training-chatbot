# 3_app.py

from flask import Flask, request, jsonify
from chatbot_search import search_answer  # Import from previous file

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answers = search_answer(question)
    
    return jsonify({'answers': answers})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
