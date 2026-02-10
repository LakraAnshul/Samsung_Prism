import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import the engine
from main import generate_guide_from_rag

load_dotenv()

app = Flask(__name__)
CORS(app)

# CONFIGURATION
IMAGE_FOLDER = os.path.abspath("./final_cleaned_dataset")
DEFAULT_MODE = os.getenv("LLM_MODE", "CLOUD").upper()


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    mode = data.get("mode", DEFAULT_MODE).upper()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        print(f"--- 📨 Incoming Request: '{user_query}' [Mode: {mode}] ---")
        result = generate_guide_from_rag(user_query, mode=mode)
        return jsonify(result)
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔴 THIS ROUTE MATCHES THE FRONTEND NOW
@app.route("/final_cleaned_dataset/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


if __name__ == "__main__":
    print(f"--- 🚀 Server running on http://localhost:5000 ---")
    app.run(debug=True, port=5000)
