import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import the engine
from main import generate_guide_from_rag


load_dotenv()

app = Flask(__name__)
CORS(app)

# CONFIGURATION
IMAGE_FOLDER = os.path.abspath("./extracted_images")
DEFAULT_MODE = os.getenv("LLM_MODE", "CLOUD").upper()


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    mode = data.get("mode", DEFAULT_MODE).upper()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Strict Query Validation: Require a model number
    # Looks for typical Samsung model patterns (e.g., WA5471ABP, WW90T504DAN)
    model_pattern = r"\b(?:WA|WW|WD|WF|DC)\d+[A-Za-z0-9]*\b"
    if not re.search(model_pattern, user_query, re.IGNORECASE):
        return jsonify(
            {
                "status": "error",
                "message": "Please specify the exact model number (e.g., WA5471ABP or WW90T504DAN) to retrieve accurate technician steps.",
            }
        ), 400

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
    app.run(debug=False, port=5000)
