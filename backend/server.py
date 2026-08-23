import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add backend directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the engine
from main import generate_guide_from_rag, KNOWN_MODELS, extract_model

load_dotenv()
load_dotenv(os.path.join(current_dir, ".env"))

app = Flask(__name__)
CORS(app)

# CONFIGURATION
EXTRACTED_IMAGE_FOLDER = os.path.abspath("./extracted_images")
CLEANED_IMAGE_FOLDER = os.path.abspath("./final_cleaned_dataset")
DEFAULT_MODE = os.getenv("LLM_MODE", "CLOUD").upper()

@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns available supported Samsung washing machine models."""
    return jsonify({
        "status": "success",
        "models": KNOWN_MODELS,
        "default_model": "WA5471ABP"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_query = data.get("query", "").strip()
    model_param = data.get("model", "").strip()
    mode = data.get("mode", DEFAULT_MODE).upper()

    if not user_query:
        return jsonify({"status": "error", "message": "No query provided"}), 400

    # Model resolution: query text or explicit model parameter
    detected_model = extract_model(user_query, model_hint=model_param)
    
    # If no specific model could be detected and query doesn't specify one
    if detected_model == "General" and not data.get("allow_generic", False):
        # Check if user query is too ambiguous
        return jsonify({
            "status": "disambiguation_required",
            "message": "Please specify your Samsung washing machine model (e.g., WA5471ABP, WF5M5100AW, WF350ANR) for accurate grounded steps and diagrams.",
            "available_models": KNOWN_MODELS,
            "detected_model": "General"
        }), 200

    try:
        print(f"--- 📨 Request: '{user_query}' [Model: {detected_model}, Mode: {mode}] ---")
        result = generate_guide_from_rag(user_query, model=detected_model, mode=mode)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/extracted_images/<path:filename>")
def serve_extracted_image(filename):
    return send_from_directory(EXTRACTED_IMAGE_FOLDER, filename)

@app.route("/final_cleaned_dataset/<path:filename>")
def serve_cleaned_image(filename):
    if os.path.exists(os.path.join(CLEANED_IMAGE_FOLDER, filename)):
        return send_from_directory(CLEANED_IMAGE_FOLDER, filename)
    return send_from_directory(EXTRACTED_IMAGE_FOLDER, filename)

if __name__ == "__main__":
    print("--- 🚀 Samsung Prism Grounded RAG Server running on http://localhost:5000 ---")
    app.run(debug=False, port=5000)
