import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import the consolidated engine
from main import generate_guide_from_rag

load_dotenv()

app = Flask(__name__)
CORS(app)

# CONFIGURATION
IMAGE_FOLDER = os.path.abspath("./final_cleaned_dataset")

# Default Mode from .env (fallback to CLOUD if not set)
# Add "LLM_MODE=LOCAL" to your .env file to change default
DEFAULT_MODE = os.getenv("LLM_MODE", "CLOUD").upper()


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    user_query = data.get("query")

    # Allow Frontend to override mode, otherwise use default
    # Example JSON: { "query": "Help me", "mode": "LOCAL" }
    mode = data.get("mode", DEFAULT_MODE).upper()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    if mode not in ["CLOUD", "LOCAL"]:
        return jsonify({"error": "Invalid Mode. Use 'CLOUD' or 'LOCAL'"}), 400

    try:
        print(f"--- 📨 Incoming Request: '{user_query}' [Mode: {mode}] ---")
        result = generate_guide_from_rag(user_query, mode=mode)
        return jsonify(result)
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/final_cleaned_dataset/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


if __name__ == "__main__":
    print(f"--- 🚀 Server running on http://localhost:5000 ---")
    print(f"--- ⚙️  Default LLM Mode: {DEFAULT_MODE} ---")
    app.run(debug=True, port=5000)
