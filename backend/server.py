"""
Guide Weave — Flask API Server
HTTP/API layer for the Samsung Prism Grounded RAG system.

Responsibilities:
    - HTTP request/response handling
    - Three-state model request routing
    - Secure image serving with path traversal prevention
    - Response formatting
    - Health endpoint
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path so backend.* and scripts.* imports work
project_root = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the orchestrator and model resolver
from backend.main import generate_guide_from_rag
from backend.model_resolver import get_available_database_models, resolve_model_context

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = Flask(__name__)
CORS(app)

# CONFIGURATION
def _find_image_folder():
    folder_name = "generated_step_images_20260824_0052"
    candidates = [
        os.path.join(project_root, folder_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name),
        os.path.abspath(f"./{folder_name}"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(project_root, folder_name))

GENERATED_IMAGE_FOLDER = _find_image_folder()
DEFAULT_MODE = os.getenv("LLM_MODE", "CLOUD").upper()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# --- HEALTH ENDPOINT ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "llm_provider": "groq",
        "llm_model": GROQ_MODEL,
        "retrieval": "qdrant"
    })


# --- MODELS ENDPOINT ---
@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns models actually available for grounded model-specific retrieval."""
    available_models = get_available_database_models()
    return jsonify({
        "status": "success",
        "models": available_models,
        "default_model": None
    })


# --- CHAT ENDPOINT ---
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_query = data.get("query", "").strip()
    model_param = data.get("model", "")
    if model_param is not None:
        model_param = str(model_param).strip()
    mode = data.get("mode", DEFAULT_MODE).upper()

    if not user_query:
        return jsonify({"status": "error", "message": "No query provided"}), 400

    try:
        result = generate_guide_from_rag(user_query, model=model_param, mode=mode)

        # Map status to HTTP codes
        status = result.get("status", "error")
        if status == "error":
            return jsonify(result), 500
        elif status in ["disambiguation_required", "model_conflict", "no_results", "success"]:
            return jsonify(result), 200

        return jsonify(result), 200
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"status": "error", "message": "An internal server error occurred."}), 500


# --- IMAGE SERVING WITH PATH TRAVERSAL PREVENTION ---
def _is_safe_path(base_dir: str, path: str) -> bool:
    """Ensure the requested path is strictly within the allowed base directory."""
    try:
        base_dir_resolved = os.path.realpath(base_dir)
        full_path = os.path.realpath(os.path.join(base_dir, path))
        return full_path.startswith(base_dir_resolved + os.sep) or full_path == base_dir_resolved
    except Exception:
        return False


@app.route("/generated_step_images_20260824_0052/<path:filename>")
def serve_generated_image(filename):
    """Serve generated step images securely with path traversal protection."""
    # Check for path traversal attempt
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    if not _is_safe_path(GENERATED_IMAGE_FOLDER, filename):
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    full_path = os.path.join(GENERATED_IMAGE_FOLDER, filename)
    if not os.path.isfile(full_path):
        return jsonify({"status": "error", "message": "Image not found"}), 404

    return send_from_directory(GENERATED_IMAGE_FOLDER, filename)


@app.route("/extracted_images/<path:filename>")
def serve_extracted_image(filename):
    """Fallback route to serve extracted images securely."""
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    extracted_folder = os.path.abspath(os.path.join(project_root, "extracted_images"))
    if not _is_safe_path(extracted_folder, filename):
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    full_path = os.path.join(extracted_folder, filename)
    if os.path.isfile(full_path):
        return send_from_directory(extracted_folder, filename)

    # Check generated images folder
    if _is_safe_path(GENERATED_IMAGE_FOLDER, filename) and os.path.isfile(os.path.join(GENERATED_IMAGE_FOLDER, filename)):
        return send_from_directory(GENERATED_IMAGE_FOLDER, filename)

    return jsonify({"status": "error", "message": "Image not found"}), 404


if __name__ == "__main__":
    print("==================================================")
    print("Samsung Prism Grounded RAG Server")
    print("==================================================")
    print(f"Qdrant retrieval: Stage 7A/7B/8")
    print(f"LLM provider: Groq")
    print(f"LLM model: {GROQ_MODEL}")
    print(f"Image source: Qdrant + {GENERATED_IMAGE_FOLDER}")
    print(f"Server: http://localhost:5000")
    print("==================================================")
    app.run(debug=False, port=5000)
