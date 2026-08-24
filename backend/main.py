"""
Guide Weave — Main Orchestrator
Thin application orchestrator connecting three-state model resolution, retrieval (Stage 8),
and LLM generation with authoritative metadata injection.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment
load_dotenv()
load_dotenv("backend/.env")

# Add project root so scripts/ and backend/ can be imported
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.build_retrieval_context import build_retrieval_context
from backend.llm_generator import generate_grounded_guide
from backend.model_resolver import (
    resolve_model_context,
    get_available_database_models,
    extract_models_from_text,
    get_database_models
)

DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"


def extract_model(query: str, model_hint: Optional[str] = None) -> str:
    """Legacy helper for backward compatibility."""
    ctx = resolve_model_context(query, model_hint)
    return ctx.get("canonical_model") or "General"


def _image_path_to_url(file_path: str) -> str:
    """Convert a stored file_path to a browser-usable relative URL."""
    if not file_path:
        return ""
    # ./generated_step_images_20260824_0052/foo.png → /generated_step_images_20260824_0052/foo.png
    if file_path.startswith("./"):
        return "/" + file_path[2:]
    if file_path.startswith("/"):
        return file_path
    return "/" + file_path


def _attach_image_urls(response: Dict) -> Dict:
    """Add browser-usable 'url' field to every image in the response."""
    for step in response.get("steps", []):
        for img in step.get("images", []):
            fp = img.get("file_path", "")
            img["url"] = _image_path_to_url(fp)
    return response


def _strip_debug_fields(response: Dict) -> Dict:
    """Remove internal debug fields from production responses."""
    response.pop("_internal_error", None)
    response.pop("_validation_errors", None)
    return response


def generate_guide_from_rag(query: str, model: Optional[str] = None,
                            mode: str = "CLOUD") -> Dict:
    """
    Main orchestration function implementing explicit three-state model handling.

    Flow:
        1. Validate query
        2. Resolve model context (Three states: State A, State B, State C, or Conflict)
        3. For State A (No model): Return disambiguation response immediately
        4. For Model Conflict: Return conflict response immediately
        5. For State B / State C: Invoke Stage 8 retrieval context
        6. Handle zero/insufficient evidence cleanly for known vs generic mode
        7. Generate grounded guide with Groq
        8. Authoritatively inject warning, model_known, guidance_scope
        9. Attach authoritative image URLs
        10. Return final JSON response
    """
    # 1. Validate query
    if not query or not query.strip():
        return {"status": "error", "message": "No query provided."}

    query = query.strip()

    # 2. Resolve model context
    model_ctx = resolve_model_context(query, model_hint=model)
    resolution_status = model_ctx.get("status")

    # Structured logging
    print("[MODEL]")
    print(f"  requested={model_ctx.get('requested_model')}")
    print(f"  canonical={model_ctx.get('canonical_model')}")
    print(f"  database_model={model_ctx.get('database_model')}")
    print(f"  known={str(model_ctx.get('model_known')).lower()}")
    print(f"  retrieval_mode={model_ctx.get('retrieval_mode')}")
    print(f"  resolution_status={resolution_status}")

    # 3. STATE A: No model provided
    if resolution_status == "disambiguation_required":
        return {
            "status": "disambiguation_required",
            "message": model_ctx.get("message", "Please enter your Samsung washing machine model number so I can provide accurate troubleshooting guidance."),
            "model": None,
            "model_known": False,
            "available_models": model_ctx.get("available_models", get_available_database_models())
        }

    # 4. EDGE CASE: Model conflict
    if resolution_status == "model_conflict":
        return {
            "status": "model_conflict",
            "message": model_ctx.get("message", "Two different washing machine models were provided. Please specify only one model."),
            "models_detected": model_ctx.get("models_detected", []),
            "model": None,
            "model_known": False
        }

    canonical_model = model_ctx.get("canonical_model")
    model_known = model_ctx.get("model_known", False)
    retrieval_mode = model_ctx.get("retrieval_mode", "generic")
    warning = model_ctx.get("warning")

    # 5. Call Stage 8 Retrieval
    try:
        retrieval_context = build_retrieval_context(
            query=query,
            appliance_type="washing_machine",
            brand="Samsung",
            model=model_ctx.get("database_model"),
            model_context=model_ctx,
            retrieval_mode=retrieval_mode,
            text_top_k=8,
            image_top_k=3
        )
    except SystemExit:
        return {
            "status": "error",
            "message": "Retrieval infrastructure error. Please check Qdrant and Jina API availability."
        }
    except Exception as e:
        print(f"    Retrieval error: {e}")
        return {
            "status": "error",
            "message": "Retrieval service is temporarily unavailable."
        }

    # 6. Handle retrieval status (zero/insufficient evidence)
    if retrieval_context.get("status") == "no_text_evidence":
        if model_known:
            return {
                "status": "no_results",
                "model": canonical_model,
                "model_known": True,
                "guidance_scope": "model_specific",
                "warning": None,
                "message": "No relevant model-specific technical documentation was found for this request.",
                "query": query
            }
        else:
            return {
                "status": "no_results",
                "model": canonical_model,
                "model_known": False,
                "guidance_scope": "generic",
                "warning": warning,
                "message": "No generic troubleshooting evidence was found for this request.",
                "query": query
            }

    # 7. Call LLM
    try:
        llm_result = generate_grounded_guide(
            query=query,
            retrieval_context=retrieval_context,
            model=canonical_model,
            mode=mode
        )
    except Exception as e:
        print(f"    LLM error: {e}")
        return {
            "status": "error",
            "message": "Guide generation service is temporarily unavailable."
        }

    # 8. Check LLM result
    if llm_result.get("status") == "error":
        return llm_result

    # 9. Authoritative metadata injection (Application layer is authoritative)
    llm_result["model"] = canonical_model
    llm_result["model_known"] = model_known
    llm_result["guidance_scope"] = "model_specific" if model_known else "generic"
    llm_result["warning"] = warning
    llm_result["query"] = query
    llm_result["status"] = "success"

    # 10. Attach image URLs (server-side, not LLM-generated)
    llm_result = _attach_image_urls(llm_result)

    # 11. Strip debug fields in production
    if not DEBUG_RETRIEVAL:
        llm_result = _strip_debug_fields(llm_result)

    return llm_result


if __name__ == "__main__":
    test_query = "How do I clean the debris filter on Samsung WA5471ABP?"
    result = generate_guide_from_rag(test_query, model="WA5471ABP", mode="CLOUD")
    print(json.dumps(result, indent=2))
