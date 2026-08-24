"""
Guide Weave — Authoritative Model Resolution Module
Handles Samsung appliance model discovery, normalization, conflict detection,
and three-state model resolution (State A: No Model, State B: Known Model, State C: Unknown Model).
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv

# Base configuration
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Canonical normalization map: User alias -> (Canonical Display Name, Qdrant DB Model)
# Default known models in database
DEFAULT_DATABASE_MODELS = {"WA5471ABP/XAA"}

MODEL_NORMALIZATION = {
    "WA5471ABP": ("WA5471ABP", "WA5471ABP/XAA"),
    "WA5471ABP/XAA": ("WA5471ABP", "WA5471ABP/XAA"),
    "WA5471": ("WA5471ABP", "WA5471ABP/XAA"),
}

# In-memory cache for discovered database models
_CACHED_DATABASE_MODELS: Optional[Set[str]] = None


def get_database_models(force_refresh: bool = False) -> Set[str]:
    """
    Discover models actually indexed in Qdrant washing_machines collection.
    Cached after first call unless force_refresh is True.
    """
    global _CACHED_DATABASE_MODELS
    if _CACHED_DATABASE_MODELS is not None and not force_refresh:
        return _CACHED_DATABASE_MODELS

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    discovered_models: Set[str] = set()

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, timeout=3.0)
        if client.collection_exists("washing_machines"):
            # Scroll points to discover unique model payloads (up to 250 points)
            points, _ = client.scroll(
                collection_name="washing_machines",
                limit=250,
                with_payload=["model"],
                with_vectors=False
            )
            for pt in points:
                m = pt.payload.get("model") if pt.payload else None
                if m and isinstance(m, str) and m.strip():
                    discovered_models.add(m.strip())
    except Exception as e:
        # Fallback to default if Qdrant is unreachable or during unit tests
        pass

    if not discovered_models:
        discovered_models = set(DEFAULT_DATABASE_MODELS)

    _CACHED_DATABASE_MODELS = discovered_models
    return _CACHED_DATABASE_MODELS


def extract_models_from_text(text: str) -> List[str]:
    """
    Extract potential Samsung appliance model numbers from query text.
    Handles standard patterns (e.g., WA5471ABP, WF5M5100AW, WW90T504DAN, etc.)
    and case-insensitive normalization.
    """
    if not text or not text.strip():
        return []

    # Exclude common appliance words that might match regex patterns
    stopwords = {"SAMSUNG", "WASHER", "MACHINE", "DRYER", "STEP", "CODE", "ERROR", "FILTER", "PUMP", "HOSE", "DOOR"}

    found_models: List[str] = []
    seen: Set[str] = set()

    # 1. Check known aliases first
    for alias in MODEL_NORMALIZATION:
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            canonical, _ = MODEL_NORMALIZATION[alias]
            if canonical not in seen:
                seen.add(canonical)
                found_models.append(canonical)

    # 2. General Samsung model regex: 2-3 letters prefix + digits + optional alphanumeric/slashes
    # e.g., WA5471ABP, WA5471ABP/XAA, WF5M5100AW, WF350ANR, DC68-00000A, WW90T504DAN, WD80T654DBX, ABC123
    regex_pattern = r"\b(?:WA|WW|WD|WF|DC|DV|WT)[A-Za-z0-9/\-]+\b"
    matches = re.findall(regex_pattern, text, re.IGNORECASE)

    for m in matches:
        m_clean = m.strip().upper()
        # Remove trailing punctuation or slashes
        m_clean = m_clean.rstrip("./,-")
        if m_clean in stopwords or len(m_clean) < 4:
            continue
        # Check if already covered by canonical alias
        canonical_form = MODEL_NORMALIZATION.get(m_clean, (m_clean, None))[0]
        if canonical_form not in seen:
            seen.add(canonical_form)
            found_models.append(m_clean)

    return found_models


def normalize_model_identifier(model_str: str) -> Tuple[str, Optional[str], bool]:
    """
    Normalize a model string to (canonical_display, database_model, is_known).
    """
    if not model_str or not model_str.strip():
        return ("", None, False)

    clean_str = model_str.strip()
    upper_str = clean_str.upper()

    db_models = get_database_models()

    # Check normalization dictionary
    if upper_str in MODEL_NORMALIZATION:
        canonical, db_form = MODEL_NORMALIZATION[upper_str]
        is_known = db_form in db_models
        return (canonical, db_form if is_known else None, is_known)

    # Check direct database match
    if clean_str in db_models or upper_str in db_models:
        return (upper_str, upper_str, True)

    # Unknown model
    return (upper_str, None, False)


def resolve_model_context(query: str, model_hint: Optional[str] = None) -> Dict:
    """
    Authoritative Three-State Model Resolver.

    States:
        - STATE A (No Model): Neither hint nor query contains a model -> disambiguation_required.
        - STATE B (Known Model): Model exists in indexed database -> model_specific.
        - STATE C (Unknown Model): Model provided but not in database -> generic with warning.
        - CONFLICT: Conflicting models provided in hint vs query -> model_conflict.

    Returns:
        {
            "status": "resolved" | "disambiguation_required" | "model_conflict",
            "requested_model": Optional[str],
            "canonical_model": Optional[str],
            "database_model": Optional[str],
            "model_known": bool,
            "retrieval_mode": "model_specific" | "generic" | None,
            "warning": Optional[Dict],
            "message": Optional[str],
            "models_detected": Optional[List[str]]
        }
    """
    clean_query = query.strip() if query else ""
    clean_hint = model_hint.strip() if model_hint and isinstance(model_hint, str) else ""

    # Filter out placeholder hints like "General", "Auto-Detect", "None", ""
    if clean_hint.lower() in ["general", "auto-detect", "none", "null", "all", ""]:
        clean_hint = ""

    # Extract models from query text
    query_models = extract_models_from_text(clean_query)

    # Check for hint model
    hint_model = clean_hint if clean_hint else None

    # Edge Case: Model Conflict
    if hint_model and query_models:
        # Check if hint and query model resolve to different models
        hint_canonical, _, _ = normalize_model_identifier(hint_model)
        query_canonical, _, _ = normalize_model_identifier(query_models[0])
        if hint_canonical != query_canonical:
            return {
                "status": "model_conflict",
                "message": "Two different washing machine models were provided. Please specify only one model.",
                "models_detected": [hint_model, query_models[0]],
                "requested_model": None,
                "canonical_model": None,
                "database_model": None,
                "model_known": False,
                "retrieval_mode": None,
                "warning": None
            }

    # Determine candidate model
    target_raw_model = hint_model or (query_models[0] if query_models else None)

    # If still no model, check if hint_model is syntactically a model even if regex missed it (e.g. ABC123)
    if not target_raw_model and clean_hint:
        target_raw_model = clean_hint

    # STATE A: No model provided
    if not target_raw_model:
        return {
            "status": "disambiguation_required",
            "message": "Please enter your Samsung washing machine model number so I can provide accurate troubleshooting guidance.",
            "requested_model": None,
            "canonical_model": None,
            "database_model": None,
            "model_known": False,
            "retrieval_mode": None,
            "warning": None,
            "available_models": get_available_database_models()
        }

    # Normalize candidate model
    canonical, db_form, is_known = normalize_model_identifier(target_raw_model)

    # STATE B: Known Model
    if is_known:
        return {
            "status": "resolved",
            "requested_model": target_raw_model,
            "canonical_model": canonical,
            "database_model": db_form,
            "model_known": True,
            "retrieval_mode": "model_specific",
            "warning": None
        }

    # STATE C: Unknown Model (Generic Mode)
    warning_msg = (
        f"Model {canonical} is not available in our database. The following troubleshooting guidance "
        f"is generic for Samsung washing machines and may not exactly match your model."
    )
    return {
        "status": "resolved",
        "requested_model": target_raw_model,
        "canonical_model": canonical,
        "database_model": None,
        "model_known": False,
        "retrieval_mode": "generic",
        "warning": {
            "type": "unknown_model",
            "message": warning_msg
        }
    }


def get_available_database_models() -> List[Dict[str, str]]:
    """Returns list of models available in the database for the /api/models endpoint."""
    db_models = get_database_models()
    models_list = []
    seen = set()

    for db_m in db_models:
        # Find display name from normalization map or strip suffix
        display = db_m.split("/")[0]
        if display not in seen:
            seen.add(display)
            models_list.append({
                "display_name": display,
                "canonical_model": db_m
            })

    if not models_list:
        models_list.append({
            "display_name": "WA5471ABP",
            "canonical_model": "WA5471ABP/XAA"
        })

    return models_list
