"""
Guide Weave — Stage 7A Precision Text Reranker
Isolated Jina Reranker v3.5 client with strict validation, retry backoff,
exact identity preservation, and fail-safe fallback to RRF ordering.
"""

import os
import sys
import time
import copy
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline_logger import pipeline_logger

logger = logging.getLogger("guide_weave.reranker")

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_RERANK_MODEL = "jina-reranker-v3.5"
DEFAULT_CANDIDATE_K = 30
DEFAULT_TOP_K = 8
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 2


def load_rerank_config() -> Dict[str, Any]:
    """Load configuration for reranker from environment files."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "backend" / ".env")
    load_dotenv()

    api_key = os.environ.get("JINA_API_KEY")
    enabled_str = os.environ.get("RERANK_ENABLED", "true").lower()
    enabled = enabled_str in ["true", "1", "yes"]

    model = os.environ.get("RERANK_MODEL", DEFAULT_RERANK_MODEL)

    try:
        candidate_k = int(os.environ.get("RERANK_CANDIDATE_K", str(DEFAULT_CANDIDATE_K)))
    except ValueError:
        candidate_k = DEFAULT_CANDIDATE_K

    try:
        top_k = int(os.environ.get("RERANK_TOP_K", str(DEFAULT_TOP_K)))
    except ValueError:
        top_k = DEFAULT_TOP_K

    try:
        timeout = int(os.environ.get("RERANK_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS

    try:
        max_retries = int(os.environ.get("RERANK_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    except ValueError:
        max_retries = DEFAULT_MAX_RETRIES

    return {
        "JINA_API_KEY": api_key,
        "RERANK_ENABLED": enabled,
        "RERANK_MODEL": model,
        "RERANK_CANDIDATE_K": candidate_k,
        "RERANK_TOP_K": top_k,
        "RERANK_TIMEOUT_SECONDS": timeout,
        "RERANK_MAX_RETRIES": max_retries
    }


def build_rerank_document(candidate: Dict[str, Any]) -> str:
    """
    Construct a structured contextual document representation from retrieved candidate.
    Uses the exact retrieved child chunk text without altering source payloads.
    """
    problem = candidate.get("problem_name") or ""
    steps = candidate.get("steps") or []
    step_nums = []
    if steps:
        step_nums = [str(s.get("step_number")) for s in steps if s.get("step_number") is not None]
    elif candidate.get("step_start") is not None:
        if candidate.get("step_end") is not None and candidate.get("step_start") != candidate.get("step_end"):
            step_nums = [f"{candidate['step_start']}-{candidate['step_end']}"]
        else:
            step_nums = [str(candidate["step_start"])]

    step_str = ", ".join(step_nums) if step_nums else "N/A"
    text = candidate.get("text") or ""

    parts = []
    if problem:
        parts.append(f"Problem:\n{problem}")
    if step_str != "N/A":
        parts.append(f"Step:\n{step_str}")
    if text:
        parts.append(f"Instruction:\n{text}")

    if not parts:
        return text or ""
    return "\n\n".join(parts)


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministically deduplicate candidates by chunk_id, preserving initial order."""
    seen_ids = set()
    deduped = []
    for c in candidates:
        chunk_id = c.get("chunk_id")
        if chunk_id is not None:
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
        deduped.append(c)
    return deduped


def _build_fallback_response(candidates: List[Dict[str, Any]], top_k: int,
                            reason: str, enabled: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a fallback response retaining original RRF ordering."""
    fallback_list = []
    for rank_idx, c in enumerate(candidates[:top_k], start=1):
        item = copy.deepcopy(c)
        item["rerank_score"] = None
        item["rerank_rank"] = None
        item["retrieval_rank"] = c.get("rank", rank_idx)
        item["rank"] = rank_idx
        fallback_list.append(item)

    metadata = {
        "enabled": enabled,
        "applied": False,
        "fallback": "rrf",
        "reason": reason,
        "candidate_count": len(candidates),
        "returned_count": len(fallback_list)
    }
    pipeline_logger.log_reranker_fallback(status="FALLBACK", fallback="RRF", reason=reason, retry_count=0)
    return fallback_list, metadata


def validate_rerank_response(data: Any, candidate_count: int) -> Tuple[bool, Optional[str]]:
    """
    Validate the structure, indices, and scores returned by the Jina API.
    """
    if not isinstance(data, dict):
        return False, "Response is not a valid JSON dictionary"

    results = data.get("results")
    if results is None or not isinstance(results, list):
        return False, "Response missing 'results' list"

    if len(results) == 0 and candidate_count > 0:
        return False, "Empty 'results' list in response for non-empty candidates"

    seen_indices = set()
    for item in results:
        if not isinstance(item, dict):
            return False, "Result item is not a dictionary"

        if "index" not in item:
            return False, "Result item missing 'index' field"

        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return False, f"Invalid index type: {type(index).__name__}"

        if index < 0 or index >= candidate_count:
            return False, f"Index {index} out of bounds for candidate count {candidate_count}"

        if index in seen_indices:
            return False, f"Duplicate index {index} returned by reranker"
        seen_indices.add(index)

        if "relevance_score" not in item:
            return False, "Result item missing 'relevance_score' field"

        rel_score = item.get("relevance_score")
        if not isinstance(rel_score, (int, float)) or isinstance(rel_score, bool):
            return False, f"Invalid relevance_score type: {type(rel_score).__name__}"

    return True, None


def rerank_documents(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    enabled: Optional[bool] = None,
    session: Optional[requests.Session] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rerank candidate documents using Jina Reranker v3.5 with strict validation and fail-safe fallback.

    Args:
        query: User search query
        candidates: List of candidate chunk dicts from Qdrant RRF fusion
        top_k: Number of final precision chunks to return
        model: Reranker model identifier (default from config/env)
        api_key: Jina API key (default from config/env)
        timeout: HTTP timeout in seconds
        max_retries: Maximum number of retries for transient errors
        enabled: Feature flag override

    Returns:
        (reranked_candidates, reranking_metadata)
    """
    config = load_rerank_config()

    is_enabled = config["RERANK_ENABLED"] if enabled is None else enabled
    final_model = model or config["RERANK_MODEL"]
    final_api_key = api_key or config["JINA_API_KEY"]
    final_top_k = top_k if top_k is not None else config["RERANK_TOP_K"]
    final_timeout = timeout if timeout is not None else config["RERANK_TIMEOUT_SECONDS"]
    final_max_retries = max_retries if max_retries is not None else config["RERANK_MAX_RETRIES"]

    # 1. Feature disabled
    if not is_enabled:
        fallback_list = []
        for rank_idx, c in enumerate(candidates[:final_top_k], start=1):
            item = copy.deepcopy(c)
            item["rerank_score"] = None
            item["rerank_rank"] = None
            item["retrieval_rank"] = c.get("rank", rank_idx)
            item["rank"] = rank_idx
            fallback_list.append(item)
        return fallback_list, {
            "enabled": False,
            "applied": False
        }

    # 2. Empty candidates
    if not candidates:
        return [], {
            "enabled": True,
            "applied": False,
            "reason": "empty_candidates",
            "candidate_count": 0,
            "returned_count": 0
        }

    # 3. Deduplicate candidates deterministically by chunk_id
    deduped = deduplicate_candidates(candidates)
    if not deduped:
        return [], {
            "enabled": True,
            "applied": False,
            "reason": "empty_candidates_after_dedup",
            "candidate_count": 0,
            "returned_count": 0
        }

    # 4. Single candidate optimization (no waste of API call)
    if len(deduped) == 1:
        single = copy.deepcopy(deduped[0])
        single["rerank_score"] = None
        single["rerank_rank"] = 1
        single["retrieval_rank"] = deduped[0].get("rank", 1)
        single["rank"] = 1
        return [single], {
            "enabled": True,
            "model": final_model,
            "candidate_count": 1,
            "returned_count": 1,
            "applied": False,
            "reason": "single_candidate"
        }

    # 5. Missing API key check
    if not final_api_key:
        logger.warning("Jina reranker enabled but JINA_API_KEY is missing. Falling back to RRF.")
        return _build_fallback_response(deduped, final_top_k, "missing_api_key", enabled=True)

    # 6. Prepare listwise request payload
    documents = [build_rerank_document(c) for c in deduped]
    actual_top_n = min(final_top_k, len(deduped))
    if actual_top_n <= 0:
        return [], {
            "enabled": True,
            "applied": False,
            "reason": "invalid_top_k",
            "candidate_count": len(deduped),
            "returned_count": 0
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {final_api_key}"
    }

    payload = {
        "model": final_model,
        "query": query,
        "documents": documents,
        "top_n": actual_top_n,
        "return_documents": False
    }

    pipeline_logger.log_reranker_start(
        enabled=True,
        model=final_model,
        endpoint=JINA_RERANK_URL,
        candidate_count=len(deduped),
        top_k=actual_top_n,
        timeout=final_timeout,
        max_retries=final_max_retries
    )

    http_session = session or requests.Session()
    start_time = time.perf_counter()

    # 7. Execute request with limited retry & exponential backoff
    last_error_reason = "unknown_error"
    response_data = None

    for attempt in range(final_max_retries + 1):
        try:
            resp = http_session.post(
                JINA_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=final_timeout
            )

            status = resp.status_code
            if status == 200:
                try:
                    response_data = resp.json()
                    break
                except Exception as json_err:
                    last_error_reason = f"malformed_json: {json_err}"
                    logger.warning(f"Jina reranker returned invalid JSON: {json_err}")
                    break

            elif status in [429, 500, 502, 503, 504]:
                last_error_reason = f"http_{status}"
                if attempt < final_max_retries:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(f"Jina reranker returned HTTP {status}. Retrying in {backoff}s (attempt {attempt+1}/{final_max_retries})...")
                    time.sleep(backoff)
                    continue
                else:
                    logger.warning(f"Jina reranker HTTP {status} exhausted {final_max_retries} retries. Falling back to RRF.")
                    break

            elif status in [400, 401, 403, 404]:
                last_error_reason = f"http_{status}_client_error"
                logger.warning(f"Jina reranker returned client error HTTP {status}. No retry. Falling back to RRF.")
                break

            else:
                last_error_reason = f"http_{status}"
                logger.warning(f"Jina reranker returned unexpected HTTP status {status}. Falling back to RRF.")
                break

        except requests.exceptions.Timeout:
            last_error_reason = "timeout"
            if attempt < final_max_retries:
                backoff = 2 ** (attempt + 1)
                logger.warning(f"Jina reranker timed out after {final_timeout}s. Retrying in {backoff}s (attempt {attempt+1}/{final_max_retries})...")
                time.sleep(backoff)
                continue
            else:
                logger.warning(f"Jina reranker timeout exhausted {final_max_retries} retries. Falling back to RRF.")
                break

        except requests.exceptions.RequestException as req_err:
            last_error_reason = f"connection_error: {type(req_err).__name__}"
            if attempt < final_max_retries:
                backoff = 2 ** (attempt + 1)
                logger.warning(f"Jina reranker connection error. Retrying in {backoff}s (attempt {attempt+1}/{final_max_retries})...")
                time.sleep(backoff)
                continue
            else:
                logger.warning(f"Jina reranker connection error exhausted {final_max_retries} retries. Falling back to RRF.")
                break

        except Exception as unexp_err:
            last_error_reason = f"unexpected_exception: {type(unexp_err).__name__}"
            logger.warning(f"Unexpected exception during Jina rerank call: {unexp_err}")
            break

    # 8. Check if response was successfully obtained
    if response_data is None:
        return _build_fallback_response(deduped, final_top_k, last_error_reason, enabled=True)

    # 9. Validate API response structure and indices
    is_valid, val_reason = validate_rerank_response(response_data, len(deduped))
    if not is_valid:
        logger.warning(f"Jina reranker response failed validation: {val_reason}. Falling back to RRF.")
        return _build_fallback_response(deduped, final_top_k, f"invalid_response: {val_reason}", enabled=True)

    # 10. Map validated indices back to EXACT original candidate objects
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    results_list = response_data.get("results", [])

    reranked_candidates = []
    for rank_idx, item in enumerate(results_list[:final_top_k], start=1):
        idx = item["index"]
        score = float(item["relevance_score"])

        orig = deduped[idx]
        candidate_copy = copy.deepcopy(orig)
        candidate_copy["rerank_score"] = score
        candidate_copy["rerank_rank"] = rank_idx
        candidate_copy["retrieval_rank"] = orig.get("rank", idx + 1)
        candidate_copy["rank"] = rank_idx
        # Preserve score attribute as rerank_score while keeping rrf_score intact
        candidate_copy["score"] = score

        reranked_candidates.append(candidate_copy)

    metadata = {
        "enabled": True,
        "model": final_model,
        "candidate_count": len(deduped),
        "returned_count": len(reranked_candidates),
        "applied": True,
        "latency_ms": round(latency_ms, 2)
    }

    pipeline_logger.log_reranker_output(
        model=final_model,
        candidate_count=len(deduped),
        returned_count=len(reranked_candidates),
        latency_ms=latency_ms,
        applied=True,
        results=reranked_candidates
    )
    pipeline_logger.log_reranker_score_comparison(surviving_candidates=reranked_candidates)

    return reranked_candidates, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Jina text reranker independently")
    parser.add_argument("--query", type=str, default="How do I clean the debris filter?")
    args = parser.parse_args()

    sample_candidates = [
        {"chunk_id": "c1", "problem_name": "Debris Filter Cleaning", "text": "Remove the debris filter cap and clean out lint.", "rank": 1, "rrf_score": 0.05},
        {"chunk_id": "c2", "problem_name": "Water Leakage", "text": "Check the water supply hoses for cracks or loose fittings.", "rank": 2, "rrf_score": 0.04},
        {"chunk_id": "c3", "problem_name": "Debris Filter Maintenance", "text": "Drain residual water before removing the filter.", "rank": 3, "rrf_score": 0.03}
    ]

    print("Query:", args.query)
    reranked, meta = rerank_documents(args.query, sample_candidates, top_k=2)
    print("Metadata:", meta)
    for r in reranked:
        print(f"Rank {r['rank']} (RRF rank {r['retrieval_rank']}) - Score: {r['rerank_score']} - Chunk: {r['chunk_id']}")
