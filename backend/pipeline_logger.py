"""
Guide Weave — Centralized Backend Pipeline Logger
Provides comprehensive, thread-safe, fail-safe observability across the entire RAG pipeline.

Responsibilities:
    - Structured query lifecycle logging to logs/guide_weave.log
    - Context-propagated QUERY_ID and THREAD_ID per request
    - Stage-by-stage recording (Model Resolution, Dense, Sparse, RRF, Reranker, Stage 8, Images, Grounding, LLM, Final Response)
    - Monotonic latency tracking
    - Sensitive secret redaction (API keys, Authorization headers, Bearer tokens)
    - Configurable text preview truncation & Full Trace mode
    - Rotating file handling (10MB, 5 backups, UTF-8)
    - Fail-safe isolation: logging failures NEVER raise exceptions to caller
"""

import os
import sys
import time
import json
import uuid
import re
import hashlib
import logging
import threading
import contextvars
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- SENSITIVE DATA REDACTION PATTERNS ---
SENSITIVE_PATTERNS = [
    (re.compile(r'Bearer\s+[A-Za-z0-9_\-\.]{8,}', re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r'Authorization[\'"]?\s*:\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), "'Authorization': '[REDACTED]'"),
    (re.compile(r'jina_[a-zA-Z0-9_\-]{16,}', re.IGNORECASE), "jina_[REDACTED]"),
    (re.compile(r'gsk_[a-zA-Z0-9_\-]{16,}', re.IGNORECASE), "gsk_[REDACTED]"),
    (re.compile(r'api_key[\'"]?\s*[:=]\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), "api_key='[REDACTED]'"),
    (re.compile(r'apikey[\'"]?\s*[:=]\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), "apikey='[REDACTED]'"),
]


def redact_sensitive(text: str) -> str:
    """Sanitize strings against API keys, authorization tokens, and sensitive headers."""
    if not isinstance(text, str):
        return text
    result = text
    # Also redact actual env keys if present
    for env_var in ["JINA_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "QDRANT_API_KEY"]:
        val = os.environ.get(env_var)
        if val and len(val) > 4:
            result = result.replace(val, f"[{env_var}_REDACTED]")
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class QueryContext:
    """Request-scoped state container for pipeline execution."""
    def __init__(self, query_id: str, query: str = "", model: Optional[str] = None,
                 mode: str = "CLOUD", http_method: str = "POST",
                 endpoint: str = "/api/chat", client_ip: str = "127.0.0.1"):
        self.query_id = query_id
        self.query = query
        self.model = model
        self.mode = mode
        self.http_method = http_method
        self.endpoint = endpoint
        self.client_ip = client_ip
        self.thread_id = threading.get_ident()
        
        # Timing
        self.start_perf = time.perf_counter()
        now = datetime.now(timezone(timedelta(hours=5, minutes=30)))  # IST default
        self.start_time_iso = now.isoformat()
        
        # Stage Latency Breakdown (ms)
        self.model_resolution_ms: Optional[float] = None
        self.qdrant_latency_ms: Optional[float] = None
        self.dense_retrieval_ms: Optional[float] = None
        self.sparse_retrieval_ms: Optional[float] = None
        self.rrf_ms: Optional[float] = None
        self.rerank_latency_ms: Optional[float] = None
        self.stage8_ms: Optional[float] = None
        self.image_latency_ms: Optional[float] = None
        self.llm_latency_ms: Optional[float] = None
        self.total_latency_ms: Optional[float] = None
        
        # Stage Execution Tracking
        self.retrieval_called = False
        self.reranker_called = False
        self.image_called = False
        self.llm_called = False
        self.status = "in_progress"


# Context variable for thread/async safety
_CURRENT_QUERY_CTX: contextvars.ContextVar[Optional[QueryContext]] = contextvars.ContextVar(
    "current_query_ctx", default=None
)


class PipelineLogger:
    """Centralized logger managing guide_weave.log formatting and rotation."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(PipelineLogger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._init_logger()

    def _init_logger(self):
        """Configure logging file handler and rotation."""
        log_file_rel = os.environ.get("PIPELINE_LOG_FILE", "logs/guide_weave.log")
        log_file_path = PROJECT_ROOT / log_file_rel
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        log_level_str = os.environ.get("PIPELINE_LOG_LEVEL", "DEBUG").upper()
        log_level = getattr(logging, log_level_str, logging.DEBUG)
        
        self.logger = logging.getLogger("guide_weave.pipeline")
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Remove existing handlers to avoid duplicates on reload
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            
        # 10 MB per file, 5 backup files, UTF-8 encoding
        handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.log_file_path = str(log_file_path)

    @property
    def is_full_trace(self) -> bool:
        val = os.environ.get("PIPELINE_LOG_FULL_TRACE", "false").lower()
        return val in ("true", "1", "yes")

    @property
    def text_preview_limit(self) -> int:
        try:
            return int(os.environ.get("PIPELINE_LOG_TEXT_PREVIEW", "500"))
        except ValueError:
            return 500

    def _format_preview(self, text: Optional[str]) -> Tuple[str, bool]:
        """Truncate text for preview unless full trace is active."""
        if not text:
            return "", False
        if self.is_full_trace:
            return redact_sensitive(text), False
        limit = self.text_preview_limit
        if len(text) <= limit:
            return redact_sensitive(text), False
        truncated = text[:limit] + " ... [TRUNCATED]"
        return redact_sensitive(truncated), True

    def get_context(self) -> Optional[QueryContext]:
        """Retrieve current request context."""
        return _CURRENT_QUERY_CTX.get()

    def create_query_context(self, query: str = "", model: Optional[str] = None,
                             mode: str = "CLOUD", query_id: Optional[str] = None,
                             http_method: str = "POST", endpoint: str = "/api/chat",
                             client_ip: str = "127.0.0.1") -> QueryContext:
        """Create and bind a new query context."""
        if not query_id:
            query_id = str(uuid.uuid4())
        ctx = QueryContext(
            query_id=query_id,
            query=query,
            model=model,
            mode=mode,
            http_method=http_method,
            endpoint=endpoint,
            client_ip=client_ip
        )
        _CURRENT_QUERY_CTX.set(ctx)
        return ctx

    def clear_context(self):
        """Clear the current query context."""
        _CURRENT_QUERY_CTX.set(None)

    def _write_log_entry(self, level: str, stage: str, lines: List[str], ctx: Optional[QueryContext] = None):
        """Thread-safe, sanitized entry writer."""
        try:
            if ctx is None:
                ctx = self.get_context()
            
            q_id = ctx.query_id if ctx else "NO_QUERY_CTX"
            t_id = ctx.thread_id if ctx else threading.get_ident()
            timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            header = f"[{timestamp}] [{level.upper()}] [QUERY_ID={q_id}] [THREAD_ID={t_id}] [STAGE={stage.upper()}]"
            
            entry = [header]
            for line in lines:
                entry.append(redact_sensitive(str(line)))
            
            msg = "\n".join(entry) + "\n"
            self.logger.info(msg)
        except Exception:
            # Observability must NEVER crash the application
            pass

    # =========================================================================
    # 1. QUERY LIFECYCLE
    # =========================================================================

    def log_query_start(self, query: str, model: Optional[str] = None,
                        mode: str = "CLOUD", http_method: str = "POST",
                        endpoint: str = "/api/chat", client_ip: str = "127.0.0.1",
                        query_id: Optional[str] = None) -> QueryContext:
        """Log the visual delimiter and query start block."""
        try:
            ctx = self.create_query_context(
                query=query, model=model, mode=mode,
                query_id=query_id, http_method=http_method,
                endpoint=endpoint, client_ip=client_ip
            )
            
            lines = [
                "====================================================================",
                "====================== GUIDE WEAVE QUERY START =====================",
                "====================================================================",
                f"QUERY_ID: {ctx.query_id}",
                f"START_TIME: {ctx.start_time_iso}",
                f"HTTP_METHOD: {ctx.http_method}",
                f"ENDPOINT: {ctx.endpoint}",
                f"QUERY: {ctx.query}",
                f"REQUEST_MODEL: {ctx.model if ctx.model is not None else 'None'}",
                f"MODE: {ctx.mode}",
                f"CLIENT_IP: {ctx.client_ip}",
                f"THREAD_ID: {ctx.thread_id}",
                "===================================================================="
            ]
            self._write_log_entry("INFO", "START", lines, ctx)
            return ctx
        except Exception:
            return self.get_context() or QueryContext(query_id=str(uuid.uuid4()))

    def log_query_end(self, status: str = "success", ctx: Optional[QueryContext] = None):
        """Log the visual delimiter and query end block with latencies."""
        try:
            if ctx is None:
                ctx = self.get_context()
            if not ctx:
                return

            total_ms = (time.perf_counter() - ctx.start_perf) * 1000.0
            ctx.total_latency_ms = round(total_ms, 2)
            ctx.status = status
            
            now = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
            
            def _fmt_lat(val: Optional[float], called: bool = True) -> str:
                if not called or val is None:
                    return "NOT_CALLED"
                return f"{round(val, 2)} ms"

            lines = [
                "====================================================================",
                "======================= GUIDE WEAVE QUERY END ======================",
                "====================================================================",
                f"QUERY_ID: {ctx.query_id}",
                f"END_TIME: {now}",
                f"STATUS: {ctx.status}",
                f"TOTAL_LATENCY_MS: {round(total_ms, 2)}",
                f"MODEL_RESOLUTION_LATENCY: {_fmt_lat(ctx.model_resolution_ms, True)}",
                f"QDRANT_LATENCY_MS: {_fmt_lat(ctx.qdrant_latency_ms, ctx.retrieval_called)}",
                f"RERANK_LATENCY_MS: {_fmt_lat(ctx.rerank_latency_ms, ctx.reranker_called)}",
                f"IMAGE_LATENCY_MS: {_fmt_lat(ctx.image_latency_ms, ctx.image_called)}",
                f"LLM_LATENCY_MS: {_fmt_lat(ctx.llm_latency_ms, ctx.llm_called)}",
                "===================================================================="
            ]
            self._write_log_entry("INFO", "END", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 2. MODEL RESOLUTION
    # =========================================================================

    def log_model_resolution(self, model_ctx: Dict[str, Any], latency_ms: Optional[float] = None):
        """Log the complete model resolution state."""
        try:
            ctx = self.get_context()
            if ctx and latency_ms is not None:
                ctx.model_resolution_ms = latency_ms
                
            status = model_ctx.get("status")
            lines = [
                "-------------------- MODEL RESOLUTION ------------------------------",
                f"requested_model = {model_ctx.get('requested_model')}",
                f"canonical_model = {model_ctx.get('canonical_model')}",
                f"database_model = {model_ctx.get('database_model')}",
                f"model_known = {str(model_ctx.get('model_known')).lower()}",
                f"retrieval_mode = {model_ctx.get('retrieval_mode')}",
                f"guidance_scope = {'model_specific' if model_ctx.get('model_known') else 'generic'}",
                f"warning = {model_ctx.get('warning')}",
                f"resolution_status = {status}"
            ]
            
            if status == "disambiguation_required":
                lines.append("ACTION: Disambiguation required. Early return before retrieval.")
            elif status == "model_conflict":
                lines.append("ACTION: Model conflict detected. Early return before retrieval.")
                lines.append(f"models_detected = {model_ctx.get('models_detected', [])}")
                
            self._write_log_entry("INFO", "MODEL_RESOLUTION", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 3. DENSE RETRIEVAL
    # =========================================================================

    def log_dense_retrieval(self, collection: str, query: str, retrieval_mode: str,
                            q_filter: Any, requested_limit: int, points: List[Any],
                            latency_ms: Optional[float] = None):
        """Log Qdrant dense vector search results."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.retrieval_called = True
                ctx.dense_retrieval_ms = latency_ms
                
            lines = [
                "-------------------- DENSE RETRIEVAL -------------------------------",
                f"collection = {collection}",
                f"query = {query}",
                f"retrieval_mode = {retrieval_mode}",
                f"filter = {str(q_filter)}",
                f"requested_limit = {requested_limit}",
                f"actual_results = {len(points)}"
            ]
            
            for idx, pt in enumerate(points, start=1):
                p = pt.payload or {}
                text = p.get("text", "")
                preview, truncated = self._format_preview(text)
                
                pg_s = p.get("page_start")
                pg_e = p.get("page_end")
                page_str = f"{pg_s}-{pg_e}" if pg_s and pg_e and pg_s != pg_e else str(pg_s or "N/A")
                
                lines.append(f"\nDENSE RESULT #{idx:02d}")
                lines.append("----------------")
                lines.append(f"chunk_id = {p.get('chunk_id')}")
                lines.append(f"document_id = {p.get('document_id')}")
                lines.append(f"problem_id = {p.get('problem_id')}")
                lines.append(f"step_id = {p.get('step_id')}")
                lines.append(f"problem_name = {p.get('problem_name')}")
                lines.append(f"step_number = {p.get('step_start') or p.get('step_number')}")
                lines.append(f"model = {p.get('model')}")
                lines.append(f"page_numbers = {page_str}")
                lines.append(f"dense_score = {pt.score:.6f}")
                lines.append(f"text_truncated = {str(truncated).lower()}")
                lines.append(f"text_preview = {preview}")

            self._write_log_entry("DEBUG", "DENSE_RETRIEVAL", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 4. SPARSE RETRIEVAL
    # =========================================================================

    def log_sparse_retrieval(self, collection: str, query: str, retrieval_mode: str,
                             q_filter: Any, requested_limit: int, points: List[Any],
                             matching_tokens_count: int, reason_if_zero: Optional[str] = None,
                             latency_ms: Optional[float] = None):
        """Log Qdrant sparse lexical search results."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.sparse_retrieval_ms = latency_ms
                
            lines = [
                "-------------------- SPARSE RETRIEVAL ------------------------------",
                f"collection = {collection}",
                f"query = {query}",
                f"retrieval_mode = {retrieval_mode}",
                f"filter = {str(q_filter)}",
                f"requested_limit = {requested_limit}",
                f"actual_results = {len(points)}",
                f"matching_tokens_count = {matching_tokens_count}"
            ]
            
            if not points:
                lines.append("SPARSE RESULT COUNT: 0")
                lines.append(f"REASON: {reason_if_zero or 'No matching tokens in sparse vocabulary'}")
            else:
                for idx, pt in enumerate(points, start=1):
                    p = pt.payload or {}
                    text = p.get("text", "")
                    preview, truncated = self._format_preview(text)
                    
                    lines.append(f"\nSPARSE RESULT #{idx:02d}")
                    lines.append("-----------------")
                    lines.append(f"chunk_id = {p.get('chunk_id')}")
                    lines.append(f"document_id = {p.get('document_id')}")
                    lines.append(f"problem_id = {p.get('problem_id')}")
                    lines.append(f"step_id = {p.get('step_id')}")
                    lines.append(f"sparse_score = {pt.score:.6f}")
                    lines.append(f"sparse_rank = {idx}")
                    lines.append(f"text_truncated = {str(truncated).lower()}")
                    lines.append(f"text_preview = {preview}")

            self._write_log_entry("DEBUG", "SPARSE_RETRIEVAL", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 5. RRF FUSION
    # =========================================================================

    def log_rrf_fusion(self, rrf_k: int, dense_count: int, sparse_count: int,
                       fused_results: List[Dict[str, Any]], candidate_pool_limit: int,
                       latency_ms: Optional[float] = None):
        """Log reciprocal rank fusion rankings and scores."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.rrf_ms = latency_ms
                
            lines = [
                "-------------------- RRF FUSION ------------------------------------",
                f"rrf_k = {rrf_k}",
                f"dense_count = {dense_count}",
                f"sparse_count = {sparse_count}",
                f"fused_count = {len(fused_results)}",
                f"candidate_pool_limit = {candidate_pool_limit}"
            ]
            
            for item in fused_results:
                rank = item.get("rank")
                d_score = item.get("dense_score")
                s_score = item.get("sparse_score")
                d_score_str = f"{d_score:.6f}" if d_score is not None else "NOT_AVAILABLE"
                s_score_str = f"{s_score:.6f}" if s_score is not None else "NOT_AVAILABLE"
                
                lines.append(f"\nRRF RESULT #{rank:02d}")
                lines.append("--------------")
                lines.append(f"chunk_id = {item.get('chunk_id')}")
                lines.append(f"dense_rank = {item.get('dense_rank')}")
                lines.append(f"sparse_rank = {item.get('sparse_rank')}")
                lines.append(f"dense_score = {d_score_str}")
                lines.append(f"sparse_score = {s_score_str}")
                lines.append(f"rrf_score = {item.get('rrf_score', 0.0):.6f}")
                lines.append(f"original_rank = {rank}")

            self._write_log_entry("DEBUG", "RRF_FUSION", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 6. RERANKER LOGGING
    # =========================================================================

    def log_reranker_start(self, enabled: bool, model: str, endpoint: str,
                           candidate_count: int, top_k: int, timeout: int, max_retries: int):
        """Log Jina reranker initialization without API keys."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.reranker_called = True
                
            lines = [
                "-------------------- RERANKER START --------------------------------",
                f"enabled = {str(enabled).lower()}",
                f"model = {model}",
                f"endpoint = {endpoint}",
                f"candidate_count = {candidate_count}",
                f"top_k = {top_k}",
                f"timeout = {timeout}",
                f"max_retries = {max_retries}"
            ]
            self._write_log_entry("INFO", "RERANKER_START", lines, ctx)
        except Exception:
            pass

    def log_reranker_input_pool(self, candidates: List[Dict[str, Any]], top_k: int):
        """Log exact candidate pool prepared for Jina reranker."""
        try:
            ctx = self.get_context()
            lines = [
                "====================================================================",
                "RERANKER INPUT CANDIDATE POOL",
                "====================================================================",
                f"Candidate pool size: {len(candidates)}",
                f"Final requested size: {top_k}"
            ]
            
            for idx, c in enumerate(candidates, start=1):
                d_score = c.get("dense_score")
                s_score = c.get("sparse_score")
                d_score_str = f"{d_score:.6f}" if d_score is not None else "NOT_AVAILABLE"
                s_score_str = f"{s_score:.6f}" if s_score is not None else "NOT_AVAILABLE"
                
                lines.append(f"\n#{idx:02d}")
                lines.append(f"chunk_id: {c.get('chunk_id')}")
                lines.append(f"problem: {c.get('problem_name')}")
                lines.append(f"step: {c.get('step_start') or 'N/A'}")
                lines.append(f"rrf_score: {c.get('rrf_score', 0.0):.6f}")
                lines.append(f"dense_score: {d_score_str}")
                lines.append(f"sparse_score: {s_score_str}")
                
                if self.is_full_trace:
                    lines.append(f"instruction_full:\n{c.get('text', '')}")
                else:
                    lines.append(f"text_length: {len(c.get('text', ''))}")

            self._write_log_entry("DEBUG", "RERANKER_INPUT", lines, ctx)
        except Exception:
            pass

    def log_reranker_output(self, model: str, candidate_count: int,
                            returned_count: int, latency_ms: float, applied: bool,
                            results: List[Dict[str, Any]]):
        """Log Jina reranking results and ranking comparisons."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.rerank_latency_ms = latency_ms
                
            lines = [
                "-------------------- RERANKER OUTPUT -------------------------------",
                f"model: {model}",
                f"candidate_count: {candidate_count}",
                f"returned_count: {returned_count}",
                f"latency_ms: {round(latency_ms, 2)}",
                f"applied: {str(applied).lower()}"
            ]
            
            for idx, r in enumerate(results, start=1):
                pg_s = r.get("page_start")
                pg_e = r.get("page_end")
                page_str = f"{pg_s}-{pg_e}" if pg_s and pg_e and pg_s != pg_e else str(pg_s or "N/A")
                
                d_score = r.get("dense_score")
                s_score = r.get("sparse_score")
                r_score = r.get("rerank_score")
                d_score_str = f"{d_score:.6f}" if d_score is not None else "NOT_AVAILABLE"
                s_score_str = f"{s_score:.6f}" if s_score is not None else "NOT_AVAILABLE"
                r_score_str = f"{r_score:.6f}" if r_score is not None else "NOT_AVAILABLE"
                
                lines.append(f"\nRERANK RESULT #{idx:02d}")
                lines.append("-----------------")
                lines.append(f"chunk_id: {r.get('chunk_id')}")
                lines.append(f"original_rrf_rank: {r.get('retrieval_rank')}")
                lines.append(f"rerank_rank: {r.get('rerank_rank') or idx}")
                lines.append(f"rrf_score: {r.get('rrf_score', 0.0):.6f}")
                lines.append(f"dense_score: {d_score_str}")
                lines.append(f"sparse_score: {s_score_str}")
                lines.append(f"rerank_score: {r_score_str}")
                lines.append(f"problem: {r.get('problem_name')}")
                lines.append(f"step: {r.get('step_start') or 'N/A'}")
                lines.append(f"document_id: {r.get('document_id')}")
                lines.append(f"page_numbers: {page_str}")

            self._write_log_entry("DEBUG", "RERANKER_OUTPUT", lines, ctx)
        except Exception:
            pass

    def log_reranker_score_comparison(self, surviving_candidates: List[Dict[str, Any]]):
        """Log before vs after comparison for precision evaluation."""
        try:
            ctx = self.get_context()
            lines = [
                "-------------------- RERANK SCORE COMPARISON -----------------------"
            ]
            for c in surviving_candidates:
                rrf_s = c.get("rrf_score")
                rerank_s = c.get("rerank_score")
                lines.append(f"chunk_id: {c.get('chunk_id')}")
                lines.append(f"  original_rrf_rank: {c.get('retrieval_rank')}")
                lines.append(f"  rerank_rank: {c.get('rerank_rank') or c.get('rank')}")
                lines.append(f"  rrf_score: {f'{rrf_s:.6f}' if rrf_s is not None else 'NOT_AVAILABLE'}")
                lines.append(f"  rerank_score: {f'{rerank_s:.6f}' if rerank_s is not None else 'NOT_AVAILABLE'}")
            self._write_log_entry("DEBUG", "RERANK_COMPARISON", lines, ctx)
        except Exception:
            pass

    def log_reranker_fallback(self, status: str, fallback: str, reason: str, retry_count: int):
        """Log fallback to RRF ordering when Jina API fails."""
        try:
            ctx = self.get_context()
            lines = [
                "[RERANKER FALLBACK]",
                f"status: {status}",
                f"fallback = {fallback}",
                f"reason: {reason}",
                f"retry_count: {retry_count}",
                "Using original RRF ordering."
            ]
            self._write_log_entry("WARNING", "RERANKER_FALLBACK", lines, ctx)
        except Exception:
            pass

    def log_reranker_disabled(self):
        """Log when reranker is disabled by configuration."""
        try:
            ctx = self.get_context()
            lines = [
                "RERANKER",
                "enabled: false",
                "status: SKIPPED",
                "reason: feature_flag_disabled",
                "Using original RRF results."
            ]
            self._write_log_entry("INFO", "RERANKER_DISABLED", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 7. STAGE 8 ORCHESTRATION
    # =========================================================================

    def log_stage8_reconstruction(self, input_candidate_count: int,
                                  problems: List[Dict[str, Any]], model: Optional[str],
                                  retrieval_mode: str, latency_ms: Optional[float] = None):
        """Log problem and step reconstruction in Stage 8."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.stage8_ms = latency_ms
                
            total_steps = sum(len(p.get("steps", [])) for p in problems)
            lines = [
                "-------------------- STAGE 8 — RETRIEVAL ORCHESTRATION -------------",
                f"input_candidate_count = {input_candidate_count}",
                f"problem_count = {len(problems)}",
                f"step_count = {total_steps}",
                f"model = {model}",
                f"retrieval_mode = {retrieval_mode}"
            ]
            
            for p_idx, prob in enumerate(problems, start=1):
                rel = prob.get("relevance", {})
                max_rerank = rel.get("max_rerank_score")
                max_rerank_str = f"{max_rerank:.6f}" if max_rerank is not None else "NOT_AVAILABLE"
                
                lines.append(f"\nPROBLEM #{p_idx:02d}")
                lines.append("-----------")
                lines.append(f"problem_id: {prob.get('problem_id')}")
                lines.append(f"problem_name: {prob.get('problem_name')}")
                lines.append(f"document_id: {prob.get('document_id')}")
                lines.append(f"chunk_count: {rel.get('supporting_chunk_count', len(prob.get('supporting_chunks', [])))}")
                lines.append(f"max_rerank_score: {max_rerank_str}")
                
                for s_idx, step in enumerate(prob.get("steps", []), start=1):
                    src = step.get("source", {})
                    c_ids = src.get("chunk_ids", [])
                    pages = src.get("pages", [])
                    preview, truncated = self._format_preview(step.get("step_text", ""))
                    
                    lines.append(f"\n  STEP #{s_idx:02d}")
                    lines.append("  --------")
                    lines.append(f"  step_id: {step.get('step_id')}")
                    lines.append(f"  step_number: {step.get('step_number')}")
                    lines.append(f"  instruction_preview: {preview}")
                    lines.append(f"  chunk_ids: {c_ids}")
                    lines.append(f"  page_numbers: {pages}")
                    lines.append(f"  source_document: {src.get('document_id')}")

            self._write_log_entry("DEBUG", "STAGE8", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 8. IMAGE RETRIEVAL & FALLBACK
    # =========================================================================

    def log_image_step_retrieval(self, step_id: str, step_number: Any, problem_name: str,
                                 image_query: str, model: Optional[str],
                                 retrieval_mode: str, top_k: int,
                                 all_candidates: List[Dict[str, Any]],
                                 final_images: List[Dict[str, Any]],
                                 fallback_used: bool = False,
                                 fallback_reason: Optional[str] = None):
        """Log image candidates, scores, fallback, and paths for a step."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.image_called = True
                
            step_num_str = f"{step_number:02d}" if isinstance(step_number, int) else str(step_number)
            lines = [
                "-------------------- IMAGE RETRIEVAL -------------------------------",
                f"[STEP {step_num_str}]",
                f"step_id: {step_id}",
                f"step_number: {step_number}",
                f"problem: {problem_name}",
                f"image_query: {image_query}",
                f"model: {model}",
                f"retrieval_mode: {retrieval_mode}",
                f"image_top_k: {top_k}"
            ]
            
            if fallback_used:
                lines.append(f"IMAGE FALLBACK: requested_scope=model_specific, fallback={fallback_reason}")
                
            for idx, img in enumerate(final_images, start=1):
                fp = img.get("file_path", "")
                sem_score = img.get("semantic_score", 0.0)
                
                lines.append(f"\nIMAGE RESULT #{idx:02d}")
                lines.append("----------------")
                lines.append(f"image_id: {img.get('image_id')}")
                lines.append(f"file_path: {fp}")
                lines.append(f"image_scope: {img.get('image_scope')}")
                lines.append(f"model: {img.get('model')}")
                lines.append(f"problem_name: {img.get('original_problem_name') or problem_name}")
                lines.append(f"step_number: {img.get('original_step_number') or step_number}")
                lines.append(f"step_id: {img.get('original_step_id') or step_id}")
                lines.append(f"semantic_score: {sem_score:.4f}")
                lines.append(f"step_match: {str(img.get('step_match', False)).lower()}")

            self._write_log_entry("DEBUG", "IMAGE_RETRIEVAL", lines, ctx)
        except Exception:
            pass

    def log_image_deduplication(self, before_count: int, after_count: int,
                                removed_ids: List[str]):
        """Log image deduplication event."""
        try:
            ctx = self.get_context()
            lines = [
                "IMAGE DEDUPLICATION",
                f"before: {before_count}",
                f"after: {after_count}",
                f"removed: {len(removed_ids)}",
                f"image_ids: {removed_ids}"
            ]
            self._write_log_entry("DEBUG", "IMAGE_DEDUP", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 9. GROUNDING VALIDATION
    # =========================================================================

    def log_grounding_validation(self, status: str, steps_received: int,
                                 steps_accepted: int, steps_rejected: int,
                                 errors: List[str], rejected_details: Optional[List[Dict[str, Any]]] = None):
        """Log strict grounding validation checks."""
        try:
            ctx = self.get_context()
            lines = [
                "-------------------- GROUNDING VALIDATION --------------------------",
                f"status: {status}",
                f"steps_received: {steps_received}",
                f"steps_accepted: {steps_accepted}",
                f"steps_rejected: {steps_rejected}",
                f"errors: {errors}"
            ]
            
            if rejected_details:
                for r in rejected_details:
                    lines.append(f"  step_id: {r.get('step_id')}")
                    lines.append(f"  reason: {r.get('reason')}")
                    
            self._write_log_entry("INFO", "GROUNDING", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 10. LLM GENERATION & RESPONSE
    # =========================================================================

    def log_llm_request(self, provider: str, model: str, mode: str,
                        model_known: bool, guidance_scope: str,
                        prompt: str, evidence_counts: Dict[str, int]):
        """Log LLM prompt, tokens/hashes, and evidence counts without API keys."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.llm_called = True
                
            prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            
            lines = [
                "-------------------- LLM GENERATION --------------------------------",
                f"provider: {provider}",
                f"model: {model}",
                f"mode: {mode}",
                f"model_known: {str(model_known).lower()}",
                f"guidance_scope: {guidance_scope}",
                f"evidence_problem_count: {evidence_counts.get('problems', 0)}",
                f"evidence_step_count: {evidence_counts.get('steps', 0)}",
                f"evidence_chunk_count: {evidence_counts.get('chunks', 0)}",
                f"prompt_length: {len(prompt)}",
                f"prompt_hash: {prompt_hash}"
            ]
            
            if self.is_full_trace:
                lines.append(f"FULL_PROMPT:\n{prompt}")
                
            self._write_log_entry("INFO", "LLM_REQUEST", lines, ctx)
        except Exception:
            pass

    def log_llm_response(self, status: str, response_length: int,
                         parsed_successfully: bool, step_count: int,
                         image_count: int, raw_response: Optional[str] = None,
                         latency_ms: Optional[float] = None):
        """Log raw/parsed LLM response and latency."""
        try:
            ctx = self.get_context()
            if ctx and latency_ms is not None:
                ctx.llm_latency_ms = latency_ms
                
            lines = [
                "-------------------- LLM RESPONSE ----------------------------------",
                f"status: {status}",
                f"response_length: {response_length}",
                f"parsed_successfully: {str(parsed_successfully).lower()}",
                f"step_count: {step_count}",
                f"image_count: {image_count}"
            ]
            if latency_ms is not None:
                lines.append(f"latency_ms: {round(latency_ms, 2)}")
                
            if self.is_full_trace and raw_response:
                lines.append(f"RAW_RESPONSE:\n{raw_response}")
                
            self._write_log_entry("INFO", "LLM_RESPONSE", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 11. FINAL API RESPONSE
    # =========================================================================

    def log_final_response(self, result: Dict[str, Any], total_latency_ms: float):
        """Log final response summary / payload before returning to Flask."""
        try:
            ctx = self.get_context()
            if ctx:
                ctx.total_latency_ms = total_latency_ms
                
            status = result.get("status", "unknown")
            steps = result.get("steps", [])
            img_count = sum(len(s.get("images", [])) for s in steps if isinstance(s, dict))
            
            lines = [
                "-------------------- FINAL RESPONSE --------------------------------",
                f"status: {status}",
                f"model: {result.get('model')}",
                f"model_known: {str(result.get('model_known')).lower()}",
                f"guidance_scope: {result.get('guidance_scope')}",
                f"warning: {result.get('warning')}",
                f"step_count: {len(steps)}",
                f"image_count: {img_count}",
                f"grounding: {result.get('grounding', {}).get('grounded', False)}",
                f"total_latency_ms: {round(total_latency_ms, 2)}"
            ]
            
            if self.is_full_trace:
                lines.append(f"FINAL_JSON:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                
            self._write_log_entry("INFO", "FINAL_RESPONSE", lines, ctx)
        except Exception:
            pass

    # =========================================================================
    # 12. ERROR & EXCEPTION LOGGING
    # =========================================================================

    def log_error(self, stage: str, exc: Exception, details: Optional[str] = None):
        """Log pipeline exception with sanitized message."""
        try:
            ctx = self.get_context()
            lines = [
                "[ERROR]",
                f"STAGE = {stage.upper()}",
                f"TYPE = {type(exc).__name__}",
                f"MESSAGE = {redact_sensitive(str(exc))}"
            ]
            if details:
                lines.append(f"DETAILS = {redact_sensitive(str(details))}")
            self._write_log_entry("ERROR", stage, lines, ctx)
        except Exception:
            pass


# Global singleton instance
pipeline_logger = PipelineLogger()
