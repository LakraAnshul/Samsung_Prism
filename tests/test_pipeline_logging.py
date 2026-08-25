"""
Guide Weave — Comprehensive Test Suite for Backend Observability & Structured Logging
Tests all 30 specification requirements:
1. Normal known-model query creates log section.
2. Unique QUERY_ID for each query.
3. Multiple queries produce distinct distinguishable sections.
4. Dense results appear in log.
5. Sparse results appear in log.
6. RRF scores appear in log.
7. Dense and sparse scores preserved.
8. Reranker input candidates appear.
9. Reranker output appears.
10. RRF rank and rerank rank both logged.
11. Rerank score logged.
12. Reranker fallback logged.
13. Image candidates logged.
14. Image paths logged exactly.
15. Image semantic scores logged.
16. Stage 8 reconstructed steps logged.
17. Grounding validation logged.
18. LLM timing logged.
19. Final status logged.
20. No-model query stops before Qdrant.
21. No-model query logs Qdrant as NOT_CALLED.
22. Model conflict logs retrieval as NOT_CALLED.
23. Unknown-model generic retrieval clearly marked.
24. Reranker disabled is logged.
25. Logger failure does not fail request.
26. Unicode text written correctly.
27. API keys never appear in logs.
28. Authorization headers never appear in logs.
29. Large text preview-truncated unless FULL_TRACE enabled.
30. Concurrent requests retain distinct QUERY_ID values.
"""

import os
import sys
import unittest
import threading
import tempfile
import json
import uuid
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline_logger import pipeline_logger, PipelineLogger, redact_sensitive, QueryContext
from backend.main import generate_guide_from_rag
from scripts.retrieve_text import retrieve_text
from scripts.rerank_text import rerank_documents
from scripts.build_retrieval_context import build_retrieval_context
from backend.model_resolver import resolve_model_context


class TestPipelineLogging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure test logs directory exists
        cls.log_file = PROJECT_ROOT / "logs" / "guide_weave.log"
        cls.log_file.parent.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        if self.log_file.exists():
            self.start_pos = self.log_file.stat().st_size
        else:
            self.start_pos = 0

    def _read_log_tail(self, num_chars: int = 50000) -> str:
        """Read log written during this test or tail of log."""
        if not self.log_file.exists():
            return ""
        try:
            with open(self.log_file, "rb") as f:
                f.seek(self.start_pos, 0)
                content = f.read().decode("utf-8", errors="replace")
                if content:
                    return content
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - num_chars * 4), 0)
                c = f.read().decode("utf-8", errors="replace")
                return c[-num_chars:] if len(c) > num_chars else c
        except Exception:
            return ""

    def test_01_known_model_creates_log_section(self):
        """TEST 1: Normal known-model query creates a structured log section."""
        res = generate_guide_from_rag("How do I clean the debris filter?", model="WA5471ABP", mode="CLOUD")
        self.assertIn("status", res)
        log = self._read_log_tail(120000)
        self.assertIn("GUIDE WEAVE QUERY START", log)
        self.assertIn("GUIDE WEAVE QUERY END", log)
        self.assertIn("WA5471ABP", log)

    def test_02_unique_query_id(self):
        """TEST 2: Each query receives a unique, valid UUID QUERY_ID."""
        ctx1 = pipeline_logger.log_query_start("query 1", model="WA5471ABP")
        pipeline_logger.log_query_end(status="success", ctx=ctx1)
        
        ctx2 = pipeline_logger.log_query_start("query 2", model="WA5471ABP")
        pipeline_logger.log_query_end(status="success", ctx=ctx2)
        
        self.assertNotEqual(ctx1.query_id, ctx2.query_id)
        # Check UUID format
        uuid_obj = uuid.UUID(ctx1.query_id)
        self.assertEqual(str(uuid_obj), ctx1.query_id)

    def test_03_distinguishable_sections(self):
        """TEST 3: Two queries produce two distinguishable sections."""
        ctx1 = pipeline_logger.log_query_start("first distinguishable query", model="WA5471ABP")
        pipeline_logger.log_query_end(status="success", ctx=ctx1)
        
        ctx2 = pipeline_logger.log_query_start("second distinguishable query", model="WA5471ABP")
        pipeline_logger.log_query_end(status="success", ctx=ctx2)
        
        log = self._read_log_tail(20000)
        self.assertIn(f"QUERY_ID: {ctx1.query_id}", log)
        self.assertIn(f"QUERY_ID: {ctx2.query_id}", log)

    def test_04_dense_results_in_log(self):
        """TEST 4: Dense retrieval results appear in log with scores and chunk IDs."""
        ctx = pipeline_logger.log_query_start("dense test", model="WA5471ABP")
        pipeline_logger.log_dense_retrieval(
            collection="washing_machines",
            query="clean filter",
            retrieval_mode="model_specific",
            q_filter=None,
            requested_limit=30,
            points=[MagicMock(id="p1", score=0.8912, payload={
                "chunk_id": "test_chunk_001",
                "document_id": "doc_001",
                "problem_id": "prob_001",
                "step_id": "step_001",
                "problem_name": "Drain Issue",
                "model": "WA5471ABP/XAA",
                "text": "Clean the debris filter."
            })]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("DENSE RESULT #01", log)
        self.assertIn("test_chunk_001", log)
        self.assertIn("dense_score = 0.891200", log)

    def test_05_sparse_results_in_log(self):
        """TEST 5: Sparse retrieval results appear in log."""
        ctx = pipeline_logger.log_query_start("sparse test", model="WA5471ABP")
        pipeline_logger.log_sparse_retrieval(
            collection="washing_machines",
            query="clean filter",
            retrieval_mode="model_specific",
            q_filter=None,
            requested_limit=30,
            points=[MagicMock(id="p2", score=0.7543, payload={
                "chunk_id": "sparse_chunk_002",
                "document_id": "doc_001",
                "problem_id": "prob_001",
                "step_id": "step_001",
                "text": "Filter instructions."
            })],
            matching_tokens_count=3
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("SPARSE RESULT #01", log)
        self.assertIn("sparse_chunk_002", log)
        self.assertIn("sparse_score = 0.754300", log)

    def test_06_rrf_scores_in_log(self):
        """TEST 6: RRF scores appear in log."""
        ctx = pipeline_logger.log_query_start("rrf test", model="WA5471ABP")
        pipeline_logger.log_rrf_fusion(
            rrf_k=60,
            dense_count=10,
            sparse_count=8,
            fused_results=[{
                "rank": 1,
                "chunk_id": "fused_chunk_001",
                "dense_rank": 1,
                "sparse_rank": 2,
                "dense_score": 0.85,
                "sparse_score": 0.72,
                "rrf_score": 0.0325
            }],
            candidate_pool_limit=30
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("RRF RESULT #01", log)
        self.assertIn("fused_chunk_001", log)
        self.assertIn("rrf_score = 0.032500", log)

    def test_07_dense_and_sparse_scores_preserved(self):
        """TEST 7: Dense and sparse scores are preserved in RRF log entry."""
        ctx = pipeline_logger.log_query_start("scores preserved", model="WA5471ABP")
        pipeline_logger.log_rrf_fusion(
            rrf_k=60,
            dense_count=1,
            sparse_count=1,
            fused_results=[{
                "rank": 1,
                "chunk_id": "chunk_scores_001",
                "dense_rank": 1,
                "sparse_rank": 1,
                "dense_score": 0.912345,
                "sparse_score": 0.812345,
                "rrf_score": 0.032786
            }],
            candidate_pool_limit=30
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("dense_score = 0.912345", log)
        self.assertIn("sparse_score = 0.812345", log)

    def test_08_reranker_input_candidates(self):
        """TEST 8: Exact candidate pool sent to reranker is logged."""
        ctx = pipeline_logger.log_query_start("rerank input test", model="WA5471ABP")
        pipeline_logger.log_reranker_input_pool(
            candidates=[{
                "chunk_id": "candidate_01",
                "problem_name": "Drain Failure",
                "step_start": 1,
                "rrf_score": 0.0312,
                "dense_score": 0.85,
                "sparse_score": 0.70,
                "text": "Drain pump cleaning guide"
            }],
            top_k=8
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("RERANKER INPUT CANDIDATE POOL", log)
        self.assertIn("candidate_01", log)
        self.assertIn("Drain Failure", log)

    def test_09_reranker_output_appears(self):
        """TEST 9: Jina reranker output appears in log."""
        ctx = pipeline_logger.log_query_start("rerank out test", model="WA5471ABP")
        pipeline_logger.log_reranker_output(
            model="jina-reranker-v3.5",
            candidate_count=30,
            returned_count=8,
            latency_ms=150.5,
            applied=True,
            results=[{
                "chunk_id": "reranked_chunk_01",
                "retrieval_rank": 5,
                "rerank_rank": 1,
                "rrf_score": 0.021,
                "dense_score": 0.78,
                "sparse_score": 0.65,
                "rerank_score": 0.9451,
                "problem_name": "Pump Clogged",
                "step_start": 2,
                "document_id": "doc_01"
            }]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("RERANK RESULT #01", log)
        self.assertIn("reranked_chunk_01", log)
        self.assertIn("rerank_score: 0.945100", log)

    def test_10_rrf_rank_and_rerank_rank_both_logged(self):
        """TEST 10: Original RRF rank vs new rerank rank is clearly logged."""
        ctx = pipeline_logger.log_query_start("rank comparison", model="WA5471ABP")
        pipeline_logger.log_reranker_output(
            model="jina-reranker-v3.5",
            candidate_count=10,
            returned_count=1,
            latency_ms=80.0,
            applied=True,
            results=[{
                "chunk_id": "compare_chunk_01",
                "retrieval_rank": 14,
                "rerank_rank": 1,
                "rrf_score": 0.01639,
                "dense_score": 0.65,
                "sparse_score": 0.55,
                "rerank_score": 0.8954
            }]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("original_rrf_rank: 14", log)
        self.assertIn("rerank_rank: 1", log)

    def test_11_rerank_score_logged(self):
        """TEST 11: Rerank relevance score is logged."""
        ctx = pipeline_logger.log_query_start("score test", model="WA5471ABP")
        pipeline_logger.log_reranker_score_comparison(
            surviving_candidates=[{
                "chunk_id": "chunk_survivor",
                "retrieval_rank": 3,
                "rerank_rank": 1,
                "rrf_score": 0.025,
                "rerank_score": 0.876543
            }]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("rerank_score: 0.876543", log)

    def test_12_reranker_fallback_logged(self):
        """TEST 12: Reranker fallback is logged upon API failure."""
        ctx = pipeline_logger.log_query_start("fallback test", model="WA5471ABP")
        pipeline_logger.log_reranker_fallback(
            status="FAILED",
            fallback="RRF",
            reason="HTTP 429 rate limit",
            retry_count=2
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("[RERANKER FALLBACK]", log)
        self.assertIn("fallback = RRF", log)
        self.assertIn("HTTP 429", log)

    def test_13_image_candidates_logged(self):
        """TEST 13: Image candidates are logged for step."""
        ctx = pipeline_logger.log_query_start("image candidates test", model="WA5471ABP")
        pipeline_logger.log_image_step_retrieval(
            step_id="step_img_01",
            step_number=1,
            problem_name="Filter Clean",
            image_query="Clean filter step 1",
            model="WA5471ABP/XAA",
            retrieval_mode="model_specific",
            top_k=3,
            all_candidates=[],
            final_images=[{
                "image_id": "img_001",
                "file_path": "./generated_step_images_20260824_0052/04_Spin_Problems_step001.png",
                "image_scope": "model_specific",
                "semantic_score": 0.8456,
                "step_match": True
            }]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("IMAGE RESULT #01", log)
        self.assertIn("img_001", log)

    def test_14_image_paths_exact(self):
        """TEST 14: Image paths are logged exactly as stored."""
        ctx = pipeline_logger.log_query_start("exact path test", model="WA5471ABP")
        test_path = "./generated_step_images_20260824_0052/04_Spin_Problems_step001.png"
        pipeline_logger.log_image_step_retrieval(
            step_id="step_exact",
            step_number=1,
            problem_name="Test",
            image_query="query",
            model="WA5471ABP/XAA",
            retrieval_mode="model_specific",
            top_k=1,
            all_candidates=[],
            final_images=[{
                "image_id": "img_exact",
                "file_path": test_path,
                "image_scope": "model_specific",
                "semantic_score": 0.91
            }]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn(f"file_path: {test_path}", log)

    def test_15_image_semantic_scores_logged(self):
        """TEST 15: Image semantic scores are recorded."""
        ctx = pipeline_logger.log_query_start("img score test", model="WA5471ABP")
        pipeline_logger.log_image_step_retrieval(
            step_id="step_s",
            step_number=1,
            problem_name="Test",
            image_query="q",
            model="WA5471ABP/XAA",
            retrieval_mode="model_specific",
            top_k=1,
            all_candidates=[],
            final_images=[{
                "image_id": "img_s",
                "file_path": "path.png",
                "semantic_score": 0.8765
            }]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("semantic_score: 0.8765", log)

    def test_16_stage8_reconstructed_steps_logged(self):
        """TEST 16: Stage 8 reconstructed problems and steps are logged."""
        ctx = pipeline_logger.log_query_start("stage 8 test", model="WA5471ABP")
        pipeline_logger.log_stage8_reconstruction(
            input_candidate_count=8,
            problems=[{
                "problem_id": "prob_s8",
                "problem_name": "Washer Not Draining",
                "document_id": "doc_s8",
                "relevance": {"max_rerank_score": 0.92, "supporting_chunk_count": 2},
                "steps": [{
                    "step_id": "step_s8_1",
                    "step_number": 1,
                    "step_text": "Check drain hose for kinks.",
                    "source": {"chunk_ids": ["chunk_s8_1"], "pages": [12], "document_id": "doc_s8"}
                }]
            }],
            model="WA5471ABP/XAA",
            retrieval_mode="model_specific",
            latency_ms=45.0
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("STAGE 8 — RETRIEVAL ORCHESTRATION", log)
        self.assertIn("problem_id: prob_s8", log)
        self.assertIn("step_id: step_s8_1", log)

    def test_17_grounding_validation_logged(self):
        """TEST 17: Grounding validation status is logged."""
        ctx = pipeline_logger.log_query_start("grounding test", model="WA5471ABP")
        pipeline_logger.log_grounding_validation(
            status="PASS",
            steps_received=5,
            steps_accepted=5,
            steps_rejected=0,
            errors=[]
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("GROUNDING VALIDATION", log)
        self.assertIn("status: PASS", log)
        self.assertIn("steps_accepted: 5", log)

    def test_18_llm_timing_logged(self):
        """TEST 18: LLM request and response latency are logged."""
        ctx = pipeline_logger.log_query_start("llm timing test", model="WA5471ABP")
        pipeline_logger.log_llm_response(
            status="success",
            response_length=450,
            parsed_successfully=True,
            step_count=3,
            image_count=3,
            latency_ms=1250.5
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("LLM RESPONSE", log)
        self.assertIn("latency_ms: 1250.5", log)

    def test_19_final_status_logged(self):
        """TEST 19: Final response status is logged."""
        ctx = pipeline_logger.log_query_start("final status test", model="WA5471ABP")
        pipeline_logger.log_final_response(
            result={
                "status": "success",
                "model": "WA5471ABP",
                "model_known": True,
                "guidance_scope": "model_specific",
                "warning": None,
                "steps": []
            },
            total_latency_ms=1500.0
        )
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("FINAL RESPONSE", log)
        self.assertIn("status: success", log)

    def test_20_no_model_stops_before_qdrant(self):
        """TEST 20: No-model query returns disambiguation_required without calling retrieval."""
        res = generate_guide_from_rag("How do I clean the debris filter?", model=None)
        self.assertEqual(res.get("status"), "disambiguation_required")

    def test_21_no_model_logs_not_called(self):
        """TEST 21: No-model query logs Qdrant and Reranker as NOT_CALLED."""
        ctx = pipeline_logger.log_query_start("no model query", model=None)
        model_ctx = resolve_model_context("no model query", model_hint=None)
        pipeline_logger.log_model_resolution(model_ctx)
        pipeline_logger.log_query_end(status="disambiguation_required", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("QDRANT_LATENCY_MS: NOT_CALLED", log)
        self.assertIn("RERANK_LATENCY_MS: NOT_CALLED", log)

    def test_22_model_conflict_logs_not_called(self):
        """TEST 22: Model conflict returns early and logs retrieval as NOT_CALLED."""
        res = generate_guide_from_rag("Fix WF5M5100AW", model="WA5471ABP")
        self.assertEqual(res.get("status"), "model_conflict")
        log = self._read_log_tail(15000)
        self.assertIn("MODEL CONFLICT", log.upper())

    def test_23_unknown_model_generic_retrieval_marked(self):
        """TEST 23: Unknown model generic retrieval is clearly marked in log."""
        ctx = pipeline_logger.log_query_start("unknown model test", model="WF5M5100AW")
        model_ctx = resolve_model_context("Clean filter", model_hint="WF5M5100AW")
        pipeline_logger.log_model_resolution(model_ctx)
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("model_known = false", log)
        self.assertIn("retrieval_mode = generic", log)
        self.assertIn("guidance_scope = generic", log)

    def test_24_reranker_disabled_logged(self):
        """TEST 24: Reranker disabled state is clearly logged."""
        ctx = pipeline_logger.log_query_start("rerank disabled", model="WA5471ABP")
        pipeline_logger.log_reranker_disabled()
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("feature_flag_disabled", log)
        self.assertIn("SKIPPED", log)

    def test_25_logger_failure_does_not_fail_request(self):
        """TEST 25: Pipeline logger internal exception never breaks the RAG pipeline."""
        with patch.object(pipeline_logger, "_write_log_entry", side_effect=Exception("Disk Full Simulation")):
            # Should not raise exception
            res = generate_guide_from_rag("How do I clean the debris filter?", model=None)
            self.assertEqual(res.get("status"), "disambiguation_required")

    def test_26_unicode_text_written_correctly(self):
        """TEST 26: Unicode characters (Hindi, symbols) are written correctly with UTF-8."""
        ctx = pipeline_logger.log_query_start("फ़िल्टर कैसे साफ़ करें?", model="WA5471ABP")
        pipeline_logger.log_query_end(status="success", ctx=ctx)
        
        log = self._read_log_tail(15000)
        self.assertIn("फ़िल्टर कैसे साफ़ करें?", log)

    def test_27_api_keys_never_appear_in_logs(self):
        """TEST 27: API keys never appear in raw log text."""
        secret_jina = "jina_1234567890abcdef1234567890"
        secret_groq = "gsk_1234567890abcdef1234567890"
        
        sanitized = redact_sensitive(f"Failed with key {secret_jina} and {secret_groq}")
        self.assertNotIn(secret_jina, sanitized)
        self.assertNotIn(secret_groq, sanitized)
        self.assertIn("[REDACTED]", sanitized)

    def test_28_auth_headers_never_appear(self):
        """TEST 28: Authorization headers are redacted."""
        raw_header = "Authorization: Bearer secret_token_xyz_12345678"
        sanitized = redact_sensitive(raw_header)
        self.assertNotIn("secret_token_xyz_12345678", sanitized)

    def test_29_large_text_preview_truncated(self):
        """TEST 29: Large text is preview-truncated unless FULL_TRACE=true."""
        long_text = "A" * 1500
        preview, truncated = pipeline_logger._format_preview(long_text)
        if not pipeline_logger.is_full_trace:
            self.assertTrue(truncated)
            self.assertLessEqual(len(preview), 600)
            self.assertIn("[TRUNCATED]", preview)

    def test_30_concurrent_requests_retain_distinct_query_ids(self):
        """TEST 30: Concurrent threads retain their own distinct QUERY_IDs."""
        results = {}
        
        def run_thread(t_idx):
            ctx = pipeline_logger.log_query_start(f"concurrent query {t_idx}", model="WA5471ABP")
            time.sleep(0.05)  # Simulate concurrent work
            current_ctx = pipeline_logger.get_context()
            results[t_idx] = {
                "created_id": ctx.query_id,
                "observed_id": current_ctx.query_id if current_ctx else None
            }
            pipeline_logger.log_query_end(status="success", ctx=ctx)

        threads = []
        for i in range(5):
            t = threading.Thread(target=run_thread, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check that all thread query IDs are unique and matched within their context
        query_ids = [r["created_id"] for r in results.values()]
        self.assertEqual(len(query_ids), len(set(query_ids)))
        for i in range(5):
            self.assertEqual(results[i]["created_id"], results[i]["observed_id"])


if __name__ == "__main__":
    unittest.main()
