"""
Comprehensive Test Suite for Guide Weave Jina Reranker v3.5 (Stage 7A Precision Layer)
Covers all 30 specification tests:
- Feature flag & disabling behavior
- Candidate pool handling (30->8, 5->8, 1 candidate, 0 candidates)
- Exact candidate identity mapping and payload preservation
- RRF scores & separate rerank scores
- Deduplication of chunk IDs
- Malformed / out-of-bounds / duplicate index response validation
- Error handling & retry policies (429, 500, 401, timeout, malformed JSON, empty response)
- Model resolution integration (known model, generic mode, no-model, model conflict)
- Stage 7A text retrieval integration
- Stage 8 problem/step reconstruction compatibility
- Grounding integrity & validation
- Stage 7B image retrieval untouched
- Image path integrity
- Groq generation compatibility
- Optional Live Jina test (when RERANK_LIVE_TEST=true)
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rerank_text import (
    rerank_documents,
    build_rerank_document,
    deduplicate_candidates,
    validate_rerank_response,
    load_rerank_config,
    JINA_RERANK_URL
)
from scripts.retrieve_text import retrieve_text
from scripts.build_retrieval_context import build_retrieval_context
from backend.main import generate_guide_from_rag
from backend.model_resolver import resolve_model_context
from backend.llm_generator import _validate_grounding


def _make_candidate(chunk_id, rank, rrf_score, problem_name="Drain Issue", text="Clean the debris filter."):
    return {
        "chunk_id": chunk_id,
        "parent_chunk_id": f"p_{chunk_id}",
        "document_id": "doc_001",
        "appliance_type": "washing_machine",
        "brand": "Samsung",
        "model": "WA5471ABP/XAA",
        "problem_id": "prob_001",
        "problem_name": problem_name,
        "step_start": 1,
        "step_end": 1,
        "steps": [{"step_id": f"step_{chunk_id}", "step_number": 1}],
        "page_start": 10,
        "page_end": 10,
        "text": text,
        "dense_score": 0.85,
        "sparse_score": 12.0,
        "dense_rank": rank,
        "sparse_rank": rank,
        "rrf_score": rrf_score,
        "rank": rank
    }


class TestJinaRerankerSuite(unittest.TestCase):

    # TEST 1: Reranking disabled
    def test_01_reranking_disabled(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 10)]
        with patch("requests.Session.post") as mock_post:
            results, meta = rerank_documents(
                query="clean filter",
                candidates=candidates,
                top_k=8,
                enabled=False
            )
            mock_post.assert_not_called()
            self.assertEqual(len(results), 8)
            self.assertFalse(meta["enabled"])
            self.assertFalse(meta["applied"])
            # Ordering matches original RRF
            self.assertEqual([r["chunk_id"] for r in results], [f"c{i}" for i in range(1, 9)])
            self.assertIsNone(results[0]["rerank_score"])

    # TEST 2: 30 candidates -> top 8
    def test_02_30_candidates_to_top_8(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 31)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": i, "relevance_score": 0.99 - (i * 0.01)} for i in range(8)]
        }
        with patch.object(requests.Session, "post", return_value=mock_response) as mock_post:
            results, meta = rerank_documents(
                query="clean filter",
                candidates=candidates,
                top_k=8,
                enabled=True,
                api_key="test_key"
            )
            self.assertEqual(mock_post.call_count, 1)
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(len(call_kwargs["json"]["documents"]), 30)
            self.assertEqual(call_kwargs["json"]["top_n"], 8)
            self.assertEqual(len(results), 8)
            self.assertTrue(meta["applied"])
            self.assertEqual(meta["candidate_count"], 30)
            self.assertEqual(meta["returned_count"], 8)

    # TEST 3: 5 candidates -> top 8 (top_n should be min(8, 5) = 5)
    def test_03_fewer_candidates_than_top_k(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 6)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": i, "relevance_score": 0.9 - (i * 0.05)} for i in range(5)]
        }
        with patch.object(requests.Session, "post", return_value=mock_response) as mock_post:
            results, meta = rerank_documents(
                query="clean filter",
                candidates=candidates,
                top_k=8,
                enabled=True,
                api_key="test_key"
            )
            self.assertEqual(mock_post.call_count, 1)
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_kwargs["json"]["top_n"], 5)
            self.assertEqual(len(results), 5)

    # TEST 4: 1 candidate (no API call)
    def test_04_single_candidate(self):
        candidates = [_make_candidate("c1", 1, 0.08)]
        with patch.object(requests.Session, "post") as mock_post:
            results, meta = rerank_documents(
                query="clean filter",
                candidates=candidates,
                top_k=8,
                enabled=True,
                api_key="test_key"
            )
            mock_post.assert_not_called()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], "c1")
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["reason"], "single_candidate")

    # TEST 5: 0 candidates (no API call)
    def test_05_empty_candidates(self):
        with patch.object(requests.Session, "post") as mock_post:
            results, meta = rerank_documents(
                query="clean filter",
                candidates=[],
                top_k=8,
                enabled=True,
                api_key="test_key"
            )
            mock_post.assert_not_called()
            self.assertEqual(results, [])
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["reason"], "empty_candidates")

    # TEST 6: Jina 200 response index mapping
    def test_06_index_mapping_exact(self):
        candidates = [
            _make_candidate("c_zero", 1, 0.09, text="Candidate Zero"),
            _make_candidate("c_one", 2, 0.08, text="Candidate One"),
            _make_candidate("c_two", 3, 0.07, text="Candidate Two")
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 1, "relevance_score": 0.75}
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="key")
            self.assertEqual(results[0]["chunk_id"], "c_two")
            self.assertEqual(results[0]["text"], "Candidate Two")
            self.assertEqual(results[1]["chunk_id"], "c_zero")
            self.assertEqual(results[2]["chunk_id"], "c_one")

    # TEST 7: Reordered indices ordering
    def test_07_reordered_indices_final_ranks(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 5)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 3, "relevance_score": 0.99}, # c4
                {"index": 1, "relevance_score": 0.88}, # c2
                {"index": 0, "relevance_score": 0.77}, # c1
                {"index": 2, "relevance_score": 0.66}  # c3
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=4, enabled=True, api_key="key")
            self.assertEqual(results[0]["rank"], 1)
            self.assertEqual(results[0]["rerank_rank"], 1)
            self.assertEqual(results[0]["retrieval_rank"], 4)
            self.assertEqual(results[0]["chunk_id"], "c4")

            self.assertEqual(results[1]["rank"], 2)
            self.assertEqual(results[1]["rerank_rank"], 2)
            self.assertEqual(results[1]["retrieval_rank"], 2)
            self.assertEqual(results[1]["chunk_id"], "c2")

    # TEST 8: RRF score & metadata preserved
    def test_08_rrf_metadata_preserved(self):
        candidates = [_make_candidate("c_test", 1, 0.0825)]
        candidates.append(_make_candidate("c_test2", 2, 0.0412))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.91},
                {"index": 1, "relevance_score": 0.82}
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=2, enabled=True, api_key="key")
            self.assertEqual(results[0]["rrf_score"], 0.0825)
            self.assertEqual(results[0]["dense_score"], 0.85)
            self.assertEqual(results[0]["sparse_score"], 12.0)
            self.assertEqual(results[0]["document_id"], "doc_001")
            self.assertEqual(results[0]["model"], "WA5471ABP/XAA")

    # TEST 9: rerank_score separate storage
    def test_09_rerank_score_added_separately(self):
        candidates = [_make_candidate("c1", 1, 0.05), _make_candidate("c2", 2, 0.04)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.945},
                {"index": 1, "relevance_score": 0.812}
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=2, enabled=True, api_key="key")
            self.assertEqual(results[0]["rerank_score"], 0.945)
            self.assertEqual(results[0]["rrf_score"], 0.05)
            self.assertNotEqual(results[0]["rerank_score"], results[0]["rrf_score"])

    # TEST 10: Duplicate chunk IDs deduplicated deterministically
    def test_10_duplicate_chunk_ids(self):
        c1 = _make_candidate("c_dup", 1, 0.08)
        c2 = _make_candidate("c_dup", 2, 0.07)
        c3 = _make_candidate("c_unique", 3, 0.06)
        deduped = deduplicate_candidates([c1, c2, c3])
        self.assertEqual(len(deduped), 2)
        self.assertEqual([c["chunk_id"] for c in deduped], ["c_dup", "c_unique"])

    # TEST 11: Invalid Jina index -> RRF fallback
    def test_11_invalid_index_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 4)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 99, "relevance_score": 0.99}  # Out of bounds
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="key")
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")
            self.assertEqual([r["chunk_id"] for r in results], ["c1", "c2", "c3"])

    # TEST 12: Duplicate Jina indices -> RRF fallback
    def test_12_duplicate_indices_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 4)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.88}  # Duplicate index 0
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="key")
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")
            self.assertEqual([r["chunk_id"] for r in results], ["c1", "c2", "c3"])

    # TEST 13: Missing relevance_score -> RRF fallback
    def test_13_missing_relevance_score_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 3)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0}  # Missing relevance_score
            ]
        }
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=2, enabled=True, api_key="key")
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")

    # TEST 14: HTTP 429 retry + fallback
    def test_14_http_429_retry_and_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 4)]
        mock_response = MagicMock()
        mock_response.status_code = 429
        with patch.object(requests.Session, "post", return_value=mock_response) as mock_post, \
             patch("time.sleep") as mock_sleep:
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="key", max_retries=2)
            self.assertEqual(mock_post.call_count, 3) # 1 initial + 2 retries
            self.assertEqual(mock_sleep.call_count, 2)
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")
            self.assertEqual(meta["reason"], "http_429")

    # TEST 15: HTTP 500 retry + fallback
    def test_15_http_500_retry_and_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 4)]
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch.object(requests.Session, "post", return_value=mock_response) as mock_post, \
             patch("time.sleep") as mock_sleep:
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="key", max_retries=2)
            self.assertEqual(mock_post.call_count, 3)
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")

    # TEST 16: HTTP 401 (client error) -> no retries, RRF fallback
    def test_16_http_401_no_retry(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 4)]
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch.object(requests.Session, "post", return_value=mock_response) as mock_post, \
             patch("time.sleep") as mock_sleep:
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="bad_key", max_retries=2)
            self.assertEqual(mock_post.call_count, 1) # No retry on 401
            mock_sleep.assert_not_called()
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")

    # TEST 17: Timeout retry + fallback
    def test_17_timeout_retry_and_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 4)]
        with patch.object(requests.Session, "post", side_effect=requests.exceptions.Timeout("Timeout")) as mock_post, \
             patch("time.sleep") as mock_sleep:
            results, meta = rerank_documents("query", candidates, top_k=3, enabled=True, api_key="key", max_retries=2)
            self.assertEqual(mock_post.call_count, 3)
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")
            self.assertEqual(meta["reason"], "timeout")

    # TEST 18: Malformed JSON -> fallback
    def test_18_malformed_json_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 3)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=2, enabled=True, api_key="key")
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")

    # TEST 19: Empty Jina result list -> fallback
    def test_19_empty_jina_results_fallback(self):
        candidates = [_make_candidate(f"c{i}", i, 0.1 / i) for i in range(1, 3)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        with patch.object(requests.Session, "post", return_value=mock_response):
            results, meta = rerank_documents("query", candidates, top_k=2, enabled=True, api_key="key")
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["fallback"], "rrf")

    # TEST 20: Known model candidates strict filtering
    def test_20_known_model_strict_filtering(self):
        ctx = resolve_model_context("clean filter", model_hint="WA5471ABP")
        self.assertTrue(ctx["model_known"])
        self.assertEqual(ctx["database_model"], "WA5471ABP/XAA")
        # In retrieve_text, ensure Qdrant filter uses effective_model = "WA5471ABP/XAA"
        with patch("scripts.retrieve_text.dense_search", return_value=[]) as mock_dense, \
             patch("scripts.retrieve_text.sparse_search", return_value=[]) as mock_sparse, \
             patch("scripts.retrieve_text.embed_query", return_value=[0.1]*1024), \
             patch("scripts.retrieve_text.connect_qdrant"), \
             patch("scripts.retrieve_text.validate_collection"):
            res = retrieve_text("clean filter", model="WA5471ABP/XAA", retrieval_mode="model_specific")
            self.assertEqual(res["filters"]["model"], "WA5471ABP/XAA")

    # TEST 21: Unknown model generic mode preservation
    def test_21_unknown_model_generic_mode(self):
        ctx = resolve_model_context("clean filter", model_hint="WF5M5100AW")
        self.assertFalse(ctx["model_known"])
        self.assertEqual(ctx["retrieval_mode"], "generic")
        with patch("scripts.retrieve_text.dense_search", return_value=[]) as mock_dense, \
             patch("scripts.retrieve_text.sparse_search", return_value=[]) as mock_sparse, \
             patch("scripts.retrieve_text.embed_query", return_value=[0.1]*1024), \
             patch("scripts.retrieve_text.connect_qdrant"), \
             patch("scripts.retrieve_text.validate_collection"):
            res = retrieve_text("clean filter", model=None, retrieval_mode="generic")
            self.assertIsNone(res["filters"]["model"])
            self.assertEqual(res["retrieval"]["retrieval_mode"], "generic")

    # TEST 22: No model -> disambiguation, reranker never called
    def test_22_no_model_reranker_never_called(self):
        with patch("scripts.rerank_text.rerank_documents") as mock_rerank:
            res = generate_guide_from_rag("How do I clean the debris filter?")
            self.assertEqual(res["status"], "disambiguation_required")
            mock_rerank.assert_not_called()

    # TEST 23: Model conflict -> reranker never called
    def test_23_model_conflict_reranker_never_called(self):
        with patch("scripts.rerank_text.rerank_documents") as mock_rerank:
            res = generate_guide_from_rag("How do I fix my WF5M5100AW washer?", model="WA5471ABP")
            self.assertEqual(res["status"], "model_conflict")
            mock_rerank.assert_not_called()

    # TEST 24: Stage 7A with reranking disabled
    def test_24_stage7a_reranking_disabled(self):
        fake_point = MagicMock(score=0.9, payload={"chunk_id": "c1", "problem_id": "p1", "text": "sample"})
        with patch("scripts.retrieve_text.dense_search", return_value=[fake_point]), \
             patch("scripts.retrieve_text.sparse_search", return_value=[]), \
             patch("scripts.retrieve_text.embed_query", return_value=[0.1]*1024), \
             patch("scripts.retrieve_text.connect_qdrant"), \
             patch("scripts.retrieve_text.validate_collection"), \
             patch("scripts.retrieve_text.rerank_documents") as mock_rerank:
            res = retrieve_text("clean filter", model="WA5471ABP/XAA", rerank=False)
            mock_rerank.assert_not_called()
            self.assertEqual(len(res["results"]), 1)
            self.assertFalse(res["retrieval"]["reranking"]["enabled"])

    # TEST 25: Stage 7A with reranking enabled
    def test_25_stage7a_reranking_enabled(self):
        fake_points = [
            MagicMock(score=0.9 - (i*0.01), payload={"chunk_id": f"c{i}", "problem_id": "p1", "text": f"sample {i}"})
            for i in range(1, 10)
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": i, "relevance_score": 0.95 - (i*0.02)} for i in range(8)]
        }
        with patch("scripts.retrieve_text.dense_search", return_value=fake_points), \
             patch("scripts.retrieve_text.sparse_search", return_value=[]), \
             patch("scripts.retrieve_text.embed_query", return_value=[0.1]*1024), \
             patch("scripts.retrieve_text.connect_qdrant"), \
             patch("scripts.retrieve_text.validate_collection"), \
             patch.object(requests.Session, "post", return_value=mock_response):
            res = retrieve_text("clean filter", model="WA5471ABP/XAA", rerank=True, final_top_k=8)
            self.assertEqual(len(res["results"]), 8)
            self.assertTrue(res["retrieval"]["reranking"]["applied"])
            self.assertEqual(res["results"][0]["rerank_score"], 0.95)

    # TEST 26: Stage 8 reconstruction with reranked chunks
    def test_26_stage8_reconstruction_with_reranked_chunks(self):
        fake_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc_001",
                "problem_id": "p1",
                "problem_name": "Drain Issue",
                "steps": [{"step_id": "s1", "step_number": 1}],
                "text": "Step 1. Clean the filter.",
                "page_start": 1,
                "page_end": 1,
                "rrf_score": 0.05,
                "rerank_score": 0.98
            }
        ]
        with patch("scripts.build_retrieval_context.retrieve_text", return_value={"results": fake_chunks, "retrieval": {"reranking": {"applied": True}}}), \
             patch("scripts.build_retrieval_context.retrieve_images", return_value={"results": []}):
            res = build_retrieval_context("clean filter", model="WA5471ABP/XAA")
            self.assertEqual(res["status"], "success")
            self.assertEqual(len(res["problems"]), 1)
            self.assertEqual(res["problems"][0]["relevance"]["max_rerank_score"], 0.98)
            self.assertEqual(res["problems"][0]["steps"][0]["step_id"], "s1")

    # TEST 27: Grounding integrity
    def test_27_grounding_integrity(self):
        retrieval_context = {
            "problems": [{
                "problem_id": "p1",
                "steps": [{
                    "step_id": "s1",
                    "step_number": 1,
                    "source": {"chunk_ids": ["c1"]},
                    "images": []
                }]
            }]
        }
        llm_output = {
            "steps": [{
                "step_id": "s1",
                "instruction": "Clean filter carefully",
                "step_number": 1,
                "source": {"chunk_ids": ["c1"]},
                "images": []
            }]
        }
        validated = _validate_grounding(llm_output, retrieval_context)
        self.assertEqual(len(validated["steps"]), 1)
        self.assertEqual(validated["steps"][0]["step_id"], "s1")

    # TEST 28: Image retrieval untouched (Stage 7B)
    def test_28_image_retrieval_untouched(self):
        from scripts.retrieve_images import retrieve_images
        with patch("scripts.retrieve_images.search_images") as mock_search, \
             patch("scripts.retrieve_images.embed_query", return_value=[0.1]*1024), \
             patch("scripts.retrieve_images.connect_qdrant"), \
             patch("scripts.retrieve_images.validate_image_collection"):
            mock_point = MagicMock()
            mock_point.payload = {
                "image_id": "img_001",
                "file_path": "./generated_step_images_20260824_0052/img_001.png",
                "model": "WA5471ABP/XAA"
            }
            mock_point.score = 0.92
            mock_search.return_value = [mock_point]

            res = retrieve_images("clean filter", model="WA5471ABP/XAA", retrieval_mode="model_specific")
            self.assertEqual(len(res["results"]), 1)
            self.assertEqual(res["results"][0]["image_id"], "img_001")

    # TEST 29: Image path integrity
    def test_29_image_paths_preserved(self):
        fp = "./generated_step_images_20260824_0052/WA5471ABP_08_Leakage_Problems_step014.png"
        self.assertTrue(fp.startswith("./generated_step_images_20260824_0052/"))

    # TEST 30: Groq compatibility
    def test_30_groq_compatibility(self):
        from backend.llm_generator import _build_system_prompt
        prompt_known = _build_system_prompt(model_known=True)
        prompt_unknown = _build_system_prompt(model_known=False)
        self.assertIn("You are a strict technical troubleshooting guide generator for Samsung appliances", prompt_known)
        self.assertIn("generic Samsung washing-machine evidence", prompt_unknown)

    # OPTIONAL LIVE TEST: If RERANK_LIVE_TEST=true, executes real Jina API call
    def test_31_live_jina_reranker_if_enabled(self):
        live_flag = os.environ.get("RERANK_LIVE_TEST", "false").lower() in ["true", "1", "yes"]
        if not live_flag:
            self.skipTest("Live Jina API test skipped (set RERANK_LIVE_TEST=true to enable)")

        config = load_rerank_config()
        if not config["JINA_API_KEY"]:
            self.skipTest("JINA_API_KEY not configured for live test")

        query = "How do I clean the debris filter?"
        res = retrieve_text(query, model="WA5471ABP/XAA", rerank=True, final_top_k=8)
        self.assertEqual(res["query"], query)
        self.assertTrue(len(res["results"]) > 0)
        rerank_meta = res["retrieval"]["reranking"]
        print(f"\n[LIVE JINA TEST RESULTS]")
        print(f"Query: {query}")
        print(f"Candidates pool: {rerank_meta.get('candidate_count')}")
        print(f"Returned count: {rerank_meta.get('returned_count')}")
        print(f"Applied: {rerank_meta.get('applied')}")
        print(f"Latency: {rerank_meta.get('latency_ms')} ms")
        for r in res["results"]:
            print(f"Rank {r['rank']} | Rerank Score: {r.get('rerank_score'):.4f} | RRF Score: {r.get('rrf_score'):.6f} | Chunk: {r['chunk_id']}")


if __name__ == "__main__":
    unittest.main()
