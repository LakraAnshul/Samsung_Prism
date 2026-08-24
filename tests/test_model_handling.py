"""
Comprehensive Test Suite for Guide Weave Grounded RAG
Covers all 24 test scenarios from the specification:
- Three-state model handling (State A, State B, State C)
- Model conflict edge cases
- Sparse vocabulary loading and edge cases
- Image retrieval scoping and deduplication
- Grounding and relational validation
- Image serving path traversal security
- End-to-end Flask server endpoints
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model_resolver import (
    resolve_model_context,
    extract_models_from_text,
    normalize_model_identifier,
    get_available_database_models
)
from scripts.retrieve_text import SparseTokenizer, reciprocal_rank_fusion
from backend.llm_generator import _validate_grounding, _build_system_prompt
from backend.main import generate_guide_from_rag
from backend.server import app, _is_safe_path, GENERATED_IMAGE_FOLDER


class TestThreeStateModelHandling(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # TEST 1: No model in query and no model field
    def test_01_no_model_disambiguation_required(self):
        with patch("backend.main.build_retrieval_context") as mock_stage8, \
             patch("backend.main.generate_grounded_guide") as mock_llm:
            res = generate_guide_from_rag("How do I clean the debris filter?")
            self.assertEqual(res["status"], "disambiguation_required")
            self.assertIsNone(res["model"])
            self.assertFalse(res["model_known"])
            self.assertIn("message", res)
            mock_stage8.assert_not_called()
            mock_llm.assert_not_called()

    # TEST 2: Known model in explicit model field
    def test_02_known_model_in_field(self):
        ctx = resolve_model_context("How do I clean the filter?", model_hint="WA5471ABP")
        self.assertEqual(ctx["status"], "resolved")
        self.assertTrue(ctx["model_known"])
        self.assertEqual(ctx["retrieval_mode"], "model_specific")
        self.assertEqual(ctx["database_model"], "WA5471ABP/XAA")
        self.assertEqual(ctx["canonical_model"], "WA5471ABP")
        self.assertIsNone(ctx["warning"])

    # TEST 3: Known model in query
    def test_03_known_model_in_query(self):
        ctx = resolve_model_context("How do I fix my WA5471ABP washer?")
        self.assertEqual(ctx["status"], "resolved")
        self.assertTrue(ctx["model_known"])
        self.assertEqual(ctx["retrieval_mode"], "model_specific")
        self.assertEqual(ctx["database_model"], "WA5471ABP/XAA")
        self.assertEqual(ctx["canonical_model"], "WA5471ABP")
        self.assertIsNone(ctx["warning"])

    # TEST 4: Full canonical model in query
    def test_04_full_canonical_model_in_query(self):
        ctx = resolve_model_context("How do I fix my WA5471ABP/XAA washer?")
        self.assertEqual(ctx["status"], "resolved")
        self.assertTrue(ctx["model_known"])
        self.assertEqual(ctx["canonical_model"], "WA5471ABP")
        self.assertEqual(ctx["database_model"], "WA5471ABP/XAA")

    # TEST 5: Unknown model (WF5M5100AW)
    def test_05_unknown_model_generic_mode(self):
        ctx = resolve_model_context("How do I clean the filter?", model_hint="WF5M5100AW")
        self.assertEqual(ctx["status"], "resolved")
        self.assertFalse(ctx["model_known"])
        self.assertEqual(ctx["retrieval_mode"], "generic")
        self.assertIsNone(ctx["database_model"])
        self.assertEqual(ctx["canonical_model"], "WF5M5100AW")
        self.assertIsNotNone(ctx["warning"])
        self.assertEqual(ctx["warning"]["type"], "unknown_model")
        self.assertIn("WF5M5100AW", ctx["warning"]["message"])

    # TEST 6: Completely invalid model (ABC123)
    def test_06_invalid_model_generic_mode(self):
        ctx = resolve_model_context("How do I fix my washer?", model_hint="ABC123")
        self.assertEqual(ctx["status"], "resolved")
        self.assertFalse(ctx["model_known"])
        self.assertEqual(ctx["retrieval_mode"], "generic")
        self.assertIsNotNone(ctx["warning"])
        self.assertIn("ABC123", ctx["warning"]["message"])

    # TEST 7: Known model but irrelevant question (zero results -> no_results)
    def test_07_known_model_zero_results(self):
        fake_retrieval = {
            "status": "no_text_evidence",
            "problems": []
        }
        with patch("backend.main.build_retrieval_context", return_value=fake_retrieval):
            res = generate_guide_from_rag("How do I replace the rocket engine?", model="WA5471ABP")
            self.assertEqual(res["status"], "no_results")
            self.assertEqual(res["model"], "WA5471ABP")
            self.assertTrue(res["model_known"])
            self.assertEqual(res["guidance_scope"], "model_specific")
            self.assertIsNone(res["warning"])

    # TEST 8: Two conflicting models
    def test_08_model_conflict(self):
        with patch("backend.main.build_retrieval_context") as mock_stage8, \
             patch("backend.main.generate_grounded_guide") as mock_llm:
            res = generate_guide_from_rag("How do I fix my WF5M5100AW washer?", model="WA5471ABP")
            self.assertEqual(res["status"], "model_conflict")
            self.assertIn("Two different", res["message"])
            self.assertEqual(set(res["models_detected"]), {"WA5471ABP", "WF5M5100AW"})
            mock_stage8.assert_not_called()
            mock_llm.assert_not_called()

    # TEST 9: Empty query (400)
    def test_09_empty_query(self):
        resp = self.app.post("/api/chat", json={"query": "", "model": "WA5471ABP"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")

    # TEST 10: Unknown model + no generic evidence
    def test_10_unknown_model_no_generic_evidence(self):
        fake_retrieval = {
            "status": "no_text_evidence",
            "problems": []
        }
        with patch("backend.main.build_retrieval_context", return_value=fake_retrieval):
            res = generate_guide_from_rag("Impossible query", model="UNKNOWN999")
            self.assertEqual(res["status"], "no_results")
            self.assertEqual(res["model"], "UNKNOWN999")
            self.assertFalse(res["model_known"])
            self.assertEqual(res["guidance_scope"], "generic")
            self.assertIsNotNone(res["warning"])

    # TEST 11: Generic mode image retrieval scope
    def test_11_generic_mode_image_scope(self):
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
            mock_point.score = 0.85
            mock_search.return_value = [mock_point]

            res = retrieve_images("clean filter", model=None, retrieval_mode="generic")
            self.assertTrue(len(res["results"]) > 0)
            self.assertEqual(res["results"][0]["image_scope"], "generic")

    # TEST 12: Known model image retrieval scope
    def test_12_known_model_image_scope(self):
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
            mock_point.score = 0.90
            mock_search.return_value = [mock_point]

            res = retrieve_images("clean filter", model="WA5471ABP/XAA", retrieval_mode="model_specific")
            self.assertTrue(len(res["results"]) > 0)
            self.assertEqual(res["results"][0]["image_scope"], "model_specific")

    # TEST 13: Sparse vocabulary exists and loads from canonical path
    def test_13_sparse_vocab_loads(self):
        canonical_path = PROJECT_ROOT / "embedding_artifacts" / "sparse_vocabulary.json"
        self.assertTrue(canonical_path.exists())
        tokenizer = SparseTokenizer(canonical_path)
        self.assertTrue(len(tokenizer.vocab) > 0)
        # Verify passing directory does not duplicate filename
        tokenizer_dir = SparseTokenizer(PROJECT_ROOT / "embedding_artifacts")
        self.assertEqual(tokenizer_dir.vocab_path, canonical_path)

    # TEST 14: Sparse vocabulary missing error handling
    def test_14_sparse_vocab_missing(self):
        with self.assertRaises(SystemExit):
            SparseTokenizer(PROJECT_ROOT / "embedding_artifacts" / "non_existent_vocab.json")

    # TEST 15: Sparse vocabulary has no matching tokens (empty sparse dict)
    def test_15_sparse_vocab_no_tokens(self):
        tokenizer = SparseTokenizer()
        tokens = tokenizer.tokenize("zyxwvutsrqponmlkjihgfedcba999")
        self.assertEqual(tokens, {})
        # RRF with dense results and empty sparse results
        mock_dense = [MagicMock(score=0.9, payload={"chunk_id": "c1"})]
        fused = reciprocal_rank_fusion(mock_dense, [])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["chunk_id"], "c1")

    # TEST 16: Both dense and sparse return empty
    def test_16_both_dense_sparse_empty(self):
        fused = reciprocal_rank_fusion([], [])
        self.assertEqual(fused, [])

    # TEST 17: Duplicate image IDs deduplicated
    def test_17_duplicate_image_ids_deduplicated(self):
        from scripts.retrieve_images import rank_results
        p1 = MagicMock(score=0.7)
        p1.payload = {"image_id": "img_dup"}
        p2 = MagicMock(score=0.9)
        p2.payload = {"image_id": "img_dup"}
        ranked = rank_results([p1, p2])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["semantic_score"], 0.9)

    # TEST 18: LLM invents image_id -> rejected
    def test_18_llm_invents_image_id(self):
        retrieval_context = {
            "problems": [{
                "problem_id": "p1",
                "steps": [{
                    "step_id": "s1",
                    "step_number": 1,
                    "source": {"chunk_ids": ["c1"]},
                    "images": [{"image_id": "img_real", "file_path": "./img.png"}]
                }]
            }]
        }
        llm_output = {
            "steps": [{
                "step_id": "s1",
                "instruction": "Do something",
                "step_number": 1,
                "source": {"chunk_ids": ["c1"]},
                "images": [{"image_id": "img_fake"}]
            }]
        }
        validated = _validate_grounding(llm_output, retrieval_context)
        self.assertEqual(len(validated["steps"][0]["images"]), 0)

    # TEST 19: LLM invents chunk_id -> step rejected
    def test_19_llm_invents_chunk_id(self):
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
                "instruction": "Do something",
                "step_number": 1,
                "source": {"chunk_ids": ["c_fake"]},
                "images": []
            }]
        }
        validated = _validate_grounding(llm_output, retrieval_context)
        self.assertEqual(len(validated["steps"]), 0)

    # TEST 20: LLM uses valid chunk_id belonging to another step -> rejected
    def test_20_llm_uses_chunk_from_different_step(self):
        retrieval_context = {
            "problems": [{
                "problem_id": "p1",
                "steps": [
                    {"step_id": "s1", "step_number": 1, "source": {"chunk_ids": ["c1"]}, "images": []},
                    {"step_id": "s2", "step_number": 2, "source": {"chunk_ids": ["c2"]}, "images": []}
                ]
            }]
        }
        llm_output = {
            "steps": [{
                "step_id": "s1",
                "instruction": "Do something",
                "step_number": 1,
                "source": {"chunk_ids": ["c2"]},  # c2 belongs to s2, not s1
                "images": []
            }]
        }
        validated = _validate_grounding(llm_output, retrieval_context)
        self.assertEqual(len(validated["steps"]), 0)

    # TEST 21: LLM returns wrong step_number for valid step_id -> realigned
    def test_21_llm_wrong_step_number(self):
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
                "instruction": "Do something",
                "step_number": 99,
                "source": {"chunk_ids": ["c1"]},
                "images": []
            }]
        }
        validated = _validate_grounding(llm_output, retrieval_context)
        self.assertEqual(validated["steps"][0]["step_number"], 1)

    # TEST 22: Duplicate step_id -> duplicate rejected
    def test_22_duplicate_step_id_rejected(self):
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
            "steps": [
                {"step_id": "s1", "instruction": "Step 1", "step_number": 1, "source": {"chunk_ids": ["c1"]}},
                {"step_id": "s1", "instruction": "Duplicate Step 1", "step_number": 1, "source": {"chunk_ids": ["c1"]}}
            ]
        }
        validated = _validate_grounding(llm_output, retrieval_context)
        self.assertEqual(len(validated["steps"]), 1)

    # TEST 23: Generic mode LLM scoping and warning override
    def test_23_generic_mode_llm_scoping(self):
        sys_prompt = _build_system_prompt(model_known=False)
        self.assertIn("The requested model is NOT present in the indexed database", sys_prompt)
        self.assertIn("generic Samsung washing-machine evidence", sys_prompt)

    # TEST 24: Path traversal image request -> 403 / 404
    def test_24_image_path_traversal_prevention(self):
        # Direct path safety check
        self.assertFalse(_is_safe_path(GENERATED_IMAGE_FOLDER, "../../../etc/passwd"))
        self.assertFalse(_is_safe_path(GENERATED_IMAGE_FOLDER, "..\\..\\secret.txt"))

        # Flask route request
        resp = self.app.get("/generated_step_images_20260824_0052/../../../etc/passwd")
        self.assertIn(resp.status_code, [403, 404])

        resp2 = self.app.get("/extracted_images/..%2f..%2fpackage.json")
        self.assertIn(resp2.status_code, [403, 404])


if __name__ == "__main__":
    unittest.main()
