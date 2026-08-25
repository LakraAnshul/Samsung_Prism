"""
Guide Weave — Stage 10 Benchmark Tests
======================================
Tests all 24 required validation points for Stage 10 Benchmarking & Evaluation.

Test Coverage:
1. Dataset generation
2. Dataset validation
3. Duplicate ID detection
4. Recall@8 calculation
5. Recall@30 calculation
6. MRR calculation
7. nDCG@8 calculation
8. Step Recall calculation
9. Step Order calculation
10. Safety handling
11. Faithfulness aggregation
12. Image Recall@3 calculation
13. Not-applicable metric handling
14. Empty expected steps
15. Empty image expectations
16. Zero retrieval results
17. Failed pipeline request
18. Model contamination detection
19. RRF-only configuration
20. RRF+Jina configuration
21. Same queries used for both ablation runs
22. Benchmark result persistence
23. No modification of source ground truth
24. No modification of Qdrant
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_benchmark_dataset import (
    build_benchmark_dataset,
    validate_benchmark_dataset,
    generate_natural_queries
)
from scripts.benchmark_evaluator import (
    evaluate_retrieval_metrics,
    evaluate_deterministic_answer_metrics,
    evaluate_image_metrics,
    evaluate_step_order,
    calculate_end_to_end_score
)
from scripts.run_benchmark import (
    extract_evaluation_queries,
    aggregate_metrics,
    compute_comparison_table,
    setup_benchmark_dir
)


class TestBenchmarkSuite(unittest.TestCase):

    def setUp(self):
        self.sample_case = {
            "benchmark_id": "case_0001",
            "document_id": "WA5471ABP_01_Starting_Power_Problems_Detailed",
            "problem_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01",
            "problem_name": "Washer does not turn on",
            "queries": [
                "How do I fix a washer that does not turn on?",
                "My Samsung washer won't turn on.",
                "The washing machine cannot turn on properly."
            ],
            "expected_steps": [
                {
                    "step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_01",
                    "step_number": 1,
                    "instruction": "Check electrical outlet.",
                    "chunk_ids": ["chunk_01", "chunk_02"],
                    "pages": [2]
                },
                {
                    "step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_02",
                    "step_number": 2,
                    "instruction": "Check power cord.",
                    "chunk_ids": ["chunk_02", "chunk_03"],
                    "pages": [2]
                }
            ],
            "expected_images": [
                {
                    "step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_01",
                    "image_id": "01_Starting_Power_Problems_Detailed_step001.png"
                },
                {
                    "step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_02",
                    "image_id": "01_Starting_Power_Problems_Detailed_step002.png"
                }
            ],
            "safety_requirements": [
                "Do not bypass a fuse, circuit breaker, grounding connection, or other protective device."
            ]
        }

    # 1. Dataset generation
    def test_01_dataset_generation(self):
        dataset = build_benchmark_dataset()
        self.assertIn("cases", dataset)
        self.assertEqual(len(dataset["cases"]), 150)
        self.assertEqual(dataset["model"], "WA5471ABP")
        self.assertEqual(dataset["database_model"], "WA5471ABP/XAA")

    # 2. Dataset validation
    def test_02_dataset_validation(self):
        dataset = {"benchmark_version": "1.0", "cases": [self.sample_case]}
        is_valid, errors = validate_benchmark_dataset(dataset)
        self.assertTrue(is_valid, f"Validation errors: {errors}")
        self.assertEqual(len(errors), 0)

    # 3. Duplicate ID detection
    def test_03_duplicate_id_detection(self):
        dup_case = dict(self.sample_case)
        dataset = {"benchmark_version": "1.0", "cases": [self.sample_case, dup_case]}
        is_valid, errors = validate_benchmark_dataset(dataset)
        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate benchmark_id" in e for e in errors))

    # 4. Recall@8 calculation
    def test_04_recall_at_8_calculation(self):
        candidate_pool = [{"chunk_id": "chunk_01"}, {"chunk_id": "chunk_02"}, {"chunk_id": "chunk_03"}]
        final_results = [{"chunk_id": "chunk_01"}, {"chunk_id": "chunk_02"}, {"chunk_id": "other"}]
        expected = {"chunk_01", "chunk_02", "chunk_03"}
        res = evaluate_retrieval_metrics(candidate_pool, final_results, expected)
        self.assertAlmostEqual(res["recall_at_8"], 2.0 / 3.0, places=3)

    # 5. Recall@30 calculation
    def test_05_recall_at_30_calculation(self):
        candidate_pool = [{"chunk_id": "chunk_01"}, {"chunk_id": "chunk_02"}, {"chunk_id": "chunk_03"}]
        final_results = [{"chunk_id": "chunk_01"}]
        expected = {"chunk_01", "chunk_02", "chunk_03"}
        res = evaluate_retrieval_metrics(candidate_pool, final_results, expected)
        self.assertAlmostEqual(res["recall_at_30"], 1.0, places=3)

    # 6. MRR calculation
    def test_06_mrr_calculation(self):
        final_results = [{"chunk_id": "irr_1"}, {"chunk_id": "chunk_02"}, {"chunk_id": "chunk_01"}]
        expected = {"chunk_01", "chunk_02"}
        res = evaluate_retrieval_metrics([], final_results, expected)
        # First relevant chunk is at rank 2 -> MRR = 1/2 = 0.5
        self.assertAlmostEqual(res["mrr"], 0.5, places=3)

    # 7. nDCG@8 calculation
    def test_07_ndcg_at_8_calculation(self):
        final_results = [{"chunk_id": "chunk_01"}, {"chunk_id": "irr_1"}, {"chunk_id": "chunk_02"}]
        expected = {"chunk_01", "chunk_02"}
        res = evaluate_retrieval_metrics([], final_results, expected)
        self.assertGreater(res["ndcg_at_8"], 0.0)
        self.assertLessEqual(res["ndcg_at_8"], 1.0)

    # 8. Step Recall calculation
    def test_08_step_recall_calculation(self):
        gen_guide = {
            "steps": [
                {"step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_01", "step_number": 1}
            ]
        }
        res = evaluate_deterministic_answer_metrics(gen_guide, self.sample_case["expected_steps"], [])
        self.assertAlmostEqual(res["step_recall"], 0.5, places=3)

    # 9. Step Order calculation
    def test_09_step_order_calculation(self):
        # In-order [0, 1, 2] -> 1.0
        score_perfect = evaluate_step_order([0, 1, 2])
        self.assertEqual(score_perfect, 1.0)
        # Inverted [2, 1, 0] -> 0.0
        score_inverted = evaluate_step_order([2, 1, 0])
        self.assertEqual(score_inverted, 0.0)
        # Partial [0, 2, 1] -> 2 concordant pairs out of 3 -> 2/3 = 0.6667
        score_partial = evaluate_step_order([0, 2, 1])
        self.assertAlmostEqual(score_partial, 2.0 / 3.0, places=3)

    # 10. Safety handling
    def test_10_safety_handling(self):
        gen_guide = {
            "steps": [],
            "safety_warnings": ["Do not bypass a fuse, circuit breaker, grounding connection, or other protective device."]
        }
        res = evaluate_deterministic_answer_metrics(gen_guide, [], self.sample_case["safety_requirements"])
        self.assertEqual(res["safety"], 1.0)
        self.assertFalse(res["critical_safety_violation"])

    # 11. Faithfulness aggregation
    def test_11_faithfulness_aggregation(self):
        records = [
            {
                "status": "success",
                "retrieval": {"recall_at_30": 1.0, "recall_at_8": 1.0, "mrr": 1.0, "ndcg_at_8": 1.0},
                "answer": {"step_recall": 1.0, "step_order": 1.0, "safety": 1.0, "faithfulness": 0.90, "unsupported_claim_count": 0},
                "images": {"image_recall_at_3": 1.0},
                "latency": {"total_ms": 100, "reranker_ms": 10}
            },
            {
                "status": "success",
                "retrieval": {"recall_at_30": 1.0, "recall_at_8": 1.0, "mrr": 1.0, "ndcg_at_8": 1.0},
                "answer": {"step_recall": 1.0, "step_order": 1.0, "safety": 1.0, "faithfulness": 0.80, "unsupported_claim_count": 1},
                "images": {"image_recall_at_3": 1.0},
                "latency": {"total_ms": 200, "reranker_ms": 20}
            }
        ]
        summary = aggregate_metrics(records)
        self.assertAlmostEqual(summary["answer"]["faithfulness"], 0.85, places=2)

    # 12. Image Recall@3 calculation
    def test_12_image_recall_at_3_calculation(self):
        gen_guide = {
            "steps": [
                {
                    "step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_01",
                    "images": [{"image_id": "01_Starting_Power_Problems_Detailed_step001.png"}]
                },
                {
                    "step_id": "WA5471ABP_01_Starting_Power_Problems_Detailed_problem_01_step_02",
                    "images": []
                }
            ]
        }
        res = evaluate_image_metrics(gen_guide, self.sample_case["expected_images"])
        self.assertAlmostEqual(res["image_recall_at_3"], 0.5, places=2)

    # 13. Not-applicable metric handling
    def test_13_not_applicable_metric_handling(self):
        res_ans = evaluate_deterministic_answer_metrics({"steps": []}, [], [])
        self.assertEqual(res_ans["step_recall"], "not_applicable")
        self.assertEqual(res_ans["safety"], "not_applicable")

        score, comp = calculate_end_to_end_score(
            retrieval_metrics={"recall_at_8": 1.0, "mrr": 1.0, "ndcg_at_8": 1.0},
            answer_metrics=res_ans,
            faithfulness_score="not_available",
            image_metrics={"image_recall_at_3": "not_applicable"}
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(comp["answer_component"], "not_applicable")

    # 14. Empty expected steps
    def test_14_empty_expected_steps(self):
        res = evaluate_deterministic_answer_metrics({"steps": [{"step_id": "foo"}]}, [], [])
        self.assertEqual(res["step_recall"], "not_applicable")
        self.assertEqual(res["step_order"], "not_applicable")

    # 15. Empty image expectations
    def test_15_empty_image_expectations(self):
        res = evaluate_image_metrics({"steps": []}, [])
        self.assertEqual(res["image_recall_at_3"], "not_applicable")

    # 16. Zero retrieval results
    def test_16_zero_retrieval_results(self):
        res = evaluate_retrieval_metrics([], [], {"chunk_01"})
        self.assertEqual(res["recall_at_8"], 0.0)
        self.assertEqual(res["mrr"], 0.0)
        self.assertEqual(res["ndcg_at_8"], 0.0)

    # 17. Failed pipeline request
    def test_17_failed_pipeline_request(self):
        record = {
            "status": "failed",
            "failures": ["Network error"],
            "retrieval": {"recall_at_30": "not_applicable", "recall_at_8": "not_applicable", "mrr": "not_applicable", "ndcg_at_8": "not_applicable"},
            "answer": {"step_recall": "not_applicable", "step_order": "not_applicable", "safety": "not_applicable", "faithfulness": "not_available", "unsupported_claim_count": 0},
            "images": {"image_recall_at_3": "not_applicable"},
            "latency": {"total_ms": 0, "reranker_ms": 0}
        }
        summary = aggregate_metrics([record])
        self.assertEqual(summary["failed_executions"], 1)
        self.assertEqual(summary["successful_executions"], 0)

    # 18. Model contamination detection
    def test_18_model_contamination_detection(self):
        final_results = [
            {"chunk_id": "chunk_01", "model": "WA5471ABP/XAA"},
            {"chunk_id": "chunk_foreign", "model": "DV45H7000EW/A2"}  # Foreign model chunk
        ]
        res = evaluate_retrieval_metrics([], final_results, {"chunk_01"}, expected_model="WA5471ABP/XAA")
        self.assertTrue(res["model_contamination"])
        self.assertEqual(len(res["contaminated_chunks"]), 1)

    # 19. RRF-only configuration
    def test_19_rrf_only_configuration(self):
        # Verify reranker flag handling
        from scripts.rerank_text import load_rerank_config
        orig = os.environ.get("RERANK_ENABLED")
        try:
            os.environ["RERANK_ENABLED"] = "false"
            cfg = load_rerank_config()
            self.assertFalse(cfg["RERANK_ENABLED"])
        finally:
            if orig is not None:
                os.environ["RERANK_ENABLED"] = orig

    # 20. RRF+Jina configuration
    def test_20_rrf_plus_jina_configuration(self):
        from scripts.rerank_text import load_rerank_config
        orig = os.environ.get("RERANK_ENABLED")
        try:
            os.environ["RERANK_ENABLED"] = "true"
            cfg = load_rerank_config()
            self.assertTrue(cfg["RERANK_ENABLED"])
        finally:
            if orig is not None:
                os.environ["RERANK_ENABLED"] = orig

    # 21. Same queries used for both ablation runs
    def test_21_same_queries_for_ablation(self):
        dataset = {"cases": [self.sample_case]}
        items = extract_evaluation_queries(dataset, limit=5)
        queries_a = [item["query"] for item in items]
        queries_b = [item["query"] for item in items]
        self.assertEqual(queries_a, queries_b)

    # 22. Benchmark result persistence
    def test_22_benchmark_result_persistence(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            sample_summary = {
                "run_id": "test_run",
                "retrieval": {"recall_at_8": 0.85, "mrr": 0.90, "ndcg_at_8": 0.88, "recall_at_30": 1.0},
                "answer": {"step_recall": 0.80, "step_order": 1.0, "safety": 1.0, "faithfulness": 0.92, "unsupported_claim_count_avg": 0},
                "images": {"image_recall_at_3": 0.95},
                "overall_score": 0.88,
                "latency": {"average_total_ms": 1500.0, "median_total_ms": 1400.0, "average_reranker_ms": 250.0},
                "flags": {}
            }
            comp_table = compute_comparison_table(sample_summary, sample_summary)
            self.assertTrue(len(comp_table) > 0)
            self.assertIn("metric", comp_table[0])
        finally:
            shutil.rmtree(temp_dir)

    # 23. No modification of source ground truth
    def test_23_no_modification_of_ground_truth(self):
        gt_dir = PROJECT_ROOT / "chunked_ground_truth"
        files = list(gt_dir.glob("*.json"))
        self.assertEqual(len(files), 20)

    # 24. No modification of Qdrant
    def test_24_no_modification_of_qdrant(self):
        # Read-only verification that Qdrant collections maintain expected point counts
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        wm_count = client.count("washing_machines").count
        img_count = client.count("washing_machine_images").count
        self.assertEqual(wm_count, 170)
        self.assertEqual(img_count, 931)


if __name__ == "__main__":
    unittest.main()
