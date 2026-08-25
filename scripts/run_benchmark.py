"""
Guide Weave — Benchmark Runner (Stage 10)
=========================================
Executes Stage 10 Benchmarking & Evaluation against the production RAG pipeline.

Supports:
  --limit <N>   : Runs N benchmark queries (default 5 for development)
  --full        : Explicit flag to run the full benchmark suite
  --dry-run     : Validates dataset and shows execution plan without API/Qdrant calls

Strictly READ-ONLY. Never modifies Qdrant collections or ground truth files.
Restores environment variables on exit.
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env")

from scripts.generate_benchmark_dataset import (
    build_benchmark_dataset,
    validate_benchmark_dataset
)
from scripts.benchmark_evaluator import (
    evaluate_retrieval_metrics,
    evaluate_deterministic_answer_metrics,
    evaluate_image_metrics,
    call_groq_judge,
    calculate_end_to_end_score
)
from scripts.retrieve_text import retrieve_text
from backend.main import generate_guide_from_rag
from backend.pipeline_logger import pipeline_logger


def setup_benchmark_dir(run_id: str) -> Path:
    """Create unique run directory under benchmark_results/"""
    run_dir = PROJECT_ROOT / "benchmark_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(log_file: Path) -> logging.Logger:
    """Setup benchmark logger writing to benchmark.log and console."""
    logger = logging.getLogger("GuideWeaveBenchmark")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear previous

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def load_dataset(dataset_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load and validate benchmark dataset from disk, or build it if missing."""
    if not dataset_path.exists():
        logger.info(f"Dataset not found at {dataset_path}. Generating from ground truth...")
        dataset = build_benchmark_dataset()
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    is_valid, errors = validate_benchmark_dataset(dataset)
    if not is_valid:
        logger.error(f"Benchmark dataset validation failed with {len(errors)} errors:")
        for err in errors[:10]:
            logger.error(f"  - {err}")
        raise ValueError("Invalid benchmark dataset.")

    return dataset


def extract_evaluation_queries(
    dataset: Dict[str, Any],
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Flattens benchmark dataset cases into individual query execution items.
    """
    items = []
    cases = dataset.get("cases", [])
    for c in cases:
        bid = c.get("benchmark_id")
        pname = c.get("problem_name")
        for q_idx, q in enumerate(c.get("queries", [])):
            items.append({
                "benchmark_id": bid,
                "query_id": f"{bid}_q{q_idx + 1:02d}",
                "query": q,
                "case": c
            })

    if limit is not None and limit > 0:
        items = items[:limit]

    return items


def run_single_query(
    query_item: Dict[str, Any],
    configuration: str,  # 'rrf_only' or 'rrf_plus_jina'
    use_groq_judge: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Executes a single benchmark query through production retrieval and generation,
    then evaluates all metrics strictly.
    """
    query = query_item["query"]
    case = query_item["case"]
    bid = query_item["benchmark_id"]
    qid = query_item["query_id"]
    model_canonical = "WA5471ABP"
    database_model = "WA5471ABP/XAA"

    is_rerank = (configuration == "rrf_plus_jina")
    failures = []
    status = "success"

    expected_steps = case.get("expected_steps", [])
    expected_images = case.get("expected_images", [])
    expected_safety = case.get("safety_requirements", [])

    # Gather expected chunk IDs across steps
    expected_chunk_ids = set()
    for s in expected_steps:
        for cid in s.get("chunk_ids", []):
            if cid:
                expected_chunk_ids.add(cid)

    # 1. Measure text retrieval
    t0_retrieval = time.perf_counter()
    retrieval_obj = {}
    candidate_pool = []
    final_results = []
    reranker_latency_ms = 0.0

    try:
        # Re-use production retrieve_text
        retrieval_obj = retrieve_text(
            query=query,
            model=database_model,
            retrieval_mode="model_specific",
            final_top_k=8,
            candidate_top_k=30,
            rerank=is_rerank
        )
        final_results = retrieval_obj.get("results", [])
        retrieval_meta = retrieval_obj.get("retrieval", {})
        rerank_meta = retrieval_meta.get("reranking", {})
        reranker_latency_ms = rerank_meta.get("latency_ms", 0.0) or 0.0

        # For candidate pool diagnostic: if reranking was done, candidate pool is up to 30.
        # Otherwise final_results is the top retrieved.
        candidate_pool = final_results
    except Exception as e:
        failures.append(f"Text retrieval failed: {e}")
        status = "failed"
        if logger:
            logger.warning(f"[{qid}] [{configuration}] Text retrieval error: {e}")

    # 2. Measure end-to-end generation
    t0_gen = time.perf_counter()
    generated_guide = {}
    try:
        # Re-use production generate_guide_from_rag
        generated_guide = generate_guide_from_rag(
            query=query,
            model=model_canonical,
            mode="CLOUD"
        )
        if generated_guide.get("status") == "error":
            failures.append(f"LLM generation returned error: {generated_guide.get('message')}")
            status = "failed"
    except Exception as e:
        failures.append(f"LLM generation failed: {e}")
        status = "failed"
        if logger:
            logger.warning(f"[{qid}] [{configuration}] LLM generation error: {e}")

    total_latency_ms = (time.perf_counter() - t0_retrieval) * 1000.0

    # 3. Evaluate Metrics
    retrieval_metrics = evaluate_retrieval_metrics(
        candidate_pool=candidate_pool,
        final_results=final_results,
        expected_chunk_ids=expected_chunk_ids,
        expected_model=database_model
    )

    answer_metrics = evaluate_deterministic_answer_metrics(
        generated_guide=generated_guide,
        expected_steps=expected_steps,
        expected_safety_requirements=expected_safety
    )

    image_metrics = evaluate_image_metrics(
        generated_guide=generated_guide,
        expected_images=expected_images
    )

    # 4. Groq Judge for Faithfulness & Semantic matching
    faithfulness_score = "not_available"
    unsupported_claim_count = 0
    judge_status = "not_run"

    if use_groq_judge and status == "success" and generated_guide:
        j_status, j_result = call_groq_judge(
            query=query,
            ground_truth_case=case,
            retrieved_evidence_chunks=final_results,
            generated_guide=generated_guide
        )
        judge_status = j_status
        if j_status == "success":
            faith_dict = j_result.get("faithfulness", {})
            raw_fscore = faith_dict.get("score")
            if isinstance(raw_fscore, (int, float)):
                faithfulness_score = round(float(raw_fscore), 4)
            unsupported_claims = faith_dict.get("unsupported_claims", [])
            unsupported_claim_count = len(unsupported_claims) if isinstance(unsupported_claims, list) else 0

            # If deterministic step matching found 0 steps, check judge semantic matches
            if answer_metrics.get("step_recall") == 0.0 and expected_steps:
                semantic_matches = j_result.get("step_matches", [])
                matched_cnt = sum(1 for sm in semantic_matches if sm.get("match") is True)
                if matched_cnt > 0:
                    answer_metrics["step_recall"] = round(matched_cnt / len(expected_steps), 4)
        else:
            faithfulness_score = "not_available"
            if logger:
                logger.warning(f"[{qid}] [{configuration}] Groq judge call failed: {j_result.get('error')}")

    # 5. Composite End-to-End Score
    overall_score, score_components = calculate_end_to_end_score(
        retrieval_metrics=retrieval_metrics,
        answer_metrics=answer_metrics,
        faithfulness_score=faithfulness_score,
        image_metrics=image_metrics
    )

    # Production query ID for log tracing
    ctx = pipeline_logger.get_context()
    prod_query_id = getattr(ctx, "query_id", "N/A") if ctx else "N/A"

    record = {
        "benchmark_id": bid,
        "query_id": qid,
        "query": query,
        "model": model_canonical,
        "configuration": configuration,
        "production_query_id": prod_query_id,
        "retrieval": {
            "recall_at_30": retrieval_metrics.get("recall_at_30"),
            "recall_at_8": retrieval_metrics.get("recall_at_8"),
            "mrr": retrieval_metrics.get("mrr"),
            "ndcg_at_8": retrieval_metrics.get("ndcg_at_8"),
            "model_contamination": retrieval_metrics.get("model_contamination", False),
            "contaminated_chunks": retrieval_metrics.get("contaminated_chunks", [])
        },
        "answer": {
            "step_recall": answer_metrics.get("step_recall"),
            "step_order": answer_metrics.get("step_order"),
            "safety": answer_metrics.get("safety"),
            "critical_safety_violation": answer_metrics.get("critical_safety_violation", False),
            "faithfulness": faithfulness_score,
            "unsupported_claim_count": unsupported_claim_count,
            "judge_status": judge_status
        },
        "images": {
            "image_recall_at_3": image_metrics.get("image_recall_at_3"),
            "exact_image_count": image_metrics.get("exact_image_count", 0),
            "semantic_image_count": image_metrics.get("semantic_image_count", 0),
            "missing_image_file": image_metrics.get("missing_image_file", False)
        },
        "overall_score": overall_score,
        "overall_components": score_components,
        "latency": {
            "total_ms": round(total_latency_ms, 2),
            "reranker_ms": round(reranker_latency_ms, 2)
        },
        "status": status,
        "failures": failures
    }

    return record


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregate statistics, means, medians, and failure counts across run records.
    Correctly excludes 'not_applicable' and 'not_available' from denominators.
    """
    total_cases = len(records)
    successful = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") != "success"]

    def _mean_metric(path_fn):
        vals = []
        for r in successful:
            v = path_fn(r)
            if isinstance(v, (int, float)):
                vals.append(v)
        if not vals:
            return "not_applicable", 0
        return round(sum(vals) / len(vals), 4), len(vals)

    def _median_metric(path_fn):
        vals = []
        for r in successful:
            v = path_fn(r)
            if isinstance(v, (int, float)):
                vals.append(v)
        if not vals:
            return "not_applicable", 0
        vals.sort()
        mid = len(vals) // 2
        med = vals[mid] if len(vals) % 2 != 0 else (vals[mid - 1] + vals[mid]) / 2.0
        return round(med, 2), len(vals)

    # Retrieval
    mean_r30, den_r30 = _mean_metric(lambda r: r["retrieval"]["recall_at_30"])
    mean_r8, den_r8 = _mean_metric(lambda r: r["retrieval"]["recall_at_8"])
    mean_mrr, den_mrr = _mean_metric(lambda r: r["retrieval"]["mrr"])
    mean_ndcg8, den_ndcg8 = _mean_metric(lambda r: r["retrieval"]["ndcg_at_8"])

    # Answer
    mean_step_rec, den_step_rec = _mean_metric(lambda r: r["answer"]["step_recall"])
    mean_step_ord, den_step_ord = _mean_metric(lambda r: r["answer"]["step_order"])
    mean_safety, den_safety = _mean_metric(lambda r: r["answer"]["safety"])
    mean_faith, den_faith = _mean_metric(lambda r: r["answer"]["faithfulness"])
    mean_unsupported, _ = _mean_metric(lambda r: r["answer"]["unsupported_claim_count"])

    # Image
    mean_img_rec, den_img_rec = _mean_metric(lambda r: r["images"]["image_recall_at_3"])

    # Overall
    mean_overall, den_overall = _mean_metric(lambda r: r.get("overall_score"))

    # Latency
    mean_lat, _ = _mean_metric(lambda r: r["latency"]["total_ms"])
    med_lat, _ = _median_metric(lambda r: r["latency"]["total_ms"])
    mean_rerank_lat, _ = _mean_metric(lambda r: r["latency"]["reranker_ms"])

    # Failures & Flags
    contamination_count = sum(1 for r in records if r["retrieval"].get("model_contamination") is True)
    critical_safety_count = sum(1 for r in records if r["answer"].get("critical_safety_violation") is True)
    complete_retrieval_fails = sum(1 for r in records if r["retrieval"].get("recall_at_8") == 0.0)
    judge_failures = sum(1 for r in records if r["answer"].get("judge_status") == "failed")

    return {
        "total_executions": total_cases,
        "successful_executions": len(successful),
        "failed_executions": len(failed),
        "retrieval": {
            "recall_at_30": mean_r30,
            "recall_at_30_denominator": den_r30,
            "recall_at_8": mean_r8,
            "recall_at_8_denominator": den_r8,
            "mrr": mean_mrr,
            "mrr_denominator": den_mrr,
            "ndcg_at_8": mean_ndcg8,
            "ndcg_at_8_denominator": den_ndcg8
        },
        "answer": {
            "step_recall": mean_step_rec,
            "step_recall_denominator": den_step_rec,
            "step_order": mean_step_ord,
            "step_order_denominator": den_step_ord,
            "safety": mean_safety,
            "safety_denominator": den_safety,
            "faithfulness": mean_faith,
            "faithfulness_denominator": den_faith,
            "unsupported_claim_count_avg": mean_unsupported
        },
        "images": {
            "image_recall_at_3": mean_img_rec,
            "image_recall_at_3_denominator": den_img_rec
        },
        "overall_score": mean_overall,
        "overall_score_denominator": den_overall,
        "latency": {
            "average_total_ms": mean_lat,
            "median_total_ms": med_lat,
            "average_reranker_ms": mean_rerank_lat
        },
        "flags": {
            "model_contamination_count": contamination_count,
            "critical_safety_violation_count": critical_safety_count,
            "complete_retrieval_failure_count": complete_retrieval_fails,
            "judge_failure_count": judge_failures
        }
    }


def compute_comparison_table(
    rrf_summary: Dict[str, Any],
    jina_summary: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Computes absolute and relative differences for ablation report.
    Respects metric direction (e.g. latency lower is better).
    """
    metric_keys = [
        ("Recall@8", lambda s: s["retrieval"]["recall_at_8"], True, "%"),
        ("MRR", lambda s: s["retrieval"]["mrr"], True, "num"),
        ("nDCG@8", lambda s: s["retrieval"]["ndcg_at_8"], True, "num"),
        ("Step Recall", lambda s: s["answer"]["step_recall"], True, "%"),
        ("Step Order", lambda s: s["answer"]["step_order"], True, "%"),
        ("Safety", lambda s: s["answer"]["safety"], True, "%"),
        ("Faithfulness", lambda s: s["answer"]["faithfulness"], True, "%"),
        ("Image Recall@3", lambda s: s["images"]["image_recall_at_3"], True, "%"),
        ("Average Latency", lambda s: s["latency"]["average_total_ms"], False, "ms"),
        ("Recall@30 [diagnostic]", lambda s: s["retrieval"]["recall_at_30"], True, "%")
    ]

    rows = []
    for label, getter, higher_is_better, mtype in metric_keys:
        v_rrf = getter(rrf_summary)
        v_jina = getter(jina_summary)

        if isinstance(v_rrf, (int, float)) and isinstance(v_jina, (int, float)):
            abs_change = round(v_jina - v_rrf, 4)
            if v_rrf > 0:
                rel_pct = round(((v_jina - v_rrf) / v_rrf) * 100.0, 2)
                rel_str = f"{rel_pct:+.2f}%"
            else:
                rel_str = "N/A"

            if mtype == "%":
                rrf_str = f"{v_rrf * 100.0:.2f}%"
                jina_str = f"{v_jina * 100.0:.2f}%"
                abs_str = f"{abs_change * 100.0:+.2f}%"
            elif mtype == "ms":
                rrf_str = f"{v_rrf:.1f} ms"
                jina_str = f"{v_jina:.1f} ms"
                abs_str = f"{abs_change:+.1f} ms"
            else:
                rrf_str = f"{v_rrf:.4f}"
                jina_str = f"{v_jina:.4f}"
                abs_str = f"{abs_change:+.4f}"
        else:
            rrf_str = str(v_rrf)
            jina_str = str(v_jina)
            abs_str = "N/A"
            rel_str = "N/A"

        rows.append({
            "metric": label,
            "rrf_only": rrf_str,
            "rrf_plus_jina": jina_str,
            "absolute_change": abs_str,
            "relative_change": rel_str
        })

    return rows


def extract_worst_cases(records: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top N worst cases for retrieval, answer, and image quality."""
    def _safe_float(v):
        return float(v) if isinstance(v, (int, float)) else 1.0

    # 1. Worst retrieval (lowest Recall@8, lowest nDCG@8)
    worst_retrieval = sorted(
        records,
        key=lambda r: (_safe_float(r["retrieval"].get("recall_at_8")), _safe_float(r["retrieval"].get("ndcg_at_8")))
    )[:top_n]

    # 2. Worst answer (lowest Step Recall, lowest Faithfulness)
    worst_answer = sorted(
        records,
        key=lambda r: (_safe_float(r["answer"].get("step_recall")), _safe_float(r["answer"].get("faithfulness")))
    )[:top_n]

    # 3. Worst image (lowest Image Recall@3)
    worst_image = sorted(
        records,
        key=lambda r: _safe_float(r["images"].get("image_recall_at_3"))
    )[:top_n]

    return {
        "retrieval": worst_retrieval,
        "answer": worst_answer,
        "images": worst_image
    }


def generate_markdown_report(
    run_id: str,
    target_model: str,
    total_cases: int,
    jina_summary: Dict[str, Any],
    rrf_summary: Dict[str, Any],
    comparison_rows: List[Dict[str, Any]],
    worst_cases: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Formats the final REPORT.md strictly according to Section 48 & 49.
    """
    def _pct(v):
        return f"{v * 100.0:.2f}%" if isinstance(v, (int, float)) else str(v)

    def _num(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    r = jina_summary["retrieval"]
    a = jina_summary["answer"]
    img = jina_summary["images"]
    ov = jina_summary.get("overall_score")

    report_lines = [
        "==================================================",
        "GUIDE WEAVE — STAGE 10 EVALUATION",
        "==================================================",
        "",
        f"Benchmark Run ID: {run_id}",
        f"Benchmark Cases: {total_cases}",
        f"Model: {target_model}",
        "Appliance: Samsung Washing Machine",
        "",
        "RETRIEVAL",
        "----------------------------------",
        f"Recall@30       {_pct(r.get('recall_at_30')):<10} [diagnostic]",
        f"Recall@8        {_pct(r.get('recall_at_8'))}",
        f"MRR             {_num(r.get('mrr'))}",
        f"nDCG@8          {_num(r.get('ndcg_at_8'))}",
        "",
        "ANSWER QUALITY",
        "----------------------------------",
        f"Step Recall     {_pct(a.get('step_recall'))}",
        f"Step Order      {_pct(a.get('step_order'))}",
        f"Safety          {_pct(a.get('safety'))}",
        f"Faithfulness    {_pct(a.get('faithfulness'))}",
        "",
        "IMAGE RETRIEVAL",
        "----------------------------------",
        f"Image Recall@3  {_pct(img.get('image_recall_at_3'))}",
        "",
        "END-TO-END",
        "----------------------------------",
        f"Overall Score   {_pct(ov)}",
        "",
        "RERANKER ABLATION",
        "----------------------------------",
        f"{'':<20} {'RRF':<12} {'RRF + Jina':<12} {'Change':<12}"
    ]

    for row in comparison_rows:
        m = row["metric"]
        r_val = row["rrf_only"]
        j_val = row["rrf_plus_jina"]
        chg = row["absolute_change"]
        report_lines.append(f"{m:<20} {r_val:<12} {j_val:<12} {chg:<12}")

    report_lines.append("")
    report_lines.append("==================================================")
    report_lines.append("FAILURE FLAGS & SYSTEM DIAGNOSTICS")
    report_lines.append("----------------------------------")
    f_flags = jina_summary.get("flags", {})
    report_lines.append(f"Model Contamination Count:        {f_flags.get('model_contamination_count', 0)}")
    report_lines.append(f"Critical Safety Violation Count:  {f_flags.get('critical_safety_violation_count', 0)}")
    report_lines.append(f"Complete Retrieval Failure Count: {f_flags.get('complete_retrieval_failure_count', 0)}")
    report_lines.append(f"Judge Failure Count:              {f_flags.get('judge_failure_count', 0)}")
    report_lines.append("")

    # Worst cases
    report_lines.append("==================================================")
    report_lines.append("WORST PERFORMING CASES (DIAGNOSTIC)")
    report_lines.append("==================================================")

    report_lines.append("\nTop 5 Worst Retrieval Cases:")
    for idx, item in enumerate(worst_cases.get("retrieval", []), start=1):
        report_lines.append(f"  {idx}. [{item.get('benchmark_id')}] Query: \"{item.get('query')}\"")
        report_lines.append(f"     Recall@8: {_pct(item['retrieval'].get('recall_at_8'))}, nDCG@8: {_num(item['retrieval'].get('ndcg_at_8'))}")

    report_lines.append("\nTop 5 Worst Answer Quality Cases:")
    for idx, item in enumerate(worst_cases.get("answer", []), start=1):
        report_lines.append(f"  {idx}. [{item.get('benchmark_id')}] Query: \"{item.get('query')}\"")
        report_lines.append(f"     Step Recall: {_pct(item['answer'].get('step_recall'))}, Faithfulness: {_pct(item['answer'].get('faithfulness'))}")

    report_lines.append("\nTop 5 Worst Image Retrieval Cases:")
    for idx, item in enumerate(worst_cases.get("images", []), start=1):
        report_lines.append(f"  {idx}. [{item.get('benchmark_id')}] Query: \"{item.get('query')}\"")
        report_lines.append(f"     Image Recall@3: {_pct(item['images'].get('image_recall_at_3'))}")

    report_lines.append("\n==================================================")
    return "\n".join(report_lines)


def run_benchmark_pipeline(
    limit: Optional[int] = 5,
    full: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Main benchmark pipeline execution orchestrator.
    Handles environment preservation, execution of both ablation runs,
    evaluation, metric aggregation, artifact writing, and cleanup.
    """
    # Preserve developer environment
    original_rerank_env = os.environ.get("RERANK_ENABLED")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = setup_benchmark_dir(run_id)
    logger = setup_logger(run_dir / "benchmark.log")

    try:
        logger.info("==================================================")
        logger.info("GUIDE WEAVE — STAGE 10 BENCHMARK SUITE")
        logger.info("==================================================")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Run Directory: {run_dir}")

        dataset_path = PROJECT_ROOT / "benchmark_data" / "benchmark_dataset.json"
        dataset = load_dataset(dataset_path, logger)

        target_model = dataset.get("model", "WA5471ABP")
        database_model = dataset.get("database_model", "WA5471ABP/XAA")
        total_available_cases = len(dataset.get("cases", []))
        total_available_queries = sum(len(c.get("queries", [])) for c in dataset.get("cases", []))

        query_limit = None if full else (limit or 5)
        query_items = extract_evaluation_queries(dataset, limit=query_limit)
        num_queries_to_run = len(query_items)
        total_pipeline_executions = num_queries_to_run * 2  # RRF-only + RRF+Jina

        logger.info(f"Target Model: {target_model} ({database_model})")
        logger.info(f"Total Dataset Problems: {total_available_cases}")
        logger.info(f"Total Dataset Queries: {total_available_queries}")
        logger.info(f"Execution Mode: {'FULL' if full else f'LIMITED (limit={query_limit})'}")
        logger.info(f"Selected Queries to Evaluate: {num_queries_to_run}")
        logger.info(f"Total Pipeline Executions: {total_pipeline_executions} (2 configurations × {num_queries_to_run} queries)")

        if dry_run:
            logger.info("\n--- DRY RUN COMPLETED ---")
            logger.info("Dataset validated successfully.")
            logger.info("No Qdrant, Jina, or Groq calls made.")
            return {"status": "dry_run_success", "run_id": run_id, "cases": total_available_cases}

        # --- EXECUTION: RUN A (RRF ONLY) ---
        logger.info("\n--------------------------------------------------")
        logger.info("STARTING RUN A: RRF ONLY (RERANK_ENABLED=false)")
        logger.info("--------------------------------------------------")
        os.environ["RERANK_ENABLED"] = "false"
        rrf_records = []
        for idx, item in enumerate(query_items, start=1):
            logger.info(f"[{idx}/{num_queries_to_run}] Running RRF query: {item['query_id']} - \"{item['query'][:60]}...\"")
            rec = run_single_query(item, configuration="rrf_only", use_groq_judge=True, logger=logger)
            rrf_records.append(rec)

        # --- EXECUTION: RUN B (RRF + JINA) ---
        logger.info("\n--------------------------------------------------")
        logger.info("STARTING RUN B: RRF + JINA (RERANK_ENABLED=true)")
        logger.info("--------------------------------------------------")
        os.environ["RERANK_ENABLED"] = "true"
        jina_records = []
        for idx, item in enumerate(query_items, start=1):
            logger.info(f"[{idx}/{num_queries_to_run}] Running RRF+Jina query: {item['query_id']} - \"{item['query'][:60]}...\"")
            rec = run_single_query(item, configuration="rrf_plus_jina", use_groq_judge=True, logger=logger)
            jina_records.append(rec)

        # --- AGGREGATION & REPORTING ---
        logger.info("\n--------------------------------------------------")
        logger.info("EVALUATING AND AGGREGATING RESULTS")
        logger.info("--------------------------------------------------")

        rrf_summary = aggregate_metrics(rrf_records)
        jina_summary = aggregate_metrics(jina_records)
        comparison_rows = compute_comparison_table(rrf_summary, jina_summary)
        worst_cases = extract_worst_cases(jina_records, top_n=5)

        # 1. Write config.json
        config_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model": target_model,
            "database_model": database_model,
            "total_dataset_cases": total_available_cases,
            "total_dataset_queries": total_available_queries,
            "evaluated_queries_count": num_queries_to_run,
            "full_run": full,
            "limit": query_limit
        }
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        # 2. Write detailed_results.json
        detailed_data = {
            "run_id": run_id,
            "rrf_only_results": rrf_records,
            "rrf_plus_jina_results": jina_records
        }
        with open(run_dir / "detailed_results.json", "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, indent=4, ensure_ascii=False)

        # 3. Write summary.json
        summary_data = {
            "run_id": run_id,
            "model": target_model,
            "evaluated_queries": num_queries_to_run,
            "rrf_only": rrf_summary,
            "rrf_plus_jina": jina_summary,
            "comparison": comparison_rows
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)

        # 4. Write summary.csv & reranker_comparison.csv
        import csv
        with open(run_dir / "reranker_comparison.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "rrf_only", "rrf_plus_jina", "absolute_change", "relative_change"])
            writer.writeheader()
            for r in comparison_rows:
                writer.writerow(r)

        with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "rrf_only", "rrf_plus_jina", "absolute_change", "relative_change"])
            writer.writeheader()
            for r in comparison_rows:
                writer.writerow(r)

        # 5. Write REPORT.md
        report_md = generate_markdown_report(
            run_id=run_id,
            target_model=target_model,
            total_cases=num_queries_to_run,
            jina_summary=jina_summary,
            rrf_summary=rrf_summary,
            comparison_rows=comparison_rows,
            worst_cases=worst_cases
        )
        with open(run_dir / "REPORT.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info(f"\n{report_md}\n")
        logger.info(f"Artifacts successfully saved to {run_dir}")
        return summary_data

    finally:
        # Restore environment
        if original_rerank_env is not None:
            os.environ["RERANK_ENABLED"] = original_rerank_env
        elif "RERANK_ENABLED" in os.environ:
            del os.environ["RERANK_ENABLED"]


def main():
    parser = argparse.ArgumentParser(description="Guide Weave Stage 10 Benchmark Runner")
    parser.add_argument("--limit", type=int, default=None, help="Number of benchmark queries to run (default: 5)")
    parser.add_argument("--full", action="store_true", help="Run full benchmark across all ~450 queries")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and print execution plan without API/Qdrant calls")

    args = parser.parse_args()

    # Cost guard default
    if not args.full and args.limit is None and not args.dry_run:
        print("[COST GUARD] No --full or --limit flag specified. Defaulting to development run: --limit 5")
        limit_val = 5
    else:
        limit_val = args.limit

    run_benchmark_pipeline(
        limit=limit_val,
        full=args.full,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
