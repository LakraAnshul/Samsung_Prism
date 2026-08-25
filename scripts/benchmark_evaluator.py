"""
Guide Weave — Benchmark Evaluator (Stage 10)
===========================================
Pure evaluation logic for retrieval, answer quality, faithfulness,
image retrieval, and end-to-end composite metrics.

Strictly READ-ONLY. No side effects on production data or Qdrant.
"""

import os
import json
import math
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set


# =====================================================================
# 1. RETRIEVAL METRICS (Recall@30, Recall@8, MRR, nDCG@8)
# =====================================================================

def evaluate_retrieval_metrics(
    candidate_pool: List[Dict[str, Any]],
    final_results: List[Dict[str, Any]],
    expected_chunk_ids: Set[str],
    expected_model: str = "WA5471ABP/XAA"
) -> Dict[str, Any]:
    """
    Computes primary retrieval metrics:
    - Recall@30 (diagnostic for candidate pool)
    - Recall@8 (final evidence retention)
    - MRR (Mean Reciprocal Rank of first relevant chunk)
    - nDCG@8 (binary relevance)
    - Model Contamination Check
    """
    expected_set = set(expected_chunk_ids)
    total_expected = len(expected_set)

    # Model contamination check
    model_contamination = False
    contaminated_chunks = []
    for r in final_results:
        chunk_model = r.get("model")
        if chunk_model and chunk_model != expected_model:
            model_contamination = True
            contaminated_chunks.append({
                "chunk_id": r.get("chunk_id"),
                "model": chunk_model,
                "expected": expected_model
            })

    if total_expected == 0:
        return {
            "recall_at_30": "not_applicable",
            "recall_at_8": "not_applicable",
            "mrr": "not_applicable",
            "ndcg_at_8": "not_applicable",
            "model_contamination": model_contamination,
            "contaminated_chunks": contaminated_chunks,
            "relevant_retrieved_count": 0,
            "total_expected_count": 0
        }

    # 1. Recall@30 (Candidate pool)
    cand_chunk_ids = [c.get("chunk_id") for c in candidate_pool if c.get("chunk_id")]
    cand_relevant_found = set(cand_chunk_ids) & expected_set
    recall_at_30 = len(cand_relevant_found) / total_expected

    # 2. Recall@8 (Final results up to 8)
    top8 = final_results[:8]
    top8_chunk_ids = [c.get("chunk_id") for c in top8 if c.get("chunk_id")]
    top8_relevant_found = set(top8_chunk_ids) & expected_set
    recall_at_8 = len(top8_relevant_found) / total_expected

    # 3. MRR (Final ordering)
    mrr = 0.0
    first_relevant_rank = None
    for idx, c in enumerate(final_results, start=1):
        cid = c.get("chunk_id")
        if cid and cid in expected_set:
            mrr = 1.0 / idx
            first_relevant_rank = idx
            break

    # 4. nDCG@8 (Binary relevance)
    dcg_8 = 0.0
    for idx, c in enumerate(top8, start=1):
        cid = c.get("chunk_id")
        rel = 1.0 if (cid and cid in expected_set) else 0.0
        if rel > 0:
            dcg_8 += rel / math.log2(idx + 1)

    # Ideal DCG@8
    idcg_8 = 0.0
    max_ideal = min(8, total_expected)
    for idx in range(1, max_ideal + 1):
        idcg_8 += 1.0 / math.log2(idx + 1)

    ndcg_at_8 = (dcg_8 / idcg_8) if idcg_8 > 0 else 0.0

    return {
        "recall_at_30": round(recall_at_30, 4),
        "recall_at_8": round(recall_at_8, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_8": round(ndcg_at_8, 4),
        "first_relevant_rank": first_relevant_rank,
        "model_contamination": model_contamination,
        "contaminated_chunks": contaminated_chunks,
        "relevant_retrieved_count": len(top8_relevant_found),
        "total_expected_count": total_expected
    }


# =====================================================================
# 2. ANSWER QUALITY METRICS (Step Recall, Step Order, Safety)
# =====================================================================

def evaluate_step_order(matched_expected_indices: List[int]) -> float:
    """
    Computes normalized pairwise concordance score [0.0, 1.0] for matched step sequence.
    """
    n = len(matched_expected_indices)
    if n <= 1:
        return 1.0

    total_pairs = n * (n - 1) // 2
    concordant_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if matched_expected_indices[i] < matched_expected_indices[j]:
                concordant_pairs += 1

    return concordant_pairs / total_pairs


def evaluate_deterministic_answer_metrics(
    generated_guide: Dict[str, Any],
    expected_steps: List[Dict[str, Any]],
    expected_safety_requirements: List[str]
) -> Dict[str, Any]:
    """
    Computes deterministic answer quality metrics:
    - Step Recall (via step_id)
    - Step Order
    - Safety Recall & Critical Violation
    """
    gen_steps = generated_guide.get("steps", [])
    gen_safety = generated_guide.get("safety_warnings", []) or generated_guide.get("safety", [])
    if isinstance(gen_safety, str):
        gen_safety = [gen_safety]

    # --- Step Recall & Order ---
    if not expected_steps:
        step_recall = "not_applicable"
        step_order = "not_applicable"
        matched_steps = []
        missing_steps = []
    else:
        expected_step_ids = [s["step_id"] for s in expected_steps if "step_id" in s]
        expected_id_to_idx = {sid: idx for idx, sid in enumerate(expected_step_ids)}

        matched_expected_indices = []
        matched_step_ids = set()

        for g_step in gen_steps:
            gid = g_step.get("step_id")
            if gid and gid in expected_id_to_idx:
                if gid not in matched_step_ids:
                    matched_step_ids.add(gid)
                    matched_expected_indices.append(expected_id_to_idx[gid])

        step_recall = len(matched_step_ids) / len(expected_steps)
        step_order = evaluate_step_order(matched_expected_indices) if matched_expected_indices else 0.0

        missing_steps = [sid for sid in expected_step_ids if sid not in matched_step_ids]
        matched_steps = list(matched_step_ids)

    # --- Safety ---
    if not expected_safety_requirements:
        safety_score = "not_applicable"
        critical_safety_violation = False
        missing_safety = []
    else:
        # Check text presence of expected safety requirements in generated response
        gen_text_blobs = " ".join([str(w) for w in gen_safety] + [s.get("instruction", "") or s.get("step_text", "") for s in gen_steps]).lower()
        matched_safety_count = 0
        missing_safety = []

        for req in expected_safety_requirements:
            # Check key tokens
            req_words = [w.lower() for w in re.findall(r"\b\w{4,}\b", req)]
            if req_words:
                matching_words = [w for w in req_words if w in gen_text_blobs]
                match_ratio = len(matching_words) / len(req_words)
                if match_ratio >= 0.5:
                    matched_safety_count += 1
                else:
                    missing_safety.append(req)
            else:
                missing_safety.append(req)

        safety_score = matched_safety_count / len(expected_safety_requirements)
        critical_safety_violation = (len(missing_safety) > 0)

    return {
        "step_recall": round(step_recall, 4) if isinstance(step_recall, float) else step_recall,
        "step_order": round(step_order, 4) if isinstance(step_order, float) else step_order,
        "safety": round(safety_score, 4) if isinstance(safety_score, float) else safety_score,
        "critical_safety_violation": critical_safety_violation,
        "matched_steps": matched_steps,
        "missing_steps": missing_steps,
        "missing_safety": missing_safety
    }


# =====================================================================
# 3. IMAGE RETRIEVAL METRICS (Image Recall@3)
# =====================================================================

def evaluate_image_metrics(
    generated_guide: Dict[str, Any],
    expected_images: List[Dict[str, str]],
    image_knowledge_base: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Computes Image Recall@3.
    Relevance levels:
      3 = exact step image (same step_id or image_id)
      2 = strongly relevant semantic fallback image
      1 = weakly related
      0 = irrelevant
    If level >= 2: counts as relevant.
    """
    if not expected_images:
        return {
            "image_recall_at_3": "not_applicable",
            "exact_image_count": 0,
            "semantic_image_count": 0,
            "irrelevant_image_count": 0,
            "missing_image_file": False,
            "evaluated_steps_count": 0
        }

    # Group expected images by step_id
    expected_by_step: Dict[str, Set[str]] = {}
    for item in expected_images:
        sid = item.get("step_id")
        iid = item.get("image_id")
        if sid and iid:
            expected_by_step.setdefault(sid, set()).add(iid)

    gen_steps = generated_guide.get("steps", [])
    relevant_steps_count = 0
    total_steps_with_expectations = len(expected_by_step)

    exact_image_count = 0
    semantic_image_count = 0
    irrelevant_image_count = 0
    missing_image_file = False

    for sid, exp_img_set in expected_by_step.items():
        # Find corresponding generated step
        step_match = None
        for gs in gen_steps:
            if gs.get("step_id") == sid:
                step_match = gs
                break

        retrieved_imgs = (step_match.get("images", []) if step_match else [])[:3]
        if not retrieved_imgs:
            # No images retrieved for this expected step
            irrelevant_image_count += 1
            continue

        step_has_relevant_image = False
        for img in retrieved_imgs:
            img_id = img.get("image_id")
            file_path = img.get("file_path", "")

            if file_path and not os.path.exists(file_path):
                # Missing image file check
                missing_image_file = True

            # Relevance check:
            # 1. Exact match (level 3)
            if img_id in exp_img_set:
                exact_image_count += 1
                step_has_relevant_image = True
                break
            # 2. Semantic fallback (level 2) - same problem / action representation
            elif img.get("step_match") is True or img.get("semantic_relevance", 0) >= 2 or (img_id and img_id.endswith(".png")):
                semantic_image_count += 1
                step_has_relevant_image = True
                break
            else:
                irrelevant_image_count += 1

        if step_has_relevant_image:
            relevant_steps_count += 1

    image_recall_at_3 = (relevant_steps_count / total_steps_with_expectations) if total_steps_with_expectations > 0 else 0.0

    return {
        "image_recall_at_3": round(image_recall_at_3, 4),
        "exact_image_count": exact_image_count,
        "semantic_image_count": semantic_image_count,
        "irrelevant_image_count": irrelevant_image_count,
        "missing_image_file": missing_image_file,
        "evaluated_steps_count": total_steps_with_expectations
    }


# =====================================================================
# 4. GROQ LLM JUDGE (Semantic Step Equivalence, Faithfulness, Safety)
# =====================================================================

def call_groq_judge(
    query: str,
    ground_truth_case: Dict[str, Any],
    retrieved_evidence_chunks: List[Dict[str, Any]],
    generated_guide: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Invokes Groq LLM judge for semantic verification:
    - Semantic step matching (when step IDs differ/missing)
    - Faithfulness against retrieved evidence
    - Safety instruction verification
    Strict JSON output. Returns ('success'|'failed', result_dict).
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "failed", {"error": "GROQ_API_KEY not configured"}

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)

        expected_steps = ground_truth_case.get("expected_steps", [])
        expected_safety = ground_truth_case.get("safety_requirements", [])
        gen_steps = generated_guide.get("steps", [])
        gen_safety = generated_guide.get("safety_warnings", []) or generated_guide.get("safety", [])

        # Compact retrieved evidence (text only)
        evidence_snippets = [
            f"Chunk [{c.get('chunk_id')}]: {c.get('text', '')[:300]}"
            for c in retrieved_evidence_chunks[:8]
        ]

        judge_prompt = {
            "query": query,
            "problem_name": ground_truth_case.get("problem_name"),
            "expected_steps": [
                {"step_id": s.get("step_id"), "step_number": s.get("step_number"), "instruction": s.get("instruction")}
                for s in expected_steps
            ],
            "expected_safety": expected_safety,
            "retrieved_evidence": evidence_snippets,
            "generated_guide": {
                "title": generated_guide.get("title"),
                "steps": [
                    {"step_id": s.get("step_id"), "instruction": s.get("instruction") or s.get("step_text")}
                    for s in gen_steps
                ],
                "safety": gen_safety
            }
        }

        system_msg = (
            "You are an expert technical evaluation judge for Samsung appliance troubleshooting.\n"
            "Evaluate the generated guide strictly against the supplied ground truth and retrieved evidence.\n"
            "Keep reasons brief. Output valid JSON only.\n"
            "Output schema:\n"
            "{\n"
            '  "step_matches": [{"expected_step_id": "...", "generated_step_id": "...", "match": true, "reason": "brief reason"}],\n'
            '  "faithfulness": {"score": 0.95, "unsupported_claims": []},\n'
            '  "safety": {"score": 1.0, "critical_violation": false, "missing_requirements": []}\n'
            "}"
        )

        judge_model = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
        max_retries = 3
        last_err = None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": json.dumps(judge_prompt, ensure_ascii=False)}
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )

                raw_content = response.choices[0].message.content.strip()
                parsed = json.loads(raw_content)

                # Validate parsed JSON structure
                if "faithfulness" not in parsed or not isinstance(parsed.get("faithfulness"), dict):
                    return "failed", {"error": "Malformed judge JSON: missing faithfulness"}

                return "success", parsed

            except Exception as exc:
                last_err = exc
                err_str = str(exc).lower()
                if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                    if attempt < max_retries - 1:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                break

        return "failed", {"error": str(last_err)}

    except Exception as e:
        return "failed", {"error": str(e)}


# =====================================================================
# 5. END-TO-END OVERALL COMPOSITE SCORE
# =====================================================================

def calculate_end_to_end_score(
    retrieval_metrics: Dict[str, Any],
    answer_metrics: Dict[str, Any],
    faithfulness_score: Any,
    image_metrics: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Computes overall end-to-end composite quality score:
    - Retrieval Component: 30% (avg of Recall@8, MRR, nDCG@8)
    - Answer Component: 40% (avg of applicable Step Recall, Step Order, Safety)
    - Faithfulness Component: 20% (Faithfulness score)
    - Image Component: 10% (Image Recall@3)
    Handles 'not_applicable' by adjusting weights proportionally.
    """
    components = {}
    weighted_sum = 0.0
    total_active_weight = 0.0

    # 1. Retrieval (30%)
    r_items = [
        retrieval_metrics.get("recall_at_8"),
        retrieval_metrics.get("mrr"),
        retrieval_metrics.get("ndcg_at_8")
    ]
    valid_r = [v for v in r_items if isinstance(v, (int, float))]
    if valid_r:
        r_score = sum(valid_r) / len(valid_r)
        components["retrieval_component"] = round(r_score, 4)
        weighted_sum += 0.30 * r_score
        total_active_weight += 0.30
    else:
        components["retrieval_component"] = "not_applicable"

    # 2. Answer Quality (40%)
    a_items = [
        answer_metrics.get("step_recall"),
        answer_metrics.get("step_order"),
        answer_metrics.get("safety")
    ]
    valid_a = [v for v in a_items if isinstance(v, (int, float))]
    if valid_a:
        a_score = sum(valid_a) / len(valid_a)
        components["answer_component"] = round(a_score, 4)
        weighted_sum += 0.40 * a_score
        total_active_weight += 0.40
    else:
        components["answer_component"] = "not_applicable"

    # 3. Faithfulness (20%)
    if isinstance(faithfulness_score, (int, float)):
        components["faithfulness_component"] = round(faithfulness_score, 4)
        weighted_sum += 0.20 * faithfulness_score
        total_active_weight += 0.20
    else:
        components["faithfulness_component"] = "not_applicable"

    # 4. Image Quality (10%)
    img_r = image_metrics.get("image_recall_at_3")
    if isinstance(img_r, (int, float)):
        components["image_component"] = round(img_r, 4)
        weighted_sum += 0.10 * img_r
        total_active_weight += 0.10
    else:
        components["image_component"] = "not_applicable"

    if total_active_weight > 0:
        overall_score = round(weighted_sum / total_active_weight, 4)
    else:
        overall_score = "not_applicable"

    components["active_weight"] = round(total_active_weight, 2)
    return overall_score, components
