"""
Guide Weave — Benchmark Dataset Generator (Stage 10)
===================================================
Generates benchmark_data/benchmark_dataset.json from finalized ground-truth
data in chunked_ground_truth/, extracted_ground_truth/, and image_knowledge_base_final.json.

This script is strictly READ-ONLY with respect to original ground-truth data
and Qdrant vector databases.
"""

import os
import sys
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "benchmark_data" / "benchmark_dataset.json"


def generate_natural_queries(problem_name: str) -> List[str]:
    """
    Generate 3 realistic, semantically faithful user query variants for a problem.
    Does not invent symptoms or technical details outside the problem definition.
    """
    clean = re.sub(r"^\d+[\.\:\-\s]+", "", problem_name).strip()
    clean = re.sub(r"\s*[\u2013\u2014\-]\s*Detailed.*$", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s*[\u2013\u2014\-]\s*Troubleshooting.*$", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s*Manual\s*$", "", clean, flags=re.IGNORECASE).strip()
    clean = clean.rstrip(".").strip()

    # Determine topic with washer prefix removed if present
    has_washer_prefix = bool(re.match(r"^(?:Washer|Washing machine)\s+", clean, re.IGNORECASE))
    core_phrase = re.sub(r"^(?:Washer|Washing machine)\s+", "", clean, re.IGNORECASE).strip()

    # Case-insensitive checks
    full_lower = clean.lower()
    core_lower = core_phrase.lower()

    if has_washer_prefix:
        if re.match(r"^(?:does not|doesn't|won't|cannot|fails to)\s+", core_lower):
            v_phrase = re.sub(r"^(?:does not|doesn't|won't|cannot|fails to)\s+", "", core_phrase, flags=re.IGNORECASE).strip()
            q1 = f"How do I fix a washer that does not {v_phrase}?"
            q2 = f"My Samsung washer won't {v_phrase}."
            q3 = f"The washer cannot {v_phrase} properly."
        elif re.match(r"^(?:is |is not|isn't|keeps |stops |overfills|underfills|drains|fills|leaks|vibrates|shakes|beeps|reports)", core_lower):
            q1 = f"How do I troubleshoot a washer that {core_phrase}?"
            q2 = f"My Samsung washer {core_phrase}."
            q3 = f"What should I do when the washer {core_phrase}?"
        else:
            q1 = f"How do I resolve washer {core_phrase}?"
            q2 = f"My Samsung washer has a problem: {core_phrase}."
            q3 = f"Troubleshooting steps for washer {core_phrase}."
    else:
        if "child lock" in full_lower:
            q1 = f"How do I disable or resolve {clean} on my Samsung washer?"
            q2 = f"Samsung washer controls locked: {clean}."
            q3 = f"Troubleshoot {clean} on Samsung washer."
        elif "circuit breaker" in full_lower or "fuse" in full_lower:
            q1 = "How do I fix a Samsung washer that trips the circuit breaker or fuse?"
            q2 = "My Samsung washer keeps tripping the breaker when running."
            q3 = "Electrical fuse / breaker trips when using Samsung washer."
        elif "screen" in full_lower or "filter" in full_lower:
            q1 = f"How do I clean or inspect {clean} on Samsung washer?"
            q2 = f"My Samsung washer {clean}."
            q3 = f"Steps to check and clear {clean} on washer."
        elif "hose" in full_lower:
            q1 = f"How to fix {clean} on Samsung washer?"
            q2 = f"Samsung washer {clean}."
            q3 = f"Inspection and troubleshooting for {clean} on washer."
        elif "door" in full_lower or "lid" in full_lower or "lock" in full_lower:
            q1 = f"How do I troubleshoot {clean} on my Samsung washer?"
            q2 = f"My Samsung washer has a door/lid problem: {clean}."
            q3 = f"Samsung washer {clean}."
        elif "leak" in full_lower:
            q1 = f"How to find and fix {clean} on Samsung washer?"
            q2 = f"My Samsung washer has a leak: {clean}."
            q3 = f"Water leakage from washer: {clean}."
        elif "noise" in full_lower or "vibration" in full_lower:
            q1 = f"How do I stop {clean} on Samsung washer?"
            q2 = f"Samsung washer {clean} during cycle."
            q3 = f"Troubleshoot abnormal {clean} on washer."
        elif "sensor" in full_lower or "fault" in full_lower or "error" in full_lower or "code" in full_lower:
            q1 = f"How do I diagnose {clean} on Samsung washer?"
            q2 = f"My Samsung washer indicates {clean}."
            q3 = f"Troubleshooting guide for {clean} on washer."
        else:
            q1 = f"How do I fix {clean} on my Samsung washer?"
            q2 = f"My Samsung washer has a problem: {clean}."
            q3 = f"Troubleshooting guide for {clean} on washer."

    # Normalize whitespace
    return [" ".join(q1.split()), " ".join(q2.split()), " ".join(q3.split())]


def load_image_mapping(img_kb_path: Path) -> Dict[str, List[str]]:
    """Load step_id to image_id mapping from image_knowledge_base_final.json."""
    if not img_kb_path.exists():
        return {}
    with open(img_kb_path, "r", encoding="utf-8") as f:
        images_data = json.load(f)

    step_to_images: Dict[str, List[str]] = {}
    for img in images_data:
        sid = img.get("step_id")
        iid = img.get("image_id")
        if sid and iid:
            step_to_images.setdefault(sid, []).append(iid)
    return step_to_images


def build_benchmark_dataset() -> Dict[str, Any]:
    """
    Constructs the normalized evaluation representation from chunked and extracted ground truth.
    """
    gt_dir = PROJECT_ROOT / "chunked_ground_truth"
    img_kb_path = PROJECT_ROOT / "image_knowledge_base_final.json"
    gt_files = sorted(glob.glob(str(gt_dir / "*.json")))

    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files found in {gt_dir}")

    step_to_images = load_image_mapping(img_kb_path)

    cases: List[Dict[str, Any]] = []
    case_counter = 0

    for fpath in gt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id = data.get("document_id")

        # Map problem_id -> list of child chunks
        prob_children: Dict[str, List[Dict]] = {}
        for c in data.get("children", []):
            pid = c.get("problem_id")
            if pid:
                prob_children.setdefault(pid, []).append(c)

        # Map problem_id -> list of parent chunks
        prob_parents: Dict[str, List[Dict]] = {}
        for p in data.get("parents", []):
            pid = p.get("problem_id")
            if pid:
                prob_parents.setdefault(pid, []).append(p)

        for pid, parents in prob_parents.items():
            case_counter += 1
            benchmark_id = f"case_{case_counter:04d}"
            pname = parents[0].get("problem_name", "")
            children = prob_children.get(pid, [])

            # Collect expected steps
            steps_map: Dict[str, Dict[str, Any]] = {}
            for c in children:
                cid = c.get("chunk_id")
                c_text = c.get("text", "")
                pg_start = c.get("page_start")
                pg_end = c.get("page_end")
                pages = sorted(list(set([p for p in [pg_start, pg_end] if p is not None])))

                for s in c.get("steps", []):
                    sid = s.get("step_id")
                    snum = s.get("step_number")
                    if not sid:
                        continue

                    # Extract instruction text for this step
                    pattern = (
                        rf"(?:Step\s*{snum}[.:\-\s]+)(.+?)"
                        rf"(?=(?:Step\s*\d+[.:\-\s]+|When to escalate|After completing the steps|Safety:|\Z))"
                    )
                    m = re.search(pattern, c_text, re.DOTALL | re.IGNORECASE)
                    step_text = ""
                    if m:
                        step_text = " ".join(m.group(1).split()).strip()
                        step_text = re.sub(r"\s*Samsung\s+[A-Za-z0-9/\-]+\s*.*$", "", step_text, flags=re.IGNORECASE).strip()
                        step_text = re.sub(r"IciteI[^\s]+", "", step_text).strip()

                    if sid not in steps_map:
                        steps_map[sid] = {
                            "step_id": sid,
                            "step_number": snum,
                            "instruction": step_text,
                            "chunk_ids": [cid] if cid else [],
                            "pages": pages
                        }
                    else:
                        if cid and cid not in steps_map[sid]["chunk_ids"]:
                            steps_map[sid]["chunk_ids"].append(cid)
                        for pg in pages:
                            if pg not in steps_map[sid]["pages"]:
                                steps_map[sid]["pages"].append(pg)
                        if not steps_map[sid]["instruction"] and step_text:
                            steps_map[sid]["instruction"] = step_text

            sorted_steps = sorted(steps_map.values(), key=lambda x: (x["step_number"] or 0))

            # Expected images
            expected_images: List[Dict[str, str]] = []
            for s in sorted_steps:
                sid = s["step_id"]
                if sid in step_to_images:
                    for iid in step_to_images[sid]:
                        expected_images.append({
                            "step_id": sid,
                            "image_id": iid
                        })

            # Safety requirements (from explicit ground-truth "Safety: ..." statements)
            safety_requirements: List[str] = []
            for p in parents:
                p_text = p.get("text", "")
                s_matches = re.findall(r"Safety:\s*([^\n\r]+)", p_text, re.IGNORECASE)
                for sm in s_matches:
                    clean_s = sm.strip()
                    if clean_s and clean_s not in safety_requirements:
                        safety_requirements.append(clean_s)

            # Generate query variants
            queries = generate_natural_queries(pname)

            cases.append({
                "benchmark_id": benchmark_id,
                "document_id": doc_id,
                "problem_id": pid,
                "problem_name": pname,
                "queries": queries,
                "expected_steps": sorted_steps,
                "expected_images": expected_images,
                "safety_requirements": safety_requirements
            })

    dataset = {
        "benchmark_version": "1.0",
        "model": "WA5471ABP",
        "database_model": "WA5471ABP/XAA",
        "appliance_type": "washing_machine",
        "brand": "Samsung",
        "cases": cases
    }
    return dataset


def validate_benchmark_dataset(dataset: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates benchmark dataset integrity according to Section 8:
    - benchmark_id uniqueness
    - document_id validity
    - problem_id validity
    - step_id validity
    - chunk_id validity
    - step ordering
    - duplicate expected steps
    - expected image references
    - safety requirement structure
    """
    errors: List[str] = []
    cases = dataset.get("cases", [])
    if not cases:
        errors.append("Dataset has no cases.")
        return False, errors

    benchmark_ids = set()
    problem_ids = set()

    for idx, case in enumerate(cases):
        bid = case.get("benchmark_id")
        doc_id = case.get("document_id")
        pid = case.get("problem_id")
        pname = case.get("problem_name")
        queries = case.get("queries", [])
        expected_steps = case.get("expected_steps", [])
        expected_images = case.get("expected_images", [])
        safety_reqs = case.get("safety_requirements", [])

        # 1. benchmark_id uniqueness
        if not bid:
            errors.append(f"Case index {idx}: Missing benchmark_id.")
        elif bid in benchmark_ids:
            errors.append(f"Duplicate benchmark_id found: {bid}")
        else:
            benchmark_ids.add(bid)

        # 2. document_id validity
        if not doc_id or not isinstance(doc_id, str):
            errors.append(f"Case {bid}: Invalid or missing document_id.")

        # 3. problem_id validity
        if not pid or not isinstance(pid, str):
            errors.append(f"Case {bid}: Invalid or missing problem_id.")
        elif pid in problem_ids:
            errors.append(f"Duplicate problem_id found across cases: {pid}")
        else:
            problem_ids.add(pid)

        # 4. problem_name
        if not pname:
            errors.append(f"Case {bid}: Missing problem_name.")

        # 5. queries
        if not queries or not isinstance(queries, list) or len(queries) < 1:
            errors.append(f"Case {bid}: Missing queries list.")
        for q in queries:
            if not isinstance(q, str) or not q.strip():
                errors.append(f"Case {bid}: Empty or invalid query string.")

        # 6. step_id validity & step ordering & duplicate steps
        seen_step_ids = set()
        seen_step_numbers = []
        for s in expected_steps:
            sid = s.get("step_id")
            snum = s.get("step_number")
            cids = s.get("chunk_ids", [])
            pages = s.get("pages", [])

            if not sid:
                errors.append(f"Case {bid}: Step missing step_id.")
            elif sid in seen_step_ids:
                errors.append(f"Case {bid}: Duplicate step_id {sid}.")
            else:
                seen_step_ids.add(sid)

            if snum is None or not isinstance(snum, int):
                errors.append(f"Case {bid}: Invalid step_number {snum} for step {sid}.")
            else:
                seen_step_numbers.append(snum)

            # chunk_id validity
            if not isinstance(cids, list):
                errors.append(f"Case {bid}: step {sid} chunk_ids must be a list.")
            elif len(cids) == 0 and len(expected_steps) > 0:
                errors.append(f"Case {bid}: step {sid} has empty chunk_ids.")

            # pages validity
            if not isinstance(pages, list):
                errors.append(f"Case {bid}: step {sid} pages must be a list.")

        # Check step ordering if steps exist
        if seen_step_numbers:
            if seen_step_numbers != sorted(seen_step_numbers):
                errors.append(f"Case {bid}: Step numbers are out of order: {seen_step_numbers}")

        # 7. expected image references
        for img in expected_images:
            sid = img.get("step_id")
            iid = img.get("image_id")
            if not sid or sid not in seen_step_ids:
                errors.append(f"Case {bid}: Image reference {iid} points to unknown step_id {sid}.")
            if not iid or not isinstance(iid, str) or not iid.endswith(".png"):
                errors.append(f"Case {bid}: Invalid image_id {iid}.")

        # 8. safety requirement structure
        if not isinstance(safety_reqs, list):
            errors.append(f"Case {bid}: safety_requirements must be a list.")
        for sr in safety_reqs:
            if not isinstance(sr, str) or not sr.strip():
                errors.append(f"Case {bid}: Invalid safety requirement string: {sr}")

    is_valid = (len(errors) == 0)
    return is_valid, errors


def main():
    print("Generating benchmark dataset...")
    dataset = build_benchmark_dataset()

    print("Validating benchmark dataset...")
    is_valid, errors = validate_benchmark_dataset(dataset)

    if not is_valid:
        print(f"Validation FAILED with {len(errors)} errors:")
        for err in errors[:20]:
            print(f"  [ERROR] {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)

    print("Validation PASSED successfully.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    total_cases = len(dataset["cases"])
    total_queries = sum(len(c["queries"]) for c in dataset["cases"])
    total_steps = sum(len(c["expected_steps"]) for c in dataset["cases"])
    total_images = sum(len(c["expected_images"]) for c in dataset["cases"])
    safety_cases = sum(1 for c in dataset["cases"] if c["safety_requirements"])

    print(f"\nBenchmark Dataset Summary:")
    print(f"  Output File: {OUTPUT_FILE}")
    print(f"  Version: {dataset['benchmark_version']}")
    print(f"  Target Model: {dataset['model']} ({dataset['database_model']})")
    print(f"  Total Problems (Cases): {total_cases}")
    print(f"  Total Query Variants: {total_queries}")
    print(f"  Total Expected Steps: {total_steps}")
    print(f"  Total Expected Images: {total_images}")
    print(f"  Cases with Safety Requirements: {safety_cases}")


if __name__ == "__main__":
    main()
