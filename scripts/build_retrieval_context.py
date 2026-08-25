import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retrieve_text import retrieve_text
from scripts.retrieve_images import retrieve_images
from backend.pipeline_logger import pipeline_logger

def format_output_json(context):
    return json.dumps(context, indent=4, ensure_ascii=False)

def group_chunks_by_problem(text_results):
    problems = {}
    for chunk in text_results:
        doc_id = chunk.get("document_id")
        prob_id = chunk.get("problem_id")
        if not doc_id or not prob_id:
            continue
            
        group_key = f"{doc_id}_{prob_id}"
        if group_key not in problems:
            problems[group_key] = {
                "document_id": doc_id,
                "problem_id": prob_id,
                "problem_name": chunk.get("problem_name"),
                "chunks": []
            }
        problems[group_key]["chunks"].append(chunk)
    return problems

def _extract_step_text_from_chunk(s_dict, chunk_text):
    text = s_dict.get("text") or s_dict.get("step_text")
    if text and text.strip():
        return text.strip()
    step_num = s_dict.get("step_number")
    if step_num is not None and chunk_text:
        pattern = rf"(?:Step\s*{step_num}[.:\-\s]+)(.+?)(?=(?:Step\s*\d+[.:\-\s]+|When to escalate|\Z))"
        m = re.search(pattern, chunk_text, re.DOTALL | re.IGNORECASE)
        if m:
            clean_text = " ".join(m.group(1).split()).strip()
            clean_text = re.sub(r"\s*Samsung\s+[A-Za-z0-9/\-]+\s*.*$", "", clean_text, flags=re.IGNORECASE).strip()
            return clean_text
    return ""

def reconstruct_steps(chunks, validation):
    steps = {}
    
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        chunk_steps = chunk.get("steps", [])
        chunk_text = chunk.get("text", "")
        
        for s in chunk_steps:
            step_id = s.get("step_id")
            if not step_id:
                validation["invalid_sources"].append({"chunk_id": chunk_id, "reason": "Missing step_id"})
                continue
                
            extracted_text = _extract_step_text_from_chunk(s, chunk_text)
            
            if step_id not in steps:
                steps[step_id] = {
                    "step_id": step_id,
                    "step_number": s.get("step_number"),
                    "step_text": extracted_text,
                    "source": {
                        "chunk_ids": [],
                        "pages": set()
                    }
                }
            else:
                # Deduplicate step
                validation["duplicate_steps_removed"] += 1
                # Check for conflicting text
                existing_text = steps[step_id]["step_text"]
                new_text = extracted_text
                if not existing_text and new_text:
                    steps[step_id]["step_text"] = new_text
                elif existing_text and new_text and existing_text.strip() != new_text.strip():
                    validation["conflicting_steps"].append({
                        "step_id": step_id,
                        "conflict": [existing_text, new_text]
                    })
                    
            # Record provenance
            steps[step_id]["source"]["chunk_ids"].append(chunk_id)
            pg_start = chunk.get("page_start")
            pg_end = chunk.get("page_end")
            if pg_start: steps[step_id]["source"]["pages"].add(pg_start)
            if pg_end: steps[step_id]["source"]["pages"].add(pg_end)
            
    # Sort and finalize
    final_steps = []
    for s_id, s_data in steps.items():
        s_data["source"]["pages"] = sorted(list(s_data["source"]["pages"]))
        # Remove duplicates from chunk_ids just in case
        s_data["source"]["chunk_ids"] = sorted(list(set(s_data["source"]["chunk_ids"])))
        final_steps.append(s_data)
        
    final_steps.sort(key=lambda x: (x.get("step_number") or 999999, x.get("step_id", "")))
    
    # Check for missing intermediate steps
    if final_steps:
        known_numbers = [s["step_number"] for s in final_steps if s["step_number"] is not None]
        if known_numbers:
            min_num = min(known_numbers)
            max_num = max(known_numbers)
            for i in range(min_num, max_num):
                if i not in known_numbers:
                    validation["missing_steps"].append(i)
                    
    return final_steps

def build_retrieval_context(query, appliance_type="washing_machine", brand="Samsung", model=None,
                            document_id=None, problem_id=None, text_top_k=8, image_top_k=3,
                            min_image_similarity=None, model_context: Optional[Dict] = None,
                            retrieval_mode: Optional[str] = None):
    
    if not query or not query.strip():
        print("Error: Query cannot be empty.")
        sys.exit(1)
        
    if text_top_k <= 0 or image_top_k <= 0:
        print("Error: text_top_k and image_top_k must be positive.")
        sys.exit(1)

    # Determine retrieval mode and model from model_context if provided
    if model_context:
        effective_mode = model_context.get("retrieval_mode", "model_specific")
        target_model = model_context.get("database_model") or model
        canonical_model = model_context.get("canonical_model")
        requested_model = model_context.get("requested_model")
        model_known = model_context.get("model_known", False)
        warning = model_context.get("warning")
    else:
        effective_mode = retrieval_mode or ("generic" if not model else "model_specific")
        target_model = model
        canonical_model = model
        requested_model = model
        model_known = bool(model)
        warning = None

    constructed_model_context = {
        "requested_model": requested_model,
        "canonical_model": canonical_model,
        "database_model": target_model if model_known else None,
        "model_known": model_known,
        "retrieval_mode": effective_mode,
        "warning": warning
    }
        
    validation = {
        "missing_steps": [],
        "missing_images": [],
        "duplicate_steps_removed": 0,
        "conflicting_steps": [],
        "invalid_sources": [],
        "image_retrieval_errors": []
    }
    
    try:
        text_res = retrieve_text(
            query=query, appliance_type=appliance_type, brand=brand, model=target_model,
            document_id=document_id, problem_id=problem_id, final_top_k=text_top_k,
            retrieval_mode=effective_mode
        )
    except Exception as e:
        print(f"Error: Text retrieval failed: {e}")
        sys.exit(1)
        
    text_chunks = text_res.get("results", [])
    
    if not text_chunks:
        return {
            "schema_version": "1.0",
            "query": query,
            "request_context": {
                "appliance_type": appliance_type,
                "brand": brand,
                "model": target_model
            },
            "model_context": constructed_model_context,
            "retrieval": {
                "text": {"top_k": text_top_k, "retrieval_mode": effective_mode},
                "images": {"top_k_per_step": image_top_k}
            },
            "status": "no_text_evidence",
            "problems": [],
            "validation": validation
        }

    import time
    stage8_start = time.perf_counter()
    grouped_problems = group_chunks_by_problem(text_chunks)
    
    # Pre-sort candidate problems by max relevance score and chunk count before image retrieval
    candidate_problems = []
    for key, prob_data in grouped_problems.items():
        chunks = prob_data["chunks"]
        max_rrf = max([c.get("rrf_score", 0.0) for c in chunks]) if chunks else 0.0
        max_rerank = max([c.get("rerank_score") for c in chunks if c.get("rerank_score") is not None], default=None)
        effective_score = max_rerank if max_rerank is not None else max_rrf
        candidate_problems.append({
            "prob_data": prob_data,
            "chunks": chunks,
            "max_rrf": max_rrf,
            "max_rerank": max_rerank,
            "effective_score": effective_score,
            "chunk_count": len(chunks),
            "problem_id": prob_data.get("problem_id", "")
        })
        
    candidate_problems.sort(key=lambda x: (
        -x["effective_score"],
        -x["chunk_count"],
        x["problem_id"]
    ))
    
    # Limit to top 3 problems
    candidate_problems = candidate_problems[:3]
    
    final_problems = []
    total_image_latency_ms = 0.0
    
    for candidate in candidate_problems:
        prob_data = candidate["prob_data"]
        chunks = candidate["chunks"]
        max_rrf = candidate["max_rrf"]
        max_rerank = candidate["max_rerank"]
        
        supporting_chunks = []
        for c in chunks:
            supporting_chunks.append({
                "chunk_id": c.get("chunk_id"),
                "parent_chunk_id": c.get("parent_chunk_id"),
                "rrf_score": c.get("rrf_score"),
                "rerank_score": c.get("rerank_score"),
                "dense_score": c.get("dense_score"),
                "sparse_score": c.get("sparse_score"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end")
            })
            
        steps = reconstruct_steps(chunks, validation)
        
        for step in steps:
            step["source"]["document_id"] = prob_data["document_id"]
            step["source"]["problem_id"] = prob_data["problem_id"]
            
            p_name = prob_data["problem_name"] or ""
            s_text = step["step_text"] or ""
            image_query = f"{p_name.strip()} {s_text.strip()}".strip()
            step["image_query"] = image_query
            
            step["images"] = []
            
            try:
                img_t0 = time.perf_counter()
                img_res = retrieve_images(
                    query=image_query,
                    appliance_type=appliance_type,
                    brand=brand,
                    model=target_model,
                    step_id=step["step_id"],
                    top_k=image_top_k,
                    retrieval_mode=effective_mode
                )
                total_image_latency_ms += (time.perf_counter() - img_t0) * 1000.0
                raw_images = img_res.get("results", [])
                
                # Deduplicate images within this step
                seen_image_ids = set()
                filtered_images = []
                removed_image_ids = []
                
                for img in raw_images:
                    img_id = img.get("image_id")
                    if img_id in seen_image_ids:
                        removed_image_ids.append(img_id)
                        continue
                        
                    sem_score = img.get("semantic_score", 0.0)
                    if min_image_similarity is not None and sem_score < min_image_similarity:
                        continue
                        
                    seen_image_ids.add(img_id)
                    filtered_images.append({
                        "image_id": img_id,
                        "file_path": img.get("file_path"),
                        "semantic_score": sem_score,
                        "step_match": img.get("step_match", False),
                        "image_scope": img.get("image_scope", "generic" if not model_known else "model_specific"),
                        "original_step_id": img.get("step_id"),
                        "original_problem_id": img.get("problem_id"),
                        "original_problem_name": img.get("problem_name"),
                        "original_step_number": img.get("step_number"),
                        "dense_caption": img.get("dense_caption"),
                        "model": img.get("model")
                    })
                    
                step["images"] = filtered_images
                
                if removed_image_ids:
                    pipeline_logger.log_image_deduplication(
                        before_count=len(raw_images),
                        after_count=len(filtered_images),
                        removed_ids=removed_image_ids
                    )

                pipeline_logger.log_image_step_retrieval(
                    step_id=step["step_id"],
                    step_number=step.get("step_number", 1),
                    problem_name=p_name,
                    image_query=image_query,
                    model=target_model,
                    retrieval_mode=effective_mode,
                    top_k=image_top_k,
                    all_candidates=raw_images,
                    final_images=filtered_images,
                    fallback_used=img_res.get("retrieval", {}).get("fallback_used", False),
                    fallback_reason=img_res.get("retrieval", {}).get("fallback_reason")
                )
                
                if not filtered_images:
                    validation["missing_images"].append({
                        "step_id": step["step_id"],
                        "reason": "No relevant image found"
                    })
                    
            except Exception as e:
                validation["image_retrieval_errors"].append({
                    "step_id": step["step_id"],
                    "error": str(e)
                })
                
        final_problems.append({
            "problem_id": prob_data["problem_id"],
            "problem_name": prob_data["problem_name"],
            "document_id": prob_data["document_id"],
            "relevance": {
                "max_rrf_score": max_rrf,
                "max_rerank_score": max_rerank,
                "supporting_chunk_count": len(chunks)
            },
            "supporting_chunks": supporting_chunks,
            "steps": steps
        })
    
    stage8_latency_ms = (time.perf_counter() - stage8_start) * 1000.0
    ctx = pipeline_logger.get_context()
    if ctx:
        ctx.stage8_ms = round(stage8_latency_ms, 2)
        ctx.image_latency_ms = round(total_image_latency_ms, 2)

    pipeline_logger.log_stage8_reconstruction(
        input_candidate_count=len(text_chunks),
        problems=final_problems,
        model=target_model,
        retrieval_mode=effective_mode,
        latency_ms=stage8_latency_ms
    )

    return {
        "schema_version": "1.0",
        "query": query,
        "request_context": {
            "appliance_type": appliance_type,
            "brand": brand,
            "model": target_model
        },
        "model_context": constructed_model_context,
        "retrieval": {
            "text": {
                "top_k": text_top_k,
                "retrieval_mode": effective_mode,
                "reranking": text_res.get("retrieval", {}).get("reranking", {})
            },
            "images": {"top_k_per_step": image_top_k}
        },
        "status": "success",
        "problems": final_problems,
        "validation": validation
    }

def print_cli_output(context):
    print("==================================================")
    print("GUIDE WEAVE — STAGE 8")
    print("RETRIEVAL ORCHESTRATION")
    print("==================================================")
    
    if context["status"] == "no_text_evidence":
        print("\nNo text evidence found.")
        print("==================================================")
        return
        
    total_chunks = sum([len(p["supporting_chunks"]) for p in context["problems"]])
    total_steps = sum([len(p["steps"]) for p in context["problems"]])
    total_images = sum([sum([len(s["images"]) for s in p["steps"]]) for p in context["problems"]])
    steps_without_images = sum([sum([1 for s in p["steps"] if not s["images"]]) for p in context["problems"]])
    
    print(f"\nQuery:\n    {context['query']}")
    if "model_context" in context:
        mc = context["model_context"]
        print(f"\nModel Context:\n    Requested: {mc.get('requested_model')}\n    Database Model: {mc.get('database_model')}\n    Mode: {mc.get('retrieval_mode')}\n    Known: {mc.get('model_known')}")
    print(f"\nText chunks retrieved:\n    {total_chunks}")
    print(f"\nProblems identified:\n    {len(context['problems'])}")
    print(f"\nSteps reconstructed:\n    {total_steps}")
    print(f"\nImages retrieved:\n    {total_images}")
    print(f"\nSteps without images:\n    {steps_without_images}")
    
    missing_steps = context.get("validation", {}).get("missing_steps", [])
    print(f"\nMissing intermediate steps:\n    {len(missing_steps)}")
    
    print("\n==================================================")
    
    for prob in context["problems"]:
        print("\n--------------------------------------------------")
        print(f"Problem:\n    {prob.get('problem_name')}")
        print(f"\nProblem ID:\n    {prob.get('problem_id')}")
        print(f"\nSupporting chunks:\n    {prob['relevance']['supporting_chunk_count']}")
        print(f"\nSteps:\n    {len(prob['steps'])}")
        print("--------------------------------------------------")
        
        for step in prob["steps"]:
            print(f"\n    Step {step.get('step_number')}")
            print(f"    Step ID: {step.get('step_id')}")
            print(f"    Text: {step.get('step_text')}")
            
            print(f"\n    Images:")
            if not step["images"]:
                print("        (None)")
            else:
                for idx, img in enumerate(step["images"], start=1):
                    print(f"        {idx}. {img.get('file_path').split('/')[-1]}")
                    print(f"           score: {img.get('semantic_score'):.4f}")
                    print(f"           scope: {img.get('image_scope')}")
                    print(f"           exact step: {str(img.get('step_match')).lower()}")
            print("")

def main():
    parser = argparse.ArgumentParser(description="Retrieval Orchestration for Guide Weave")
    parser.add_argument("--query", type=str, help="The user query")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--appliance-type", type=str, default="washing_machine")
    parser.add_argument("--brand", type=str, default="Samsung")
    parser.add_argument("--model", type=str)
    parser.add_argument("--generic", action="store_true", help="Run generic retrieval")
    parser.add_argument("--text-top-k", type=int, default=8)
    parser.add_argument("--image-top-k", type=int, default=3)
    parser.add_argument("--min-image-similarity", type=float)
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    if args.interactive:
        while True:
            try:
                q = input("\nQuery > ")
            except EOFError:
                break
                
            q = q.strip()
            if q.lower() in ["exit", "quit"]:
                break
            if not q:
                continue
                
            retrieval_mode = "generic" if args.generic else "model_specific"
            res = build_retrieval_context(
                q, model=args.model,
                text_top_k=args.text_top_k,
                image_top_k=args.image_top_k,
                retrieval_mode=retrieval_mode
            )
            print_cli_output(res)
    elif args.query:
        retrieval_mode = "generic" if args.generic else "model_specific"
        res = build_retrieval_context(
            args.query,
            appliance_type=args.appliance_type,
            brand=args.brand,
            model=args.model,
            text_top_k=args.text_top_k,
            image_top_k=args.image_top_k,
            min_image_similarity=args.min_image_similarity,
            retrieval_mode=retrieval_mode
        )
        print_cli_output(res)
        
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(format_output_json(res))
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
