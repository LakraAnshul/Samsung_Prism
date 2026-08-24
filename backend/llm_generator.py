"""
Guide Weave — LLM Generator
Groq API integration for grounded guide generation with strict model scoping
and comprehensive grounding validation.
"""

import json
import os
import re
import time
from typing import Dict, Optional, List

from dotenv import load_dotenv

load_dotenv()
load_dotenv("backend/.env")

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_RETRIES = 2


def _get_groq_client():
    """Lazy Groq client initialization."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Set it in .env.")

    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _build_system_prompt(model_known: bool = True) -> str:
    if model_known:
        scoping_rule = (
            "1. The requested model is present in the indexed database. Use ONLY evidence belonging to this model.\n"
            "   Do NOT use your own knowledge or speculate beyond the provided text."
        )
    else:
        scoping_rule = (
            "1. The requested model is NOT present in the indexed database.\n"
            "   The supplied evidence is generic Samsung washing-machine evidence.\n"
            "   Do NOT claim that it came from the requested model's manual.\n"
            "   Do NOT present generic steps as model-specific (e.g. do not say 'For your <model>').\n"
            "   Use ONLY the provided generic evidence."
        )

    return f"""You are a strict technical troubleshooting guide generator for Samsung appliances.

ABSOLUTE RULES:
{scoping_rule}
2. Every step you output MUST have a step_id that exists in the provided evidence. Do NOT invent new step IDs.
3. Every chunk_id you cite MUST exist in the provided evidence and belong to that step. Do NOT invent chunk IDs.
4. Every image_id you reference MUST exist in the provided evidence and belong to that step. Do NOT invent image IDs or file paths.
5. Do NOT create steps that do not exist in the evidence. If evidence has steps 1, 2, 5 — output steps 1, 2, 5. Do NOT invent steps 3 and 4.
6. Safety warnings MUST come from the evidence text. Do NOT add warnings not present in the source.
7. You may simplify wording for readability but MUST preserve the technical meaning.
8. If the evidence is insufficient, say so in the "limitations" field. Do NOT fabricate an answer.
9. Output valid JSON only. No markdown, no code fences, no explanations outside the JSON object.
10. For grounding confidence, use only: "high", "medium", or "low". Do NOT use numerical percentages."""


def _build_user_prompt(query: str, retrieval_context: Dict, model: str, model_known: bool = True) -> str:
    problems = retrieval_context.get("problems", [])
    top_problems = problems[:2]

    evidence_problems = []
    for prob in top_problems:
        steps_compact = []
        for step in prob.get("steps", []):
            images_compact = []
            for img in step.get("images", [])[:1]:
                images_compact.append({
                    "image_id": img.get("image_id"),
                    "image_scope": img.get("image_scope", "generic" if not model_known else "model_specific")
                })

            steps_compact.append({
                "step_id": step.get("step_id"),
                "step_number": step.get("step_number"),
                "step_text": step.get("step_text") or "",
                "source": {
                    "chunk_ids": step.get("source", {}).get("chunk_ids", []),
                    "pages": step.get("source", {}).get("pages", [])
                },
                "images": images_compact
            })

        evidence_problems.append({
            "problem_id": prob.get("problem_id"),
            "problem_name": prob.get("problem_name"),
            "steps": steps_compact
        })

    evidence_json = json.dumps({
        "query": query,
        "model": model,
        "model_known": model_known,
        "guidance_scope": "model_specific" if model_known else "generic",
        "problems": evidence_problems
    }, ensure_ascii=False)

    guidance_scope = "model_specific" if model_known else "generic"

    return f"""USER QUERY: "{query}"
APPLIANCE MODEL: {model}
MODEL KNOWN IN DATABASE: {str(model_known).lower()}
GUIDANCE SCOPE: {guidance_scope}

RETRIEVAL EVIDENCE (Source of Truth — use ONLY this):
{evidence_json}

Generate a grounded troubleshooting guide using ONLY the evidence above.

OUTPUT FORMAT (JSON ONLY):
{{
    "status": "success",
    "task_title": "Concise task title derived from the evidence",
    "model": "{model}",
    "model_known": {str(model_known).lower()},
    "guidance_scope": "{guidance_scope}",
    "grounding": {{
        "grounded": true,
        "confidence": "high",
        "source_problem_ids": ["list of problem_ids used"]
    }},
    "steps": [
        {{
            "step_number": 1,
            "step_id": "exact step_id from evidence",
            "instruction": "Clear instruction text based on evidence step_text",
            "safety_warning": null,
            "source": {{
                "chunk_ids": ["from evidence"],
                "pages": [4]
            }},
            "images": [
                {{
                    "image_id": "exact image_id from evidence"
                }}
            ]
        }}
    ],
    "limitations": []
}}"""


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """Parse the LLM response, handling common formatting issues."""
    if not raw or not raw.strip():
        return None

    cleaned = _strip_json_fences(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _validate_grounding(llm_output: Dict, retrieval_context: Dict, model_known: bool = True) -> Dict:
    """
    Validate that all IDs in the LLM output exist, are strictly related to the respective step,
    steps are ordered ascending, and there are no duplicates or hallucinations.
    """
    validation_errors = []

    # Build relational map from retrieval context
    valid_step_relations = {}
    valid_problem_ids = set()
    valid_document_ids = set()
    authoritative_images = {}  # image_id -> full image dict

    for prob in retrieval_context.get("problems", []):
        pid = prob.get("problem_id")
        did = prob.get("document_id")
        if pid: valid_problem_ids.add(pid)
        if did: valid_document_ids.add(did)

        for step in prob.get("steps", []):
            sid = step.get("step_id")
            if not sid:
                continue
            
            allowed_chunks = set()
            allowed_images = set()

            for src_cid in step.get("source", {}).get("chunk_ids", []):
                allowed_chunks.add(src_cid)
            
            for img in step.get("images", []):
                iid = img.get("image_id")
                if iid:
                    allowed_images.add(iid)
                    if iid not in authoritative_images:
                        authoritative_images[iid] = img
            
            valid_step_relations[sid] = {
                "chunk_ids": allowed_chunks,
                "image_ids": allowed_images,
                "step_number": step.get("step_number"),
                "problem_id": pid,
                "document_id": did
            }

    # Validate grounding problem IDs
    grounding_data = llm_output.get("grounding", {})
    if isinstance(grounding_data, dict):
        cited_problems = grounding_data.get("source_problem_ids", [])
        valid_cited = [p for p in cited_problems if p in valid_problem_ids]
        grounding_data["source_problem_ids"] = valid_cited

    # Validate LLM steps
    raw_steps = llm_output.get("steps", [])
    validated_steps = []
    seen_step_ids = set()
    last_step_num = 0

    for step in raw_steps:
        if not isinstance(step, dict):
            continue

        step_id = step.get("step_id")
        instruction = step.get("instruction")
        source = step.get("source", {})
        step_number = step.get("step_number")

        # 1. Reject steps missing mandatory fields
        if not step_id or not instruction or not isinstance(source, dict) or step_number is None:
            validation_errors.append("Rejected step missing mandatory fields (step_id, instruction, source, step_number)")
            continue

        # 2. Reject duplicate step_ids
        if step_id in seen_step_ids:
            validation_errors.append(f"Duplicate step_id rejected: {step_id}")
            continue

        # 3. Reject hallucinated step_ids
        if step_id not in valid_step_relations:
            validation_errors.append(f"Hallucinated step_id rejected: {step_id}")
            continue

        relations = valid_step_relations[step_id]

        # 4. Validate chunk_ids in source against the specific step's allowed chunks
        chunk_ids = source.get("chunk_ids", [])
        if not chunk_ids or not isinstance(chunk_ids, list):
            validation_errors.append(f"Rejected step {step_id}: Missing chunk_ids in source")
            continue

        validated_chunks = []
        for cid in chunk_ids:
            if cid in relations["chunk_ids"]:
                validated_chunks.append(cid)
            else:
                validation_errors.append(f"Relational hallucination: chunk_id {cid} does not belong to step {step_id}")
        
        if not validated_chunks:
            validation_errors.append(f"Rejected step {step_id}: No valid chunk_ids remained after relational check")
            continue
             
        source["chunk_ids"] = validated_chunks

        # 5. Validate and re-hydrate images authoritatively against the specific step
        validated_images = []
        raw_images = step.get("images", [])
        if isinstance(raw_images, list):
            for img in raw_images:
                if not isinstance(img, dict):
                    continue
                img_id = img.get("image_id")

                if not img_id:
                    continue

                if img_id not in relations["image_ids"]:
                    validation_errors.append(f"Relational hallucination: image_id {img_id} does not belong to step {step_id}")
                    continue
                
                # Authoritative mapping: Discard LLM's payload, use the real image data
                auth_img = dict(authoritative_images[img_id])
                validated_images.append(auth_img)

        step["images"] = validated_images

        # 6. Verify step numbering
        expected_step_num = relations.get("step_number")
        if expected_step_num is not None and step_number != expected_step_num:
            # Realign with authoritative step number from evidence
            step["step_number"] = expected_step_num

        seen_step_ids.add(step_id)
        validated_steps.append(step)

    # Sort steps by ascending step_number
    validated_steps.sort(key=lambda s: s.get("step_number") or 999)

    llm_output["steps"] = validated_steps
    llm_output["model_known"] = model_known
    llm_output["guidance_scope"] = "model_specific" if model_known else "generic"

    if validation_errors:
        llm_output["_validation_errors"] = validation_errors

    return llm_output


def generate_grounded_guide(query: str, retrieval_context: Dict,
                            model: str, mode: str = "CLOUD") -> Dict:
    """
    Generate a grounded troubleshooting guide using Groq.

    Args:
        query: User query
        retrieval_context: Stage 8 structured evidence
        model: Resolved appliance model
        mode: "CLOUD" for Groq

    Returns:
        Validated dictionary with guide steps and provenance
    """
    if mode != "CLOUD":
        return {
            "status": "error",
            "message": f"{mode} mode is not configured. Use CLOUD mode with Groq."
        }

    client = _get_groq_client()

    # Read model_context metadata
    model_ctx = retrieval_context.get("model_context", {})
    model_known = model_ctx.get("model_known", True)

    system_prompt = _build_system_prompt(model_known=model_known)
    user_prompt = _build_user_prompt(query, retrieval_context, model, model_known=model_known)

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=GROQ_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
            )

            raw_content = completion.choices[0].message.content
            parsed = _parse_llm_response(raw_content)

            if parsed is None:
                return {
                    "status": "error",
                    "message": "The language model returned an invalid response."
                }

            # Validate schema
            if not isinstance(parsed, dict):
                return {
                    "status": "error",
                    "message": "The language model returned an invalid response structure."
                }

            if "steps" not in parsed or not isinstance(parsed.get("steps"), list):
                return {
                    "status": "error",
                    "message": "The language model response is missing required 'steps' field."
                }

            # Validate grounding
            validated = _validate_grounding(parsed, retrieval_context, model_known=model_known)

            return validated

        except Exception as e:
            last_error = e
            error_str = str(e)

            # Check for rate limit (429)
            if "429" in error_str or "rate" in error_str.lower():
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** (attempt + 1)
                    print(f"    [Groq Rate Limit] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            # Check for transient server errors
            if any(code in error_str for code in ["500", "502", "503", "504"]):
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** (attempt + 1)
                    print(f"    [Groq Server Error] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            # Non-retryable error
            break

    if last_error:
        print(f"    [LLM Generator] Internal error: {last_error}")

    return {
        "status": "error",
        "message": "Guide generation service is temporarily unavailable."
    }
