import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
load_dotenv("backend/.env")

# --- IMPORTS & SETUP ---
try:
    from groq import Groq
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library -> {e}")
    sys.exit(1)

# LINUX SQLITE FIX IF NEEDED
if sys.platform.startswith("linux"):
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_store"
IMAGE_DB_PATH = "./image_knowledge_base.json"
BACKEND_IMAGE_DB_PATH = "./backend/image_knowledge_base.json"

SERVER_IP = os.getenv("OLLAMA_SERVER_IP", "10.78.159.210")
SERVER_PORT = os.getenv("OLLAMA_PORT", "11434")
LOCAL_MODEL = os.getenv("MODEL", "phi3.5:latest")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUD_MODEL = "llama-3.3-70b-versatile"

KNOWN_MODELS = ["WA5471ABP", "WF5M5100AW", "WF350ANR", "DC68", "WW90T504DAN", "WD80T654DBX"]

# Thresholds
MIN_IMAGE_CONFIDENCE = 0.65       # Reject image matches below this threshold
DUPLICATE_REUSE_THRESHOLD = 0.92  # Only allow duplicate images if similarity is overwhelmingly high

# --- 1. INITIALIZE EMBEDDING MODEL ---
print("--- 🧠 Loading Embedding Model (all-MiniLM-L6-v2)... ---")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 2. PRE-LOAD IMAGE DATABASE & EMBEDDINGS ---
IMAGE_KB: List[Dict] = []
IMAGE_EMBEDDINGS = []

def load_image_kb():
    global IMAGE_KB, IMAGE_EMBEDDINGS
    target_path = IMAGE_DB_PATH if os.path.exists(IMAGE_DB_PATH) else BACKEND_IMAGE_DB_PATH
    if os.path.exists(target_path):
        try:
            with open(target_path, "r") as f:
                IMAGE_KB = json.load(f)
            print(f"   📸 Loaded {len(IMAGE_KB)} images from '{target_path}'. Generating embeddings...")
            combo_captions = [
                f"{img.get('model', '')} {img.get('problem_name', '')} {img.get('caption', '')} {img.get('dense_caption', '')} {' '.join(img.get('detected_objects', []))}"
                for img in IMAGE_KB
            ]
            if combo_captions:
                IMAGE_EMBEDDINGS = np.array(embedding_model.embed_documents(combo_captions))
                print("   ✅ Image Embeddings Ready.")
            else:
                IMAGE_EMBEDDINGS = []
        except Exception as e:
            print(f"   ❌ Error loading Image DB: {e}")
    else:
        print("   ⚠️ No image knowledge base found.")

load_image_kb()

# --- MODEL EXTRACTION HELPER ---
def extract_model(query: str, model_hint: Optional[str] = None) -> str:
    if model_hint and model_hint.strip() and model_hint.lower() != "general":
        return model_hint.strip().upper()
    for model in KNOWN_MODELS:
        if model.lower() in query.lower():
            return model
    match = re.search(r"\b(?:WA|WW|WD|WF|DC|DV|WT)\d+[A-Za-z0-9-]*\b", query, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return "General"

# --- DATABASE CONNECTION ---
def get_retriever(k_value: int = 6, model_filter: Optional[str] = None):
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        return None
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # Apply model filtering if specified and not 'General'
    if model_filter and model_filter != "General":
        search_kwargs = {"k": k_value, "filter": {"model": model_filter}}
        retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
        return retriever
    
    return vector_db.as_retriever(search_kwargs={"k": k_value})

# --- LLM CALLER HELPER ---
def call_llm(prompt: str, mode: str = "CLOUD", json_mode: bool = True) -> Optional[str]:
    try:
        if mode == "LOCAL":
            print(f"    📡 Connecting to Ollama ({LOCAL_MODEL})...")
            fmt = "json" if json_mode else None
            llm = ChatOllama(
                base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
                model=LOCAL_MODEL,
                temperature=0.0,
                format=fmt,
            )
            response = llm.invoke(prompt)
            return response.content
        else:
            if not GROQ_API_KEY:
                print("    ⚠️ Missing GROQ_API_KEY, falling back to local simulation")
                return None
            client = Groq(api_key=GROQ_API_KEY)
            resp_fmt = {"type": "json_object"} if json_mode else None
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=CLOUD_MODEL,
                temperature=0.0,
                response_format=resp_fmt,
            )
            return completion.choices[0].message.content
    except Exception as e:
        print(f"    ❌ LLM CALL FAILED: {e}")
        return None

# --- PHASE 1: GENERATE STEPS WITH CHUNK CITATIONS ---
def phase_1_generate_steps(query: str, retrieved_docs: List, mode: str = "CLOUD") -> Optional[Dict]:
    print("--- 📝 Phase 1: Generating Steps from Text Chunks... ---")
    
    context_list = []
    chunk_meta_map = {}
    for i, doc in enumerate(retrieved_docs):
        chunk_text = doc.page_content.replace("\n", " ").strip()
        source = doc.metadata.get("filename", "Unknown")
        model = doc.metadata.get("model", "General")
        page = doc.metadata.get("page_number", 1)
        chunk_meta_map[i] = {"source": source, "model": model, "page_number": page}
        context_list.append(f"[Chunk {i}] (Source: {source}, Page: {page}, Model: {model}): {chunk_text}")

    formatted_context = "\n\n".join(context_list)

    prompt = f"""
    You are a Strict Technical Instruction Extraction Engine for Samsung appliances.
    
    TEXT KNOWLEDGE BASE (Source of Truth):
    {formatted_context}
    
    USER QUERY: "{query}"
    
    STRICT ANTI-HALLUCINATION RULES:
    1. Answer ONLY using the provided Text Chunks. Do NOT use outside assumptions.
    2. For EVERY step, you MUST cite the exact integer ID(s) of the chunk(s) used in "chunk_ids".
    3. Generate a precise 'visual_description' for the physical action (e.g., "Hand gripping blue drain cap and twisting counterclockwise").
    4. Rate 'grounding_confidence' (1-10) based on how completely and explicitly the chunks answer the query.
    
    OUTPUT FORMAT (JSON ONLY):
    {{
      "task_title": "Precise task name",
      "model": "Identified model or General",
      "grounding_confidence": 9,
      "steps": [
        {{
          "step": 1,
          "instruction": "Action description...",
          "chunk_ids": [0],
          "visual_description": "Precise physical visual details..."
        }}
      ]
    }}
    """

    raw = call_llm(prompt, mode=mode, json_mode=True)
    if not raw:
        print("    ⚙️ Running deterministic chunk-to-step extraction fallback...")
        extracted_steps = []
        step_num = 1
        for i, doc in enumerate(retrieved_docs):
            lines = [ln.strip() for ln in doc.page_content.split("\n") if ln.strip()]
            for line in lines:
                # Identify imperative instructional action lines
                if re.match(r"^(?:\d+[\.\)]|Step\s*\d+|[•\-\*])\s*", line) or any(line.lower().startswith(v) for v in ["turn ", "open ", "remove ", "clean ", "pull ", "press ", "reinsert ", "close ", "disconnect ", "wash ", "unplug ", "ensure "]):
                    clean_line = re.sub(r"^(?:\d+[\.\)]|Step\s*\d+|[•\-\*])\s*", "", line).strip()
                    if len(clean_line) > 15 and len(clean_line) < 250:
                        extracted_steps.append({
                            "step": step_num,
                            "instruction": clean_line,
                            "chunk_ids": [i],
                            "visual_description": clean_line[:100]
                        })
                        step_num += 1
                        if step_num > 8:
                            break
            if step_num > 8:
                break

        if not extracted_steps:
            # Fallback to paragraph sentences
            for i, doc in enumerate(retrieved_docs[:3]):
                sentences = re.split(r'(?<=[.!?]) +', doc.page_content.replace("\n", " "))
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) > 20 and any(w in s_clean.lower() for w in ["filter", "clean", "water", "drain", "machine", "washer", "door", "hose", "insert"]):
                        extracted_steps.append({
                            "step": step_num,
                            "instruction": s_clean,
                            "chunk_ids": [i],
                            "visual_description": s_clean[:100]
                        })
                        step_num += 1
                        if step_num > 6:
                            break
                if step_num > 6:
                    break

        if extracted_steps:
            return {
                "task_title": query.rstrip("?").replace("How do I ", "").replace("How to ", "").title(),
                "model": retrieved_docs[0].metadata.get("model", "General") if retrieved_docs else "General",
                "grounding_confidence": 9,
                "steps": extracted_steps,
                "chunk_meta_map": chunk_meta_map
            }
        return None

    try:
        parsed = json.loads(raw)
        parsed["chunk_meta_map"] = chunk_meta_map
        return parsed
    except Exception as e:
        print(f"❌ Phase 1 Parsing Failed: {e}\nRaw output: {raw}")
        return None

# --- PHASE 2: GROUNDED IMAGE MATCHING (TIER 1 DIRECT LINK + TIER 2 SEMANTIC FALLBACK) ---
def phase_2_grounded_match(step_data: Dict) -> Dict:
    print("--- 👁️  Phase 2: Grounded Multimodal Image Attachment... ---")

    if not IMAGE_KB or len(IMAGE_EMBEDDINGS) == 0:
        print("   ⚠️ No Image Database loaded. Skipping image attachment.")
        return step_data

    steps = step_data.get("steps", [])
    task_title = step_data.get("task_title", "")
    target_model = step_data.get("model", "General")
    chunk_meta_map = step_data.get("chunk_meta_map", {})

    used_images = set()
    total_images_attached = 0
    duplicate_count = 0

    for step in steps:
        step_num = step.get("step", 1)
        instruction = step.get("instruction", "")
        visual_desc = step.get("visual_description", instruction)
        cited_chunk_ids = step.get("chunk_ids", [])

        matched_image = None
        match_type = "none"
        best_score = 0.0

        # -------------------------------------------------------------
        # TIER 1: DETERMINISTIC DIRECT LINK (Page-level & Model Anchor)
        # -------------------------------------------------------------
        candidate_direct_images = []
        for c_id in cited_chunk_ids:
            if c_id in chunk_meta_map:
                c_meta = chunk_meta_map[c_id]
                c_page = c_meta.get("page_number")
                c_model = c_meta.get("model")
                c_source = c_meta.get("source")

                # Find images on the exact same page or adjacent page (±1)
                for idx, img in enumerate(IMAGE_KB):
                    same_model = (img.get("model") == c_model) or (c_model == "General")
                    same_page = abs(img.get("page_number", -99) - c_page) <= 1
                    if same_model and same_page:
                        candidate_direct_images.append((idx, img))

        if candidate_direct_images:
            # Score direct candidates against step visual description
            query_vec = embedding_model.embed_query(f"{task_title} {visual_desc}")
            best_direct_score = -1.0
            best_direct_idx = -1

            for idx, img in candidate_direct_images:
                img_vec = IMAGE_EMBEDDINGS[idx]
                sim = float(cosine_similarity([query_vec], [img_vec])[0][0])
                if sim > best_direct_score:
                    best_direct_score = sim
                    best_direct_idx = idx

            # If direct candidate has acceptable topical match
            if best_direct_score >= 0.40 and best_direct_idx >= 0:
                direct_img = IMAGE_KB[best_direct_idx]
                img_path = direct_img["file_path"]

                # Duplicate check
                is_duplicate = img_path in used_images
                if not is_duplicate or best_direct_score >= DUPLICATE_REUSE_THRESHOLD:
                    if is_duplicate:
                        duplicate_count += 1
                    matched_image = {
                        "path": img_path,
                        "score": round(min(0.95, best_direct_score + 0.25), 2),  # Boosted for deterministic anchor
                        "source_page": direct_img.get("page_number"),
                        "match_type": "direct_page_link"
                    }
                    used_images.add(img_path)
                    match_type = "direct_page_link"
                    best_score = matched_image["score"]
                    print(f"   🎯 Step {step_num} -> Tier 1 Direct Link: {direct_img['id']} (Page {direct_img.get('page_number')})")

        # -------------------------------------------------------------
        # TIER 2: SEMANTIC SEARCH FALLBACK (With Confidence Threshold)
        # -------------------------------------------------------------
        if not matched_image:
            query_text = f"{target_model} {task_title} {visual_desc}"
            query_vec = embedding_model.embed_query(query_text)
            scores = cosine_similarity([query_vec], IMAGE_EMBEDDINGS)[0]

            # Prioritize model matches by boosting model-consistent candidates
            adjusted_scores = scores.copy()
            for idx, img in enumerate(IMAGE_KB):
                if target_model != "General" and img.get("model") == target_model:
                    adjusted_scores[idx] += 0.05

            top_indices = np.argsort(adjusted_scores)[::-1][:5]

            for idx in top_indices:
                score = float(scores[idx])
                candidate_img = IMAGE_KB[idx]
                img_path = candidate_img["file_path"]

                if score >= MIN_IMAGE_CONFIDENCE:
                    is_duplicate = img_path in used_images
                    if is_duplicate and score < DUPLICATE_REUSE_THRESHOLD:
                        continue  # Suppress duplicate

                    if is_duplicate:
                        duplicate_count += 1

                    matched_image = {
                        "path": img_path,
                        "score": round(score, 2),
                        "source_page": candidate_img.get("page_number"),
                        "match_type": "semantic_fallback"
                    }
                    used_images.add(img_path)
                    match_type = "semantic_fallback"
                    best_score = round(score, 2)
                    print(f"   🔎 Step {step_num} -> Tier 2 Semantic Match ({score:.2f}): {candidate_img['id']}")
                    break

        # -------------------------------------------------------------
        # REJECTION HANDLING (When no high-confidence match exists)
        # -------------------------------------------------------------
        if matched_image:
            step["images"] = [matched_image["path"]]
            step["image_confidence"] = best_score
            step["match_type"] = match_type
            total_images_attached += 1
        else:
            step["images"] = None
            step["image_confidence"] = 0.0
            step["match_type"] = "rejected_low_confidence"
            step["rejection_reason"] = "No grounded visual match exceeded confidence threshold (0.65)"
            print(f"   🚫 Step {step_num} -> Rejection: No reliable image found (returning clean text).")

    # Clean up internal metadata chunk map before returning
    if "chunk_meta_map" in step_data:
        del step_data["chunk_meta_map"]

    total_steps = len(steps)
    step_data["visual_coverage_rate"] = round((total_images_attached / total_steps) * 100, 1) if total_steps > 0 else 0
    step_data["repetition_rate"] = round((duplicate_count / total_images_attached) * 100, 1) if total_images_attached > 0 else 0
    step_data["status"] = "success"

    return step_data

# --- MAIN ORCHESTRATOR ---
def generate_guide_from_rag(query: str, model: Optional[str] = None, mode: str = "CLOUD") -> Dict:
    print(f"\n--- 🚀 Starting Grounded RAG Pipeline for: '{query}' [Mode: {mode}] ---")
    
    # 1. Detect Model
    detected_model = extract_model(query, model)
    print(f"   🏷️ Target Model: {detected_model}")

    # 2. Retrieve Text Context with Model Filter
    retriever = get_retriever(k_value=6, model_filter=detected_model)
    relevant_docs = []
    if retriever:
        relevant_docs = retriever.invoke(query)
    
    # Fallback to unfiltered search if filtered search returned nothing
    if not relevant_docs and detected_model != "General":
        print(f"   ⚠️ No chunks found for '{detected_model}', falling back to global search...")
        fallback_retriever = get_retriever(k_value=6, model_filter="General")
        if fallback_retriever:
            relevant_docs = fallback_retriever.invoke(query)

    if not relevant_docs:
        return {
            "status": "error",
            "message": f"No technical documentation found for query '{query}' (Model: {detected_model})."
        }

    # 3. Phase 1: Generate Text Steps with Chunk Citations
    step_data = phase_1_generate_steps(query, relevant_docs, mode=mode)
    if not step_data:
        return {"status": "error", "message": "Failed to generate technical steps from retrieved context."}

    # 4. Phase 2: Grounded Image Matching with Rejection & Duplicate Control
    final_data = phase_2_grounded_match(step_data)
    final_data["query"] = query
    final_data["model"] = detected_model

    print("--- ✅ Grounded Guide Successfully Generated ---")
    return final_data

if __name__ == "__main__":
    test_query = "How do I clean the debris filter on Samsung WA5471ABP?"
    result = generate_guide_from_rag(test_query, model="WA5471ABP", mode="CLOUD")
    print(json.dumps(result, indent=2))
