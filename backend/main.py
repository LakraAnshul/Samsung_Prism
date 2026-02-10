import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# --- IMPORTS & SETUP ---
try:
    from groq import Groq
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library -> {e}")
    sys.exit(1)

# LINUX FIX
if sys.platform.startswith("linux"):
    try:
        __import__("pysqlite3")
        import sys

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_store"
IMAGE_DB_PATH = "./image_knowledge_base.json"

SERVER_IP = os.getenv("OLLAMA_SERVER_IP", "10.212.139.210")
SERVER_PORT = os.getenv("OLLAMA_PORT", "11434")
LOCAL_MODEL = "mistral-8k:latest"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUD_MODEL = "llama-3.3-70b-versatile"

# --- 1. INITIALIZE EMBEDDING MODEL (GLOBAL) ---
print("--- 🧠 Loading Embedding Model... ---")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 2. PRE-LOAD IMAGE DATABASE ---
print("--- ⚙️ Pre-loading Image Database for Semantic Search... ---")
IMAGE_KB = []
IMAGE_EMBEDDINGS = []

if os.path.exists(IMAGE_DB_PATH):
    try:
        with open(IMAGE_DB_PATH, "r") as f:
            IMAGE_KB = json.load(f)

        print(f"   📸 Found {len(IMAGE_KB)} images. Generating embeddings...")

        combo_captions = [
            f"{img.get('problem_name', '')} {img.get('dense_caption', '')} {', '.join(img.get('detected_objects', []))}"
            for img in IMAGE_KB
        ]

        if combo_captions:
            IMAGE_EMBEDDINGS = embedding_model.embed_documents(combo_captions)
            print("   ✅ Image Embeddings Ready.")
        else:
            print("   ⚠️ Image Database appears empty.")

    except Exception as e:
        print(f"   ❌ Error loading Image DB: {e}")
else:
    print("   ⚠️ No image_knowledge_base.json found. Image features disabled.")


# --- DATABASE CONNECTION ---
def get_retriever(k_value=10):
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)

    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_db.as_retriever(search_kwargs={"k": k_value})


# --- LLM CALLER HELPER ---
def call_llm(prompt, mode="CLOUD", json_mode=True):
    try:
        if mode == "LOCAL":
            print(f"    📡 Connecting to Ollama ({LOCAL_MODEL})...")
            fmt = "json" if json_mode else None
            llm = ChatOllama(
                base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
                model=LOCAL_MODEL,
                temperature=0.1,
                format=fmt,
            )
            response = llm.invoke(prompt)
            # print(response.content)
            return response.content
        else:
            print(f"    ☁️ Connecting to Groq Cloud ({CLOUD_MODEL})...")
            if not GROQ_API_KEY:
                raise ValueError("Missing GROQ_API_KEY in .env")
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


# --- PHASE 1: GENERATE STEPS ---
def phase_1_generate_steps(query, text_context, mode):
    print(f"--- 📝 Phase 1: Generating Steps from Text... ---")

    prompt = f"""
    You are a technical instruction extractor.
    CONTEXT: {text_context}
    QUERY: "{query}"

    TASK: Extract steps.
    OUTPUT FORMAT (JSON ONLY):
    {{
      "task_title": "string",
      "steps": [ {{ "step": 1, "instruction": "string", "chunks": [1] }} ]
    }}
    """

    raw = call_llm(prompt, mode, json_mode=True)
    if not raw:
        return None

    try:
        return json.loads(raw)
    except:
        print(f"❌ Phase 1 Parsing Failed: {raw}")
        return None


# --- PHASE 2: MATCH IMAGES (SEMANTIC SEARCH TOP-3) ---
def phase_2_semantic_match(steps_json):
    print(f"--- 👁️  Phase 2: Semantic Image Matching (Top 3)... ---")

    if not IMAGE_KB or not IMAGE_EMBEDDINGS:
        print("   ⚠️ Skipping Phase 2: No Image Database loaded.")
        return steps_json

    steps = steps_json.get("steps", [])
    task_title = steps_json.get("task_title", "")

    for step in steps:
        instruction = step.get("instruction", "")

        # 1. Create a Search Vector
        query_text = f"{task_title} {instruction}"
        query_vec = embedding_model.embed_query(query_text)

        # 2. Calculate Cosine Similarity
        scores = cosine_similarity([query_vec], IMAGE_EMBEDDINGS)[0]

        # 3. Find Top 3 Matches
        # Sort descending and take top 3 indices
        top_indices = np.argsort(scores)[::-1][:3]

        matched_paths = []

        for idx in top_indices:
            score = scores[idx]
            # Threshold: Only accept reasonable matches
            if score > 0.35:
                img_path = IMAGE_KB[idx]["file_path"]
                matched_paths.append(img_path)
                print(
                    f"   ✅ Step {step['step']} -> Match ({score:.2f}): {img_path.split('/')[-1]}"
                )

        # 4. Assign list (or null)
        step["images"] = matched_paths if matched_paths else None

    return steps_json


# --- MAIN ORCHESTRATOR ---
def generate_guide_from_rag(query, mode="CLOUD"):
    print(f"\n--- 🚀 Starting RAG Pipeline (Mode: {mode}) ---")

    # 1. Retrieve Text Context
    retriever = get_retriever(k_value=5)
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return {"status": "error", "message": "No info found."}

    context_parts = [
        f"[Chunk {i + 1}] {d.page_content.replace(chr(10), ' ')}"
        for i, d in enumerate(relevant_docs)
    ]
    text_context = "\n\n".join(context_parts)

    # 2. RUN PHASE 1 (Generate Text Steps)
    step_data = phase_1_generate_steps(query, text_context, mode)
    if not step_data:
        return {"status": "error", "message": "Step generation failed."}

    # 3. RUN PHASE 2 (Semantic Image Match)
    final_data = phase_2_semantic_match(step_data)

    final_data["status"] = "success"

    print("--- ✅ Pipeline Complete ---")
    return final_data


if __name__ == "__main__":
    # Test
    print(
        json.dumps(
            generate_guide_from_rag("How do I clean the filter?", mode="LOCAL"),
            indent=2,
        )
    )
