import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION (REMOTE/LOCAL SERVER) ---
SERVER_IP = os.getenv("OLLAMA_SERVER_IP", "10.212.139.210")
SERVER_PORT = os.getenv("OLLAMA_PORT", "11434")
LOCAL_MODEL = "phi3.5"

# File Paths
DB_PATH = "./chroma_db_store"
IMAGE_DB_PATH = "./image_knowledge_base.json"

# --- 2. SETUP & IMPORTS ---
try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library -> {e}")
    sys.exit(1)

# LINUX FIX: Ensure compatible SQLite for ChromaDB (If needed)
if sys.platform.startswith("linux"):
    try:
        __import__("pysqlite3")
        import sys

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass


# --- 3. DATABASE CONNECTION ---
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)

    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    return vector_db.as_retriever(search_kwargs={"k": 5})


# --- HELPER: LOAD IMAGE DATA ---
def load_image_context():
    """Reads the JSON file containing image captions and paths."""
    if not os.path.exists(IMAGE_DB_PATH):
        print(
            f"⚠️ Warning: Image Knowledge Base '{IMAGE_DB_PATH}' not found. Skipping images."
        )
        return "NO IMAGES AVAILABLE."

    try:
        with open(IMAGE_DB_PATH, "r") as f:
            image_data = json.load(f)

        # Format the data into a string the LLM can read easily
        context_str = ""
        for i, img in enumerate(image_data):
            context_str += f"[Image {i + 1}]\n"
            context_str += f"Path: {img.get('file_path')}\n"
            context_str += f"Description: {img.get('dense_caption')}\n"
            context_str += f"Objects: {', '.join(img.get('detected_objects', []))}\n\n"

        return context_str
    except Exception as e:
        return f"Error loading images: {str(e)}"


# --- 4. GENERATION PIPELINE ---
def generate_guide_from_rag(query):
    print(f"\n--- 🔍 Step 1: Searching Knowledge Base for: '{query}' ---")
    retriever = get_retriever()

    # Retrieve docs
    relevant_docs = retriever.invoke(query)

    # --- DEBUGGING: FULL PRINT ---
    print(f"--- 🧐 DEBUG: Retrieved {len(relevant_docs)} chunks from DB ---")

    context_parts = []
    for i, doc in enumerate(relevant_docs):
        chunk_id = i + 1
        filename = doc.metadata.get("filename", "Unknown")
        content = doc.page_content.replace("\n", " ")

        print(f"\n[Chunk {chunk_id} | Source: {filename}]")
        print(f"{'-' * 20}")
        print(content[:200] + "...")
        print(f"{'-' * 20}")

        context_parts.append(f"[Chunk {chunk_id}] [Source: {filename}]\n{content}")

    if not relevant_docs:
        return {"error": "No relevant info found in manuals."}

    # Join text context
    text_context = "\n\n".join(context_parts)

    # LOAD IMAGE CONTEXT
    print("--- 🖼️  Loading Image Knowledge Base... ---")
    image_context = load_image_context()

    # --- CONNECT TO LOCAL LLM ---
    print(f"--- 📡 Connecting to Local LLM at {SERVER_IP}... ---")
    try:
        llm = ChatOllama(
            base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
            model=LOCAL_MODEL,
            temperature=0.0,  # Zero temp for strictness
            format="json",  # FORCE JSON OUTPUT
        )
    except Exception as e:
        return {"error": f"Failed to connect to Ollama: {e}"}

    # --- STRICT PROMPT ENGINEERING ---
    prompt = f"""
    You are a strict technical assistant. You have a Text Knowledge Base and an Image Knowledge Base.

    TEXT KNOWLEDGE BASE (Use for instructions):
    {text_context}

    IMAGE KNOWLEDGE BASE (Use for image selection):
    {image_context}

    USER REQUEST:
    "{query}"

    RULES:
    1. Answer ONLY using the information in the TEXT KNOWLEDGE BASE.
    2. Do NOT use your own outside knowledge.
    3. For every step, list the integer ID(s) of the Chunk(s) used in "chunks".

    4. IMAGE SELECTION RULE:
       - Look at the "IMAGE KNOWLEDGE BASE" section.
       - Compare the step instruction with the Image "Description".
       - If you find a highly relevant image, copy its "Path" EXACTLY into the "images" field.
       - If NO image matches the step, leave "images" as "null".

    OUTPUT FORMAT (Strict JSON):
    IF ANSWER FOUND:
    {{
      "status": "success",
      "task_title": "Title",
      "steps": [
        {{
            "step": 1,
            "instruction": "Detailed action to take",
            "images": "./final_cleaned_dataset/example_image.png",
            "chunks": [1, 3]
        }}
      ]
    }}

    IF ANSWER NOT FOUND IN CONTEXT:
    {{
      "status": "error",
      "message": "This query is outside the scope of the provided manuals."
    }}
    """

    print(f"--- ⚡ Step 3: Sending to {LOCAL_MODEL}... ---")
    try:
        # LangChain invoke
        response = llm.invoke(prompt)

        # 🔴 DEBUG: DUMP RAW RESPONSE
        print("\n" + "=" * 40)
        print("🐛 RAW LLM RESPONSE DUMP:")
        print("=" * 40)
        print(response.content)
        print("=" * 40 + "\n")

        # Parse content
        return json.loads(response.content)
    except Exception as e:
        print(f"❌ ERROR PARSING JSON: {e}")
        return {
            "error": f"Local Inference Error: {str(e)}",
            "raw_response": getattr(response, "content", "No Response"),
        }


if __name__ == "__main__":
    # Test 1: Valid Query
    q1 = "How do I clean the mesh filter?"
    print(f"\n👉 TESTING VALID QUERY: {q1}")
    result1 = generate_guide_from_rag(q1)
    print(json.dumps(result1, indent=2))
