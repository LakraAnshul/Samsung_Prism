import json
import os
import sys

from dotenv import load_dotenv

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

# LINUX FIX: Ensure compatible SQLite for ChromaDB
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

# Local LLM Config
SERVER_IP = os.getenv("OLLAMA_SERVER_IP", "10.212.139.210")
SERVER_PORT = os.getenv("OLLAMA_PORT", "11434")
LOCAL_MODEL = "phi3.5"  # Ensure you have pulled this model: `ollama pull phi3.5`

# Cloud LLM Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUD_MODEL = "llama-3.3-70b-versatile"


# --- DATABASE CONNECTION ---
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)

    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    # k=5 is a good balance for both local and cloud contexts
    return vector_db.as_retriever(search_kwargs={"k": 5})


# --- IMAGE CONTEXT LOADER ---
def load_image_context():
    if not os.path.exists(IMAGE_DB_PATH):
        return "NO IMAGES AVAILABLE."

    try:
        with open(IMAGE_DB_PATH, "r") as f:
            image_data = json.load(f)

        context_str = ""
        for i, img in enumerate(image_data):
            context_str += f"[Image {i + 1}]\n"
            context_str += f"Path: {img.get('file_path')}\n"
            context_str += f"Description: {img.get('dense_caption')}\n"
            context_str += f"Objects: {', '.join(img.get('detected_objects', []))}\n\n"
        return context_str
    except Exception as e:
        return f"Error loading images: {str(e)}"


# --- MAIN GENERATION FUNCTION ---
def generate_guide_from_rag(query, mode="CLOUD"):
    """
    Generates a guide using RAG.
    :param query: User's question
    :param mode: "CLOUD" (Groq) or "LOCAL" (Ollama)
    """
    print(f"\n--- 🔍 Step 1: Searching Knowledge Base for: '{query}' ---")

    # 1. Retrieve Text Context
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return {"error": "No relevant info found in manuals."}

    # Format Text Context
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        chunk_id = i + 1
        filename = doc.metadata.get("filename", "Unknown")
        content = doc.page_content.replace("\n", " ")
        context_parts.append(f"[Chunk {chunk_id}] [Source: {filename}]\n{content}")

    text_context = "\n\n".join(context_parts)

    # 2. Retrieve Image Context
    print("--- 🖼️  Loading Image Knowledge Base... ---")
    image_context = load_image_context()

    # 3. Construct Prompt
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

    # 4. Send to Selected LLM
    print(f"--- ⚡ Step 3: Sending to LLM (Mode: {mode}) ---")

    try:
        if mode == "LOCAL":
            # --- LOCAL OLLAMA PATH ---
            print(f"    📡 Connecting to Ollama at {SERVER_IP}...")
            llm = ChatOllama(
                base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
                model=LOCAL_MODEL,
                temperature=0.0,
                format="json",
            )
            response = llm.invoke(prompt)
            content = response.content

        else:
            # --- CLOUD GROQ PATH ---
            print(f"    ☁️ Connecting to Groq Cloud ({CLOUD_MODEL})...")
            if not GROQ_API_KEY:
                return {"status": "error", "message": "Missing GROQ_API_KEY in .env"}

            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=CLOUD_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content

        # 5. Parse Response
        return json.loads(content)

    except Exception as e:
        print(f"❌ LLM ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Simple test if run directly
    print(
        json.dumps(
            generate_guide_from_rag("How do I clean the filter?", mode="LOCAL"),
            indent=2,
        )
    )
