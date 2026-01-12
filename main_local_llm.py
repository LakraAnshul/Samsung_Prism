import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION (REMOTE SERVER) ---
# 🔴 IP of your RTX 3050 Laptop (From Hotspot step)
SERVER_IP = "10.159.195.210"  
SERVER_PORT = "11434"
# 🔴 Must match the model you pulled on the Server (e.g., phi3.5)
LOCAL_MODEL = "phi3.5" 

# --- 2. SETUP & IMPORTS ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    # NEW IMPORT: For connecting to remote Ollama
    from langchain_ollama import ChatOllama
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library -> {e}")
    sys.exit(1)

# CONFIGURATION
DB_PATH = "./chroma_db_store"

# --- 3. DATABASE CONNECTION ---
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)
        
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # CHANGE 1: Increased k from 3 to 7
    return vector_db.as_retriever(search_kwargs={"k": 7})

# --- 4. GENERATION PIPELINE ---
def generate_guide_from_rag(query):
    print(f"\n--- 🔍 Step 1: Searching Knowledge Base for: '{query}' ---")
    retriever = get_retriever()
    
    # Retrieve docs
    relevant_docs = retriever.invoke(query)
    
    # --- DEBUGGING: FULL PRINT ---
    print(f"--- 🧐 DEBUG: Retrieved {len(relevant_docs)} chunks from DB ---")
    for i, doc in enumerate(relevant_docs):
        filename = doc.metadata.get('filename', 'Unknown')
        
        # CHANGE 2: Removed [:200]. Now printing the FULL text content.
        print(f"\n[Chunk {i+1} | Source: {filename}]")
        print(f"{'-'*20}")
        print(doc.page_content) 
        print(f"{'-'*20}")
    # -----------------------------------------------

    if not relevant_docs:
        return {"error": "No relevant info found in manuals."}
    
    # Combine text for the LLM
    context_text = "\n\n".join([f"Source: {doc.metadata.get('filename')} Content: {doc.page_content}" for doc in relevant_docs])
    
    # --- CHANGED: Initialize Remote LLM Connection (Was Groq) ---
    print(f"--- 📡 Connecting to Remote Server at {SERVER_IP}... ---")
    llm = ChatOllama(
        base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
        model=LOCAL_MODEL,
        temperature=0.0, # ZERO Temperature = Maximum strictness
        format="json"    # Force JSON mode
    )
    
    # --- STRICT PROMPT ENGINEERING (KEPT EXACTLY AS IS) ---
    prompt = f"""
    You are a strict technical assistant. You have a Knowledge Base of Samsung Washing Machine Manuals.
    
    CONTEXT (Retrieved from Database):
    {context_text}
    
    USER REQUEST:
    "{query}"
    
    RULES:
    1. Answer ONLY using the information in the CONTEXT above.
    2. If the user asks about a topic NOT present in the CONTEXT (like iPhones, cooking, weather), return a JSON error.
    3. Do NOT use your own outside knowledge.
    
    OUTPUT FORMAT (Strict JSON):
    IF ANSWER FOUND:
    {{
      "status": "success",
      "task_title": "Title",
      "steps": [
        {{ "step": 1, "instruction": "Action", "visual_description": "Image description" }}
      ]
    }}
    
    IF ANSWER NOT FOUND IN CONTEXT:
    {{
      "status": "error",
      "message": "This query is outside the scope of the provided manuals."
    }}
    """
    
    print(f"--- ⚡ Step 3: Sending to {LOCAL_MODEL} (Remote) ---")
    try:
        # CHANGED: Using LangChain invoke instead of Groq completion
        response = llm.invoke(prompt)
        
        # Parse the content string into JSON
        return json.loads(response.content)
        
    except Exception as e:
        return {"error": f"Remote Inference Error: {str(e)}"}

if __name__ == "__main__":
    # Test 1: Valid Query
    q1 = "My Washing Machine is not Spinning properly"
    print(f"\n👉 TESTING VALID QUERY: {q1}")
    result1 = generate_guide_from_rag(q1)
    print(json.dumps(result1, indent=2))