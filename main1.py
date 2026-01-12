import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- 2. SETUP & IMPORTS ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from groq import Groq
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library -> {e}")
    sys.exit(1)

# Logic: Use the direct key
api_key = os.getenv("GROQ_API_KEY")

# CONFIGURATION
DB_PATH = "./chroma_db_store"

# --- 3. DATABASE CONNECTION ---
# --- 3. DATABASE CONNECTION ---
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)
        
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # CHANGE 1: Increased k from 3 to 7
    # This means it will fetch the top 7 pages relevant to your query
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
    
    # Initialize Groq
    client = Groq(api_key=api_key)
    
    # --- STRICT PROMPT ENGINEERING ---
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
    
    print("--- ⚡ Step 3: Sending to Llama 3.3 (Groq) ---")
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, # ZERO Temperature = Maximum strictness
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": f"Groq API Error: {str(e)}"}

if __name__ == "__main__":
    # Test 1: Valid Query
    q1 = "My Washing Machine is not Spinning properly"
    print(f"\n👉 TESTING VALID QUERY: {q1}")
    result1 = generate_guide_from_rag(q1)
    print(json.dumps(result1, indent=2))
