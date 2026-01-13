import os
import sys
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION (REMOTE SERVER) ---
SERVER_IP = "10.212.139.210"  
SERVER_PORT = "11434"
LOCAL_MODEL = "phi3.5" 
IMAGE_DB_PATH = "image_knowledge_base.json" 

load_dotenv()

# --- 2. SETUP & IMPORTS ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import ChatOllama
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library -> {e}")
    sys.exit(1)

# CONFIGURATION
DB_PATH = "./chroma_db_store"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 3. IMAGE SEARCH ENGINE ---
# (Kept exactly the same as before for stability)
print("--- ⚙️ Pre-loading Image Database... ---")
IMAGE_KB = []
IMAGE_EMBEDDINGS = []
if os.path.exists(IMAGE_DB_PATH):
    with open(IMAGE_DB_PATH, 'r') as f:
        IMAGE_KB = json.load(f)
    print(f"   📸 Embedding {len(IMAGE_KB)} images...")
    combo_captions = [f"{img.get('problem_name', '')} {img.get('dense_caption', '')}" for img in IMAGE_KB]
    if combo_captions:
        IMAGE_EMBEDDINGS = embedding_model.embed_documents(combo_captions)
        print("   ✅ Image Database Ready.")
else:
    print("   ⚠️ No image DB found.")

def find_best_images(task_title, step_description, top_k=3):
    if not IMAGE_KB or not IMAGE_EMBEDDINGS: return []
    search_query = f"{task_title} {step_description}"
    query_vec = embedding_model.embed_query(search_query)
    scores = cosine_similarity([query_vec], IMAGE_EMBEDDINGS)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0.35:
            results.append({"path": IMAGE_KB[idx]['file_path'], "score": float(scores[idx])})
    return results

# --- 4. DATABASE CONNECTION ---
def get_retriever():
    if not os.path.exists(DB_PATH):
        sys.exit(1)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_db.as_retriever(search_kwargs={"k": 7})

# --- 5. GENERATION PIPELINE ---
def generate_guide_from_rag(query):
    print(f"\n--- 🔍 Step 1: Searching Knowledge Base for: '{query}' ---")
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return {"error": "No relevant info found."}, ""
    
    # 🔴 CHANGE A: Build "Labeled" Context for the LLM
    # We assign an ID to each chunk (Chunk 0, Chunk 1...)
    context_list = []
    formatted_context_for_file = ""
    
    for i, doc in enumerate(relevant_docs):
        chunk_text = doc.page_content.replace("\n", " ")
        source = doc.metadata.get('filename', 'Unknown')
        
        # Format for LLM Prompt
        context_list.append(f"[Chunk {i}] (Source: {source}): {chunk_text}")
        
        # Format for Output File
        formatted_context_for_file += f"--- [Chunk {i}] ---\nSource: {source}\nContent: {chunk_text}\n\n"

    final_context_block = "\n\n".join(context_list)
    
    print(f"--- 📡 Connecting to Remote Server at {SERVER_IP}... ---")
    llm = ChatOllama(
        base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
        model=LOCAL_MODEL,
        temperature=0.0, # Strict
        format="json"
    )
    
    # 🔴 CHANGE B: The Strict "Anti-Hallucination" Prompt
    prompt = f"""
    You are a Strict Technical Extraction Engine.
    You are forbidden from creating steps that do not exist in the text chunks below.

    TEXT KNOWLEDGE BASE (Use ONLY this):
    {final_context_block}
    
    USER REQUEST: "{query}"
    
    STRICT RULES:
    1. SOURCE OF TRUTH: Answer ONLY using the provided Text Chunks.
    2. CITATION: For every step, you MUST list the integer ID(s) of the chunk(s) used (e.g., "chunk_ids": [0, 2]).
    3. NO OUTSIDE KNOWLEDGE: If the text doesn't say it, do not write it.
    4. VISUALS: Write a 'visual_description' for each step based on the text context (e.g., "Hand turning blue cap").

    OUTPUT FORMAT (JSON):
    {{
      "task_title": "Title based on text",
      "steps": [
        {{ 
            "step": 1, 
            "instruction": "Action text...", 
            "chunk_ids": [0], 
            "visual_description": "Visual details..." 
        }}
      ]
    }}
    """
    
    print(f"--- ⚡ Step 3: Sending to {LOCAL_MODEL} (Remote) ---")
    try:
        response = llm.invoke(prompt)
        result_json = json.loads(response.content)
        
        # Post-Process: Find Images
        if "steps" in result_json:
            task_title = result_json.get("task_title", "General")
            for step in result_json['steps']:
                visual_desc = step.get('visual_description', step['instruction'])
                matched_images = find_best_images(task_title, visual_desc, top_k=3)
                step['images'] = [match['path'] for match in matched_images]
        
        return result_json, formatted_context_for_file
        
    except Exception as e:
        return {"error": f"Remote Inference Error: {str(e)}"}, formatted_context_for_file

if __name__ == "__main__":
    q1 = "How to clean the debris filter?"
    print(f"\n👉 TESTING QUERY: {q1}")
    
    # Run Pipeline
    final_json, raw_context = generate_guide_from_rag(q1)
    
    # 🔴 CHANGE C: Save Audit Trail to 'guide.txt'
    output_filename = "guide.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write(f" QUERY: {q1}\n")
            f.write("==================================================\n\n")
            
            f.write("**************************************************\n")
            f.write(" PART 1: RETRIEVED TEXT CHUNKS (The Ground Truth)\n")
            f.write("**************************************************\n\n")
            f.write(raw_context)
            
            f.write("\n**************************************************\n")
            f.write(" PART 2: GENERATED GUIDE (JSON with Citations)\n")
            f.write("**************************************************\n\n")
            f.write(json.dumps(final_json, indent=2))
            
        print(f"\n--- 💾 SUCCESS: Full Audit Report saved to '{output_filename}' ---")
        print(json.dumps(final_json, indent=2))
        
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")