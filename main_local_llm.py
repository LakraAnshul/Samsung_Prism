import os
import sys
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION (REMOTE SERVER) ---
SERVER_IP = "10.183.45.210"  
SERVER_PORT = "11434"
LOCAL_MODEL = "mistral-8k:latest" 
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
print("--- ⚙️ Pre-loading Image Database... ---")
IMAGE_KB = []
IMAGE_EMBEDDINGS = []

if os.path.exists(IMAGE_DB_PATH):
    with open(IMAGE_DB_PATH, 'r') as f:
        IMAGE_KB = json.load(f)
    
    print(f"   📸 Embedding {len(IMAGE_KB)} images (Contextualized)...")
    # Combine Problem Name + Caption for better accuracy
    combo_captions = [f"{img.get('problem_name', '')} {img.get('dense_caption', '')}" for img in IMAGE_KB]
    
    if combo_captions:
        IMAGE_EMBEDDINGS = embedding_model.embed_documents(combo_captions)
        print("   ✅ Image Database Ready.")
    else:
        print("   ⚠️ Image Database is empty.")
else:
    print("   ⚠️ No image_knowledge_base.json found. Image features disabled.")

def find_best_images(task_title, step_description, top_k=3):
    """
    Finds the top 3 images matching Task + Step Description.
    """
    if not IMAGE_KB or not IMAGE_EMBEDDINGS:
        return []

    search_query = f"{task_title} {step_description}"
    query_vec = embedding_model.embed_query(search_query)
    
    scores = cosine_similarity([query_vec], IMAGE_EMBEDDINGS)[0]
    
    # Get top_k indices sorted descending
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = scores[idx]
        if score > 0.35: # Relevance Threshold
            results.append({
                "path": IMAGE_KB[idx]['file_path'],
                "score": float(score)
            })
            
    return results

# --- 4. DATABASE CONNECTION ---
def get_retriever():
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_db.as_retriever(search_kwargs={"k": 7})

# --- 5. GENERATION PIPELINE ---
def generate_guide_from_rag(query):
    print(f"\n--- 🔍 Step 1: Searching Knowledge Base for: '{query}' ---")
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return {"error": "No relevant info found in manuals."}
    
    context_text = "\n\n".join([f"Source: {doc.metadata.get('filename')} Content: {doc.page_content}" for doc in relevant_docs])
    
    print(f"--- 📡 Connecting to Remote Server at {SERVER_IP}... ---")
    llm = ChatOllama(
        base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
        model=LOCAL_MODEL,
        temperature=0.0,
        format="json"
    )
    
    prompt = f"""
    You are a technical assistant.
    CONTEXT:
    {context_text}
    
    USER REQUEST: "{query}"
    
    TASK:
    Create a step-by-step guide based ONLY on the context.
    For each step, write a 'visual_description' describing specific visual elements (e.g. "Close up of blue filter cap").
    
    OUTPUT JSON:
    {{
      "status": "success",
      "task_title": "Short Task Name",
      "steps": [
        {{ 
            "step": 1, 
            "instruction": "Detailed text instruction", 
            "visual_description": "Short visual description" 
        }}
      ]
    }}
    """
    
    print(f"--- ⚡ Step 3: Sending to {LOCAL_MODEL} (Remote) ---")
    try:
        response = llm.invoke(prompt)
        result_json = json.loads(response.content)
        
        print("--- 🖼️  Finding Matching Images (Top 3)... ---")
        if "steps" in result_json:
            task_title = result_json.get("task_title", "General")
            
            for step in result_json['steps']:
                visual_desc = step.get('visual_description', step['instruction'])
                
                matched_images = find_best_images(task_title, visual_desc, top_k=3)
                step['images'] = [match['path'] for match in matched_images]
                
                if matched_images:
                    print(f"   ✅ Step {step['step']}: Found {len(matched_images)} images.")
                else:
                    print(f"   ⚠️ Step {step['step']}: No matching images.")
        
        return result_json
        
    except Exception as e:
        return {"error": f"Remote Inference Error: {str(e)}"}

if __name__ == "__main__":
    # 1. Define Query
    q1 = "How to clean Samsung Washing machine"
    print(f"\n👉 TESTING VALID QUERY: {q1}")
    
    # 2. Run Pipeline
    result1 = generate_guide_from_rag(q1)
    
    # 3. Print to Console
    formatted_json = json.dumps(result1, indent=2)
    print(formatted_json)
    
    # 4. SAVE TO FILE (New Logic)
    output_filename = "guide.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        print(f"\n--- 💾 SUCCESS: Output saved to '{output_filename}' ---")
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")