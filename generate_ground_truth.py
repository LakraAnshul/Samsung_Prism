import os
import json
import fitz  # PyMuPDF
from groq import Groq
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BENCHMARK_FILE = "benchmark_data.json"
PDF_FOLDER = "./Knowledge_Base/text" 

def extract_text_from_pdf(pdf_path):
    """Reads all text from the PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def analyze_pdf_content(filename, text_content):
    """Asks Groq to strictly extract the main task and steps."""
    client = Groq(api_key=GROQ_API_KEY)
    
    # Limit text to ~30k chars to stay within context limits
    safe_text = text_content[:30000]

    # --- 🔴 STRICT PROMPT ENGINEERING ---
    prompt = f"""
    You are a Strict Data Extraction Engine.
    
    SOURCE DOCUMENT: "{filename}"
    RAW TEXT CONTENT:
    {safe_text}
    
    TASK:
    Extract the single main procedural task described in this text.
    
    CRITICAL RULES (DO NOT BREAK):
    1. NO OUTSIDE KNOWLEDGE: You must ONLY use the text provided above. Do not use your own knowledge about washing machines.
    2. NO HALLUCINATION: If a step is not explicitly written in the text, DO NOT include it.
    3. PRESERVE WORDING: Maintain the exact technical terminology used in the text (e.g., if it says "Debris Filter", do not change it to "Lint Trap").
    4. ACCURACY: The 'ground_truth' list must be an accurate, ordered sequence of actions found in the text.
    
    OUTPUT FORMAT (JSON):
    {{
      "query": "Write a natural user question for this specific task (e.g. 'How do I clean the mesh filter?')",
      "ground_truth": [
        "Exact text of step 1...",
        "Exact text of step 2...",
        "Exact text of step 3..."
      ]
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, # Zero temp ensures reproducibility and less creativity
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"      ❌ Error parsing {filename}: {e}")
        return None

def main():
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Error: Folder '{PDF_FOLDER}' not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ No PDFs found.")
        return

    print(f"--- 🚀 Starting Strict Batch Extraction for {len(pdf_files)} Files ---")
    
    # Load existing data to append to it
    existing_data = []
    if os.path.exists(BENCHMARK_FILE):
        try:
            with open(BENCHMARK_FILE, "r") as f:
                existing_data = json.load(f)
        except:
            existing_data = []

    # Iterate through all files
    new_entries = []
    for i, pdf_file in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}] Processing: {pdf_file}...")
        
        # 1. Read Text
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        
        # 2. Get Ground Truth from AI
        result = analyze_pdf_content(pdf_file, text)
        
        if result:
            # Add metadata
            result['id'] = len(existing_data) + len(new_entries) + 1
            result['source_pdf'] = pdf_file
            
            new_entries.append(result)
            print(f"   ✅ Extracted: {result['query']}")
            print(f"      (Steps found: {len(result['ground_truth'])})")
            
            # Save progressively
            with open(BENCHMARK_FILE, "w") as f:
                json.dump(existing_data + new_entries, f, indent=2)
        else:
            print("   ⚠️ Skipped (AI could not extract steps)")

    print(f"\n--- 🎉 Done! Added {len(new_entries)} high-accuracy items to {BENCHMARK_FILE} ---")

if __name__ == "__main__":
    main()