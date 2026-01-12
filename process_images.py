import os
import fitz  # PyMuPDF
import json
import base64
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# 🔴 CHANGE THIS PDF PATH for every new manual you run
PDF_PATH = "C:\\Users\\user\\Documents\\GitHub\\Prism\\Knowledge_Base\\text\\How to Clean a Samsung DC68 Washing Machine Filter.pdf"

OUTPUT_DIR = "./extracted_images"
OUTPUT_JSON = "image_knowledge_base.json"
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_images_and_context(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []
    
    # Get a safe prefix from the PDF filename (e.g., "Samsung_Washer" from "Samsung_Washer.pdf")
    pdf_prefix = Path(pdf_path).stem.replace(" ", "_")
    
    print(f"--- 📂 Opening PDF: {pdf_path} ({len(doc)} pages) ---")

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        page_text = page.get_text("text")
        clean_text = " ".join(page_text.split()[:500]) 
        
        if image_list:
            print(f"   📄 Page {page_index + 1}: Found {len(image_list)} images.")
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Filters
            if len(image_bytes) < 6000: continue
            if len(image_bytes) > 3 * 1024 * 1024: continue

            # 🔴 NEW: Unique filename including PDF name to prevent overwrites
            image_filename = f"{pdf_prefix}_p{page_index+1}_img{img_index+1}.{image_ext}"
            image_filepath = os.path.join(OUTPUT_DIR, image_filename)
            
            # Save Image
            with open(image_filepath, "wb") as f:
                f.write(image_bytes)
            
            extracted_data.append({
                "id": image_filename,
                "file_path": image_filepath,
                "page_context": clean_text, 
                "page_number": page_index + 1
            })
            
    return extracted_data

def generate_metadata_with_groq(extracted_items):
    client = Groq(api_key=GROQ_API_KEY)
    new_entries = []
    
    print(f"\n--- 👁️ Processing {len(extracted_items)} Images with Llama 4 Scout ---")
    
    for item in extracted_items:
        try:
            base64_image = encode_image_to_base64(item['file_path'])
            
            prompt_text = f"""
            You are a technical expert analyzing images from a technical manual.
            
            CONTEXT TEXT FROM THE SAME PDF PAGE:
            "{item['page_context']}"
            
            TASK:
            1. Analyze the image + Context.
            2. If the image is a Logo, Barcode, QR Code, random decorative line, or completely blank/black -> Set 'problem_name' to "DELETE_ME".
            3. If it is a valid instructional image, describe it in detail.
            4. 'dense_caption': MUST describe the action. Do NOT say "No image provided". Use context to infer.
            
            OUTPUT JSON ONLY:
            {{
                "problem_name": "Specific Task Name or DELETE_ME",
                "dense_caption": "Detailed description of action + parts.",
                "detected_objects": ["list", "of", "parts"]
            }}
            """
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model=MODEL_ID,
                response_format={"type": "json_object"}, 
                temperature=0.1 
            )
            
            ai_data = json.loads(chat_completion.choices[0].message.content)
            p_name = ai_data.get("problem_name", "General")
            
            if p_name == "DELETE_ME" or "logo" in p_name.lower():
                print(f"🗑️ Trash Identified: {item['id']} (Ignoring...)")
                continue 

            entry = {
                "id": item['id'],
                "file_path": item['file_path'],
                "problem_name": p_name,
                "dense_caption": ai_data.get("dense_caption", "Context suggests this is part of " + p_name),
                "detected_objects": ai_data.get("detected_objects", [])
            }
            
            new_entries.append(entry)
            print(f"✅ Kept: {item['id']} -> {p_name}")
            
        except Exception as e:
            print(f"❌ Error on {item['id']}: {str(e)}")

    return new_entries

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file '{PDF_PATH}' not found.")
    else:
        # 1. Extract & Process NEW Images
        raw_items = extract_images_and_context(PDF_PATH)
        new_kb_data = generate_metadata_with_groq(raw_items)
        
        # 2. Load EXISTING Data (Append Logic)
        existing_data = []
        if os.path.exists(OUTPUT_JSON):
            try:
                with open(OUTPUT_JSON, "r") as f:
                    existing_data = json.load(f)
                print(f"--- 📥 Loaded {len(existing_data)} existing entries from {OUTPUT_JSON} ---")
            except json.JSONDecodeError:
                print("⚠️ Existing JSON was corrupt. Starting fresh.")
        
        # 3. Combine and Save
        combined_data = existing_data + new_kb_data
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(combined_data, f, indent=2)
            
        print(f"\n--- 🎉 Success! Added {len(new_kb_data)} new entries. Total DB Size: {len(combined_data)} ---")