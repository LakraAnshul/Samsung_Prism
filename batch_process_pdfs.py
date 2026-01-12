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

# 📂 FOLDER CONTAINING YOUR PDF MANUALS
PDF_SOURCE_FOLDER = "./Knowledge_Base/text" 

# Output Config
OUTPUT_DIR = "./extracted_images"
OUTPUT_JSON = "image_knowledge_base.json"
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []
    
    # Create a unique prefix from the PDF filename (e.g., "Manual_A" from "Manual_A.pdf")
    pdf_prefix = Path(pdf_path).stem.replace(" ", "_")
    
    print(f"   📖 Reading PDF: {Path(pdf_path).name} ({len(doc)} pages)")

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        page_text = page.get_text("text")
        # Grab first 500 words of text for context
        clean_text = " ".join(page_text.split()[:500]) 
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # FILTERS: Skip tiny icons (<6KB) or huge backgrounds (>3MB)
            if len(image_bytes) < 6000: continue
            if len(image_bytes) > 3 * 1024 * 1024: continue

            # Unique Filename: PDF_Name + Page + ImgID
            image_filename = f"{pdf_prefix}_p{page_index+1}_img{img_index+1}.{image_ext}"
            image_filepath = os.path.join(OUTPUT_DIR, image_filename)
            
            # Write Image to Disk
            with open(image_filepath, "wb") as f:
                f.write(image_bytes)
            
            extracted_data.append({
                "id": image_filename,
                "file_path": image_filepath,
                "page_context": clean_text, 
                "page_number": page_index + 1
            })
            
    return extracted_data

def analyze_images_with_groq(extracted_items):
    client = Groq(api_key=GROQ_API_KEY)
    valid_entries = []
    
    print(f"   👁️  Analyzing {len(extracted_items)} potential images...")
    
    for item in extracted_items:
        try:
            base64_image = encode_image_to_base64(item['file_path'])
            
            prompt_text = f"""
            You are a technical expert.
            CONTEXT: "{item['page_context']}"
            
            TASK:
            1. Analyze this image.
            2. If it is junk (Logo, Icon, Blank, QR Code) -> Set 'problem_name' to "DELETE_ME".
            3. If valid, write a 'dense_caption' describing the action being performed.
            
            OUTPUT JSON:
            {{
                "problem_name": "Task Name or DELETE_ME",
                "dense_caption": "Description",
                "detected_objects": ["list"]
            }}
            """
            
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                model=MODEL_ID,
                response_format={"type": "json_object"}, 
                temperature=0.1 
            )
            
            ai_data = json.loads(chat_completion.choices[0].message.content)
            p_name = ai_data.get("problem_name", "General")
            
            # Skip Junk
            if p_name == "DELETE_ME" or "logo" in p_name.lower():
                print(f"      🗑️  Skipping Junk: {item['id']}")
                continue 

            entry = {
                "id": item['id'],
                "file_path": item['file_path'],
                "problem_name": p_name,
                "dense_caption": ai_data.get("dense_caption", "No description"),
                "detected_objects": ai_data.get("detected_objects", [])
            }
            valid_entries.append(entry)
            print(f"      ✅  Valid: {p_name}")

        except Exception as e:
            print(f"      ❌ Error on {item['id']}: {e}")

    return valid_entries

def main():
    # 1. Get List of PDFs
    if not os.path.exists(PDF_SOURCE_FOLDER):
        print(f"❌ Error: Folder '{PDF_SOURCE_FOLDER}' not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_SOURCE_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"❌ No PDFs found in {PDF_SOURCE_FOLDER}")
        return

    print(f"--- 🚀 Starting Batch Process for {len(pdf_files)} Manuals ---")

    # 2. Iterate through each PDF
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_SOURCE_FOLDER, pdf_file)
        
        print(f"\n[{i+1}/{len(pdf_files)}] Processing: {pdf_file}")
        
        # A. Extract Images
        raw_images = extract_images_from_pdf(pdf_path)
        if not raw_images:
            print("   ⚠️  No images found in this PDF.")
            continue
            
        # B. Analyze with Groq
        new_entries = analyze_images_with_groq(raw_images)
        
        # C. Append to JSON immediately (Safe Save)
        existing_data = []
        if os.path.exists(OUTPUT_JSON):
            try:
                with open(OUTPUT_JSON, "r") as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        combined_data = existing_data + new_entries
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(combined_data, f, indent=2)
            
        print(f"   💾 Saved {len(new_entries)} new entries from {pdf_file}. Total DB: {len(combined_data)}")

    print("\n--- 🎉 ALL PDFS PROCESSED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()