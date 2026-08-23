import base64
import json
import os
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq

# --- CONFIGURATION ---
load_dotenv()
load_dotenv("backend/.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PDF_SOURCE_FOLDER = "./Knowledge_Base/text"
OUTPUT_DIR = "./extracted_images"
OUTPUT_JSON = "image_knowledge_base.json"
BACKEND_OUTPUT_JSON = "backend/image_knowledge_base.json"
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_MODELS = ["WA5471ABP", "WF5M5100AW", "WF350ANR", "DC68", "WW90T504DAN", "WD80T654DBX"]

def extract_model_from_filename(filename: str) -> str:
    for model in KNOWN_MODELS:
        if model.lower() in filename.lower():
            return model
    match = re.search(r"\b(?:WA|WW|WD|WF|DC|DV|WT)\d+[A-Za-z0-9-]*\b", filename, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return "General"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_surrounding_caption(page, img_rect):
    """Finds text immediately above or below the image bounding box."""
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    captions = []
    
    for b in blocks:
        if len(b) >= 5 and isinstance(b[4], str):
            bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4].strip()
            # If block is within 100 vertical points of image
            if (abs(by0 - img_rect.y1) < 80) or (abs(img_rect.y0 - by1) < 80):
                # Check for caption-like phrases (e.g. Figure, Step, Caution, Note, or bold headers)
                if re.search(r"^(?:Figure|Fig\.|Step|\d+\.|Caution|Note|Warning)", btext, re.IGNORECASE) or len(btext) < 120:
                    captions.append(btext.replace("\n", " "))
                    
    return " | ".join(captions[:2]) if captions else ""

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []
    pdf_filename = Path(pdf_path).name
    pdf_prefix = Path(pdf_path).stem.replace(" ", "_")
    model_tag = extract_model_from_filename(pdf_filename)

    print(f"   📖 Reading PDF: {pdf_filename} (Model: {model_tag}, {len(doc)} pages)")

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        page_text = page.get_text("text")
        clean_page_text = " ".join(page_text.split()[:400])

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # FILTERS: Skip tiny icons (<1.5KB) or huge background blobs (>3MB)
            if len(image_bytes) < 1500:
                continue
            if len(image_bytes) > 3 * 1024 * 1024:
                continue

            # Bounding box extraction
            img_rects = page.get_image_rects(xref)
            bbox = [round(v, 2) for v in (img_rects[0] if img_rects else fitz.Rect(0, 0, 0, 0))]
            caption = extract_surrounding_caption(page, img_rects[0]) if img_rects else ""

            image_filename = f"{pdf_prefix}_p{page_index + 1}_img{img_index + 1}.{image_ext}"
            image_filepath = os.path.join(OUTPUT_DIR, image_filename)

            # Write Image to Disk
            with open(image_filepath, "wb") as f:
                f.write(image_bytes)

            extracted_data.append(
                {
                    "id": image_filename,
                    "file_path": image_filepath,
                    "model": model_tag,
                    "source_pdf": pdf_filename,
                    "page_number": page_index + 1,
                    "bbox": bbox,
                    "caption": caption,
                    "page_context": clean_page_text,
                }
            )

    return extracted_data

def analyze_images_with_groq(extracted_items, existing_lookup=None):
    if existing_lookup is None:
        existing_lookup = {}

    client = None
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            print(f"⚠️ Could not initialize Groq client: {e}")

    valid_entries = []
    print(f"   👁️  Analyzing {len(extracted_items)} extracted images...")

    for item in extracted_items:
        # Check cache
        if item["id"] in existing_lookup:
            cached = existing_lookup[item["id"]]
            # Update spatial fields while preserving analyzed captions
            cached["bbox"] = item.get("bbox", [0, 0, 0, 0])
            cached["page_number"] = item.get("page_number", 1)
            cached["source_pdf"] = item.get("source_pdf", "")
            cached["model"] = item.get("model", "General")
            cached["caption"] = item.get("caption", "")
            valid_entries.append(cached)
            print(f"      ⚡ Cached: {item['id']} ({cached.get('problem_name', 'OK')})")
            continue

        # If no Groq API Key, fallback to structured heuristic metadata
        if not client:
            caption_text = item.get("caption") or item.get("page_context")[:150]
            entry = {
                "id": item["id"],
                "file_path": item["file_path"],
                "model": item["model"],
                "source_pdf": item["source_pdf"],
                "page_number": item["page_number"],
                "bbox": item["bbox"],
                "caption": item["caption"],
                "problem_name": item["model"] + " Component/Step",
                "dense_caption": f"Instructional diagram on page {item['page_number']} for {item['model']}. {caption_text}",
                "detected_objects": ["washing machine", "component", item["model"].lower()]
            }
            valid_entries.append(entry)
            continue

        try:
            base64_image = encode_image_to_base64(item["file_path"])
            prompt_text = f"""
            You are a technical manual vision expert.
            MODEL: "{item.get('model', 'Samsung Washer')}"
            PAGE CONTEXT: "{item['page_context']}"
            LOCAL CAPTION: "{item.get('caption', '')}"

            TASK:
            1. Analyze this image.
            2. If it is JUNK (pure company logo, barcode, QR code, blank, or decorative line) -> Set 'problem_name' to "DELETE_ME".
            3. If it is a valid instructional illustration or component photo, generate:
               - 'problem_name': Specific task or component name (e.g. 'Debris Filter Removal', 'Drain Hose Connection').
               - 'dense_caption': Exact description of the visual action and parts shown.
               - 'detected_objects': List of identifiable physical parts.

            OUTPUT JSON ONLY:
            {{
                "problem_name": "Task Name or DELETE_ME",
                "dense_caption": "Detailed visual action description",
                "detected_objects": ["part1", "part2"]
            }}
            """

            max_retries = 2
            ai_data = None
            for attempt in range(max_retries):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        model=MODEL_ID,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )
                    ai_data = json.loads(chat_completion.choices[0].message.content)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(3)
                    else:
                        print(f"      ⚠️ Groq vision API error on {item['id']}: {e}")

            p_name = ai_data.get("problem_name", "General Component") if ai_data else "Instructional Image"

            # Skip Junk
            if p_name == "DELETE_ME" or "logo" in p_name.lower():
                print(f"      🗑️  Skipping Junk: {item['id']}")
                continue

            entry = {
                "id": item["id"],
                "file_path": item["file_path"],
                "model": item["model"],
                "source_pdf": item["source_pdf"],
                "page_number": item["page_number"],
                "bbox": item["bbox"],
                "caption": item["caption"],
                "problem_name": p_name,
                "dense_caption": ai_data.get("dense_caption", item.get("caption") or "Component diagram") if ai_data else "Component diagram",
                "detected_objects": ai_data.get("detected_objects", []) if ai_data else ["washing machine"],
            }
            valid_entries.append(entry)
            print(f"      ✅  Valid: {item['id']} -> {p_name}")

        except Exception as e:
            print(f"      ❌ Error on {item['id']}: {e}")

    return valid_entries

def main():
    if not os.path.exists(PDF_SOURCE_FOLDER):
        print(f"❌ Error: Folder '{PDF_SOURCE_FOLDER}' not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_SOURCE_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDFs found in '{PDF_SOURCE_FOLDER}'")
        return

    print(f"--- 🚀 Starting Linked Image Extraction for {len(pdf_files)} Manuals ---")

    # Load existing KB into lookup dict to preserve existing annotations
    existing_lookup = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                existing_list = json.load(f)
                for entry in existing_list:
                    if "id" in entry:
                        existing_lookup[entry["id"]] = entry
            print(f"--- 📥 Loaded {len(existing_lookup)} pre-existing entries from '{OUTPUT_JSON}' ---")
        except Exception as e:
            print(f"⚠️ Notice: Starting fresh KB ({e})")

    all_valid_entries = []

    for i, pdf_file in enumerate(sorted(pdf_files)):
        pdf_path = os.path.join(PDF_SOURCE_FOLDER, pdf_file)
        print(f"\n[{i + 1}/{len(pdf_files)}] Processing: {pdf_file}")

        raw_images = extract_images_from_pdf(pdf_path)
        if not raw_images:
            print("   ⚠️  No images found in this PDF.")
            continue

        new_entries = analyze_images_with_groq(raw_images, existing_lookup)
        all_valid_entries.extend(new_entries)

    # Save to root and backend
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_valid_entries, f, indent=2)

    os.makedirs(os.path.dirname(BACKEND_OUTPUT_JSON), exist_ok=True)
    with open(BACKEND_OUTPUT_JSON, "w") as f:
        json.dump(all_valid_entries, f, indent=2)

    print(f"\n--- 🎉 Linked Image Extraction Complete. Total DB Size: {len(all_valid_entries)} images ---")

if __name__ == "__main__":
    main()
