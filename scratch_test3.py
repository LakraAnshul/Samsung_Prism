import json
import os
import re
from pathlib import Path

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'IciteIturn\d+search\d+I', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def main():
    gt_dir = Path("chunked_ground_truth")
    source_metadata_file = Path("generated_step_images_metadata.json")
    documents = {}
    for gt_file in gt_dir.glob("*.json"):
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_id = data.get("document_id")
            if not doc_id: continue
            documents[doc_id] = {"steps": []}
            global_step_idx = 1
            for chunk in data.get("children", []):
                for step in chunk.get("steps", []):
                    step_num = step["step_number"]
                    pattern = rf"Step {step_num}\.\s*(.*?)(?=\n\nStep \d+\.|\nAfter completing the steps|\n\nSafety:|\Z)"
                    match = re.search(pattern, chunk.get("text", ""), re.DOTALL)
                    step_text = match.group(1).strip() if match else ""
                    documents[doc_id]["steps"].append({
                        "step_number": step_num,
                        "global_step_number": global_step_idx,
                        "normalized_text": normalize_text(step_text),
                        "step_text": step_text
                    })
                    global_step_idx += 1
                    
    with open(source_metadata_file, "r", encoding="utf-8") as f:
        images_metadata = json.load(f)
        
    for img in images_metadata:
        guide_name = img["guide_name"]
        img_step_num = img["step_number"]
        norm_img_text = normalize_text(img.get("step_text", ""))
        
        target_doc_id = next((doc_id for doc_id in documents.keys() if guide_name in doc_id), None)
        if not target_doc_id: continue
            
        doc_data = documents[target_doc_id]
        possible_matches = [gt for gt in doc_data["steps"] if norm_img_text == gt["normalized_text"] or norm_img_text in gt["normalized_text"] or gt["normalized_text"] in norm_img_text]
        
        if len(possible_matches) > 1:
            filtered_matches = [m for m in possible_matches if m["global_step_number"] == img_step_num or m["step_number"] == img_step_num]
            if len(filtered_matches) != 1:
                print(f"AMBIGUOUS: Image step {img_step_num} '{img.get('step_text')}'")
                for pm in possible_matches:
                    print(f"  -> GT local {pm['step_number']} / global {pm['global_step_number']}: '{pm['step_text']}'")
                break

if __name__ == "__main__":
    main()
