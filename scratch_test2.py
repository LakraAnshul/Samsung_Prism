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

def clean_artifacts(text):
    if not text:
        return text
    return re.sub(r'IciteIturn\d+search\d+I', '', text)

def main():
    gt_dir = Path("chunked_ground_truth")
    source_metadata_file = Path("generated_step_images_metadata.json")
    
    documents = {}
    
    for gt_file in gt_dir.glob("*.json"):
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_id = data.get("document_id")
            if not doc_id:
                continue
                
            documents[doc_id] = {
                "appliance_type": data.get("appliance_type"),
                "brand": data.get("brand"),
                "model": data.get("model"),
                "steps": []
            }
            
            # Keep a global step counter for the document
            global_step_idx = 1
            
            for chunk in data.get("children", []):
                steps = chunk.get("steps", [])
                if not steps:
                    continue
                    
                problem_id = chunk.get("problem_id")
                problem_name = chunk.get("problem_name")
                chunk_text = chunk.get("text", "")
                
                for step in steps:
                    step_num = step["step_number"]
                    step_id = step["step_id"]
                    
                    pattern = rf"Step {step_num}\.\s*(.*?)(?=\n\nStep \d+\.|\nAfter completing the steps|\n\nSafety:|\Z)"
                    match = re.search(pattern, chunk_text, re.DOTALL)
                    step_text = match.group(1).strip() if match else ""
                    
                    documents[doc_id]["steps"].append({
                        "problem_id": problem_id,
                        "problem_name": problem_name,
                        "step_id": step_id,
                        "step_number": step_num, # Local step number
                        "global_step_number": global_step_idx,
                        "step_text": step_text,
                        "normalized_text": normalize_text(step_text)
                    })
                    global_step_idx += 1
                    
    with open(source_metadata_file, "r", encoding="utf-8") as f:
        images_metadata = json.load(f)
        
    matched_count = 0
    unmatched_count = 0
    ambiguous_count = 0

    for img in images_metadata:
        guide_name = img["guide_name"]
        img_step_num = img["step_number"]
        img_step_text = img.get("step_text", "")
        norm_img_text = normalize_text(img_step_text)
        
        target_doc_id = None
        for doc_id in documents.keys():
            if guide_name in doc_id:
                target_doc_id = doc_id
                break
                
        if not target_doc_id:
            unmatched_count += 1
            continue
            
        doc_data = documents[target_doc_id]
        
        possible_matches = []
        for gt_step in doc_data["steps"]:
            # Check both text match and either local or global step number match
            if norm_img_text == gt_step["normalized_text"] or norm_img_text in gt_step["normalized_text"] or gt_step["normalized_text"] in norm_img_text:
                possible_matches.append(gt_step)
                    
        if len(possible_matches) == 1:
            matched_count += 1
        elif len(possible_matches) > 1:
            # disambiguate with global or local step number
            filtered_matches = [m for m in possible_matches if m["global_step_number"] == img_step_num or m["step_number"] == img_step_num]
            if len(filtered_matches) == 1:
                matched_count += 1
            else:
                ambiguous_count += 1
        else:
            unmatched_count += 1
            
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")
    print(f"Ambiguous: {ambiguous_count}")

if __name__ == "__main__":
    main()
