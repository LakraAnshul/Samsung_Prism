import json
import os
import re
from pathlib import Path
from collections import Counter

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
    print("==================================================")
    print("GUIDE WEAVE — STAGE 5 (REVISION)")
    print("IMAGE METADATA NORMALIZATION")
    print("==================================================")

    gt_dir = Path("chunked_ground_truth")
    source_metadata_file = Path("generated_step_images_metadata.json")
    
    # 1. Load ground truth
    documents = {}
    total_gt_documents = 0
    
    unique_problems = set()
    all_gt_step_ids = []

    for gt_file in gt_dir.glob("*.json"):
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_id = data.get("document_id")
            if not doc_id:
                continue
            
            total_gt_documents += 1
            documents[doc_id] = {
                "appliance_type": data.get("appliance_type"),
                "brand": data.get("brand"),
                "model": data.get("model"),
                "steps": []
            }
            
            global_step_idx = 1
            
            for chunk in data.get("children", []):
                steps = chunk.get("steps", [])
                if not steps:
                    continue
                    
                problem_id = chunk.get("problem_id")
                problem_name = chunk.get("problem_name")
                if problem_id:
                    unique_problems.add(problem_id)
                
                chunk_text = chunk.get("text", "")
                
                for step in steps:
                    step_num = step["step_number"]
                    step_id = step["step_id"]
                    all_gt_step_ids.append(step_id)
                    
                    pattern = rf"Step {step_num}\.\s*(.*?)(?=\n\nStep \d+\.|\nAfter completing the steps|\n\nSafety:|\Z)"
                    match = re.search(pattern, chunk_text, re.DOTALL)
                    step_text = match.group(1).strip() if match else ""
                    
                    documents[doc_id]["steps"].append({
                        "problem_id": problem_id,
                        "problem_name": problem_name,
                        "step_id": step_id,
                        "step_number": step_num,
                        "global_step_number": global_step_idx,
                        "step_text": step_text,
                        "normalized_text": normalize_text(step_text)
                    })
                    global_step_idx += 1
                    
    total_gt_problems = len(unique_problems)
    total_gt_steps = len(all_gt_step_ids)
    
    # 2. Process images
    with open(source_metadata_file, "r", encoding="utf-8") as f:
        images_metadata = json.load(f)
        
    matched_count = 0
    unmatched_count = 0
    ambiguous_count = 0
    missing_files_count = 0
    
    final_metadata = []
    report_details = []
    
    seen_image_ids = set()
    seen_file_paths = set()
    linked_step_ids = []
    
    citation_artifacts_removed_count = 0

    for img in images_metadata:
        orig_filename = img["id"]
        guide_name = img["guide_name"]
        img_step_num = img["step_number"]
        img_step_text = img.get("step_text", "")
        img_dense_caption = img.get("dense_caption", "")
        
        if re.search(r'IciteIturn\d+search\d+I', img_step_text) or re.search(r'IciteIturn\d+search\d+I', img_dense_caption):
            citation_artifacts_removed_count += 1
            
        clean_img_dense_caption = clean_artifacts(img_dense_caption)
        norm_img_text = normalize_text(img_step_text)
        
        final_file_path = f"./generated_step_images_20260824_0052/{orig_filename}"
        
        if not os.path.exists(Path(final_file_path)):
            missing_files_count += 1
            report_details.append({
                "image_id": orig_filename,
                "guide_name": guide_name,
                "step_number": img_step_num,
                "reason": "Missing image file"
            })
            continue

        target_doc_id = None
        for doc_id in documents.keys():
            if guide_name in doc_id:
                target_doc_id = doc_id
                break
                
        if not target_doc_id:
            unmatched_count += 1
            report_details.append({
                "image_id": orig_filename,
                "guide_name": guide_name,
                "step_number": img_step_num,
                "reason": f"No matching document found for guide_name: {guide_name}"
            })
            continue
            
        doc_data = documents[target_doc_id]
        
        # PRIMARY MATCHING BY guide_name + step_number (global)
        possible_matches = [s for s in doc_data["steps"] if s["global_step_number"] == img_step_num]
        
        if len(possible_matches) == 1:
            match = possible_matches[0]
            matched_count += 1
            
            # Check text match for metadata validation
            gt_norm_text = match["normalized_text"]
            is_text_match = (norm_img_text == gt_norm_text or norm_img_text in gt_norm_text or gt_norm_text in norm_img_text)
            
            record = {
                "image_id": orig_filename,
                "file_path": final_file_path,
                "appliance_type": doc_data["appliance_type"],
                "brand": doc_data["brand"],
                "model": doc_data["model"],
                "document_id": target_doc_id,
                "problem_id": match["problem_id"],
                "problem_name": match["problem_name"],
                "step_id": match["step_id"],
                "step_number": img_step_num, # Keep image's original step_number
                "step_text": clean_artifacts(match["step_text"]), # Authoritative GT text
                "dense_caption": clean_img_dense_caption,
                "detected_objects": img.get("detected_objects", []),
                "metadata_text_match": is_text_match,
                "linking_method": "guide_name + step_number"
            }
            final_metadata.append(record)
            
            seen_image_ids.add(orig_filename)
            seen_file_paths.add(final_file_path)
            linked_step_ids.append(match["step_id"])
            
        elif len(possible_matches) > 1:
            ambiguous_count += 1
            report_details.append({
                "image_id": orig_filename,
                "guide_name": guide_name,
                "step_number": img_step_num,
                "reason": f"Ambiguous match: {len(possible_matches)} possible ground-truth steps found via step_number"
            })
        else:
            unmatched_count += 1
            report_details.append({
                "image_id": orig_filename,
                "guide_name": guide_name,
                "step_number": img_step_num,
                "reason": "No reliable ground-truth match (step_number out of bounds)"
            })
            
    # Sort final metadata
    final_metadata.sort(key=lambda x: (x["document_id"], x["step_number"]))
    
    # Save final metadata
    with open("image_knowledge_base_final.json", "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=4)
        
    # Analyze images per step
    step_counts = Counter(linked_step_ids)
    images_per_step_distribution = {}
    
    steps_without_images = []
    for sid in all_gt_step_ids:
        c = step_counts.get(sid, 0)
        label = f"{c} images"
        images_per_step_distribution[label] = images_per_step_distribution.get(label, 0) + 1
        if c == 0:
            steps_without_images.append(sid)
            
    # Add count for steps that weren't in the ground truth but were linked (should be 0)
    for sid, c in step_counts.items():
        if sid not in all_gt_step_ids:
            label = f"{c} images (unknown step)"
            images_per_step_distribution[label] = images_per_step_distribution.get(label, 0) + 1

    # Validation report
    report = {
        "total_ground_truth_steps": total_gt_steps,
        "total_images": len(images_metadata),
        "successfully_linked": matched_count,
        "unmatched": unmatched_count,
        "ambiguous": ambiguous_count,
        "steps_without_images": len(steps_without_images),
        "images_per_step_distribution": images_per_step_distribution,
        "missing_files": missing_files_count,
        "duplicate_image_ids": len(final_metadata) - len(seen_image_ids),
        "duplicate_file_paths": len(final_metadata) - len(seen_file_paths),
        "unique_documents": len(set(r["document_id"] for r in final_metadata)),
        "unique_problems": len(set(r["problem_id"] for r in final_metadata)),
        "unique_steps": len(set(r["step_id"] for r in final_metadata)),
        "citation_artifacts_removed": citation_artifacts_removed_count,
        "details": report_details
    }
    
    with open("image_metadata_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"Total ground-truth steps: {total_gt_steps}")
    print(f"Total images: {len(images_metadata)}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")
    print(f"Ambiguous: {ambiguous_count}")
    print(f"Steps without images: {len(steps_without_images)}")
    print(f"Missing image files: {missing_files_count}")
    
    if steps_without_images:
        print("\nGround-truth steps WITHOUT images:")
        for sid in steps_without_images:
            print(f"  - {sid}")

if __name__ == "__main__":
    main()
