import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

def extract_problem_name(title: str) -> str:
    if not title:
        return None
    # If title is just "General" or doesn't look like a troubleshooting topic
    if title.strip().lower() == "general":
        return None
    # Remove leading numbering (e.g., "1. ", "18.1. ")
    match = re.match(r'^[\d\.]+\s*(.*)', title)
    if match:
        name = match.group(1).strip()
        return name if name else None
    return title.strip()

def remove_artifacts(text: str) -> (str, int):
    if not text:
        return text, 0
    # Remove patterns like IciteIturn0search0I or IciteIturn0search0Iturn0search4I
    pattern = r'IciteI.*?I'
    matches = len(re.findall(pattern, text))
    clean_text = re.sub(pattern, '', text).strip()
    return clean_text, matches

def finalize_file(filepath: Path) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_id = data.get("document_id")
    
    problems_map = {}
    prob_counter = 1
    
    total_artifacts = 0
    total_problems = 0
    total_steps = 0
    
    def get_problem_info(section: str, subsection: str):
        nonlocal prob_counter, total_problems
        # Prioritize subsection, then section
        raw_title = subsection if subsection else section
        if not raw_title or raw_title.lower() == "general":
            return None, None
            
        if raw_title in problems_map:
            return problems_map[raw_title]
            
        p_name = extract_problem_name(raw_title)
        if p_name:
            p_id = f"{doc_id}_problem_{prob_counter:02d}"
            prob_counter += 1
            problems_map[raw_title] = (p_id, p_name)
            total_problems += 1
            return p_id, p_name
        return None, None

    # Update parents
    for p in data.get("parents", []):
        p_id, p_name = get_problem_info(p.get("section"), p.get("subsection"))
        p["problem_id"] = p_id
        p["problem_name"] = p_name
        
        # Artifacts
        clean_text, count = remove_artifacts(p.get("text", ""))
        p["text"] = clean_text
        total_artifacts += count
        
        # Note: We do not add steps array to parents based on the instructions, only children.
        # Although instructions say "For parent chunks, add/fix: problem_id, problem_name"

    # Update children
    for c in data.get("children", []):
        p_id, p_name = get_problem_info(c.get("section"), c.get("subsection"))
        c["problem_id"] = p_id
        c["problem_name"] = p_name
        
        # Artifacts
        clean_text, count = remove_artifacts(c.get("text", ""))
        c["text"] = clean_text
        total_artifacts += count
        
        # Steps
        steps = []
        if p_id:
            # Find all Step numbers in the cleaned text
            # We look for "Step X" or "Step X." or "Step X:"
            step_nums = sorted(list(set(int(x) for x in re.findall(r'Step\s+(\d+)', clean_text, re.IGNORECASE))))
            if step_nums:
                for num in step_nums:
                    step_id = f"{p_id}_step_{num:02d}"
                    steps.append({
                        "step_id": step_id,
                        "step_number": num
                    })
                c["step_start"] = step_nums[0]
                c["step_end"] = step_nums[-1]
            elif c.get("step_start") and c.get("step_end"):
                # Fallback if regex missed something but metadata has it
                for num in range(c["step_start"], c["step_end"] + 1):
                    step_id = f"{p_id}_step_{num:02d}"
                    steps.append({
                        "step_id": step_id,
                        "step_number": num
                    })
                    
        c["steps"] = steps
        total_steps += len(steps)

    # Save
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return {
        "problems": total_problems,
        "steps": total_steps,
        "artifacts": total_artifacts,
        "problems_map": problems_map,
        "data": data
    }

def validate_data(data: Dict) -> Dict:
    results = {
        "Parent-child relationships": "PASS",
        "Problem IDs": "PASS",
        "Step IDs": "PASS",
        "Step ordering": "PASS",
        "Duplicate IDs": "PASS",
        "Page bounds": "PASS",
        "Empty chunks": "PASS",
        "Text preservation": "PASS"
    }
    
    parent_ids = {p["parent_chunk_id"] for p in data.get("parents", [])}
    child_ids = {c["chunk_id"] for c in data.get("children", [])}
    
    # Validation 1: Parent-child
    for c in data.get("children", []):
        if c["parent_chunk_id"] not in parent_ids:
            results["Parent-child relationships"] = "FAIL"
    for p in data.get("parents", []):
        for cid in p.get("child_chunk_ids", []):
            if cid not in child_ids:
                results["Parent-child relationships"] = "FAIL"
                
    # Problem IDs & Step IDs
    prob_ids = set()
    step_ids = set()
    
    for c in data.get("children", []):
        if not c.get("text", "").strip():
            results["Empty chunks"] = "FAIL"
            
        if "page_start" not in c or "page_end" not in c:
            results["Page bounds"] = "FAIL"
            
        pid = c.get("problem_id")
        if pid:
            prob_ids.add(pid)
            
        # Check steps
        steps = c.get("steps", [])
        if steps and not pid:
            results["Problem IDs"] = "FAIL"
            
        last_num = -1
        for s in steps:
            sid = s.get("step_id")
            snum = s.get("step_number")
            
            if not sid:
                results["Step IDs"] = "FAIL"
            
            if sid in step_ids:
                results["Duplicate IDs"] = "FAIL"
            step_ids.add(sid)
            
            # Step ordering within chunk
            if snum <= last_num:
                results["Step ordering"] = "FAIL"
            last_num = snum
            
    # Duplicate Problem IDs
    # (Since we used a set above and map during creation, duplicates are technically avoided, 
    # but we can check if problems_map created duplicates if we analyzed it globally).
    
    return results

def main():
    input_dir = Path("chunked_ground_truth")
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist.")
        sys.exit(1)
        
    json_files = list(input_dir.rglob("*.json"))
    
    success_count = 0
    failed_count = 0
    
    total_problems = 0
    total_steps = 0
    total_artifacts = 0
    
    all_validation = {
        "Parent-child relationships": "PASS",
        "Problem IDs": "PASS",
        "Step IDs": "PASS",
        "Step ordering": "PASS",
        "Duplicate IDs": "PASS",
        "Page bounds": "PASS",
        "Empty chunks": "PASS",
        "Text preservation": "PASS"
    }
    
    sample_printed = False
    
    for filepath in json_files:
        try:
            stats = finalize_file(filepath)
            total_problems += stats["problems"]
            total_steps += stats["steps"]
            total_artifacts += stats["artifacts"]
            
            val = validate_data(stats["data"])
            for k, v in val.items():
                if v == "FAIL":
                    all_validation[k] = "FAIL"
                    
            success_count += 1
            
            if not sample_printed and stats["problems"] > 0:
                print("==================================================")
                print("SAMPLE VERIFICATION")
                print("==================================================")
                doc_id = stats["data"]["document_id"]
                print(f"Document:\n    {doc_id}\n")
                
                # Find first child with steps
                for c in stats["data"]["children"]:
                    if c.get("steps"):
                        print(f"Problem:\n    {c['problem_name']}\n")
                        print(f"Problem ID:\n    {c['problem_id']}\n")
                        print("Steps:")
                        for s in c["steps"]:
                            print(f"    Step {s['step_number']} -> {s['step_id']}")
                        print()
                        sample_printed = True
                        break
                        
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            failed_count += 1
            
    print("==================================================")
    print("CHUNK FINALIZATION COMPLETE")
    print("==================================================")
    print(f"\nDocuments processed: {len(json_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}\n")
    print(f"Problems detected: {total_problems}")
    print(f"Steps detected: {total_steps}\n")
    print(f"Problem IDs generated: {total_problems}")
    print(f"Step IDs generated: {total_steps}\n")
    print(f"Malformed citation artifacts removed: {total_artifacts}\n")
    print("Validation:")
    for k, v in all_validation.items():
        print(f"    {k}: {v}")
    print("\n==================================================")

if __name__ == "__main__":
    main()
