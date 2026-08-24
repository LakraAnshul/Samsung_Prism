import json

with open("chunked_ground_truth/08_Leakage_Problems.json", "r", encoding="utf-8") as f:
    doc = json.load(f)

global_idx = 1
found = False
for chunk in doc.get("children", []):
    for step in chunk.get("steps", []):
        if global_idx == 14:
            print(f"Global Step 14 is: {step['step_id']}")
            found = True
        global_idx += 1
        
if not found:
    print(f"Global Step 14 not found. Total steps: {global_idx - 1}")
