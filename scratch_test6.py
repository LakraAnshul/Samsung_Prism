import json
import re

with open("chunked_ground_truth/08_Leakage_Problems.json", "r", encoding="utf-8") as f:
    doc = json.load(f)

global_idx = 1
for chunk in doc.get("children", []):
    for step in chunk.get("steps", []):
        if global_idx == 14:
            step_num = step["step_number"]
            pattern = rf"Step {step_num}\.\s*(.*?)(?=\n\nStep \d+\.|\nAfter completing the steps|\n\nSafety:|\Z)"
            match = re.search(pattern, chunk.get("text", ""), re.DOTALL)
            print(f"GT Text for 14: {match.group(1).strip() if match else 'None'}")
        global_idx += 1
