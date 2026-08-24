import json

with open("generated_step_images_metadata.json", "r", encoding="utf-8") as f:
    images = json.load(f)

img_9 = next((img for img in images if img["step_number"] == 9 and img["guide_name"] == "01_Starting_Power_Problems_Detailed"), None)
if img_9:
    print(f"Image 9 text: {img_9.get('step_text')}")
else:
    print("Image 9 not found")

with open("chunked_ground_truth/01_Starting_Power_Problems_Detailed.json", "r", encoding="utf-8") as f:
    doc = json.load(f)

print("\nGround truth steps 9:")
import re
for chunk in doc.get("children", []):
    for step in chunk.get("steps", []):
        if step["step_number"] == 9:
            pattern = rf"Step 9\.\s*(.*?)(?=\n\nStep \d+\.|\nAfter completing the steps|\n\nSafety:|\Z)"
            match = re.search(pattern, chunk["text"], re.DOTALL)
            if match:
                print(f"- {chunk['problem_name']}: {match.group(1).strip()}")
