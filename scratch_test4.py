import json

with open("generated_step_images_metadata.json", "r", encoding="utf-8") as f:
    images = json.load(f)
    
unmatched_ids = [
    "08_Leakage_Problems_step014.png",
    "08_Leakage_Problems_step021.png",
    "09_Temperature_Problems_step014.png",
    "09_Temperature_Problems_step021.png",
    "12_Maintenance_Problems_step042.png"
]

for img in images:
    if img["id"] in unmatched_ids:
        print(f"\nImage ID: {img['id']}")
        print(f"Step text: '{img.get('step_text')}'")
        print(f"Dense caption: '{img.get('dense_caption')}'")
