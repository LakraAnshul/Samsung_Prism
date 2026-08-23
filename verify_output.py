import json, os
from collections import Counter

meta_path = "./generated_step_images_20260824_0052/generated_step_images_metadata.json"
img_dir   = "./generated_step_images_20260824_0052"

with open(meta_path, encoding="utf-8") as f:
    data = json.load(f)

ids        = [d["id"] for d in data]
unique_ids = set(ids)
png_files  = set(f for f in os.listdir(img_dir) if f.endswith(".png"))

print("=== FINAL VERIFICATION REPORT ===")
print(f"Metadata entries   : {len(data)}")
print(f"Unique IDs in meta : {len(unique_ids)}")
print(f"Duplicate entries  : {len(data) - len(unique_ids)}")
print(f"PNG files on disk  : {len(png_files)}")
print(f"Meta <-> Disk match: {unique_ids == png_files}")
print()

by_guide = Counter(d["guide_name"] for d in data)
print("Steps per guide:")
for g, cnt in sorted(by_guide.items()):
    print(f"  {g}: {cnt}")

print()
print("Sample entries (first 3):")
for e in data[:3]:
    print(f"  {e['id']} | step {e['step_number']} | {e['step_text'][:70]}...")

print()
print("Sample entries (last 3):")
for e in data[-3:]:
    print(f"  {e['id']} | step {e['step_number']} | {e['step_text'][:70]}...")
