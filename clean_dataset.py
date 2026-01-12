import os
import json
import shutil

# --- CONFIGURATION ---
JSON_PATH = "image_knowledge_base.json"
SOURCE_DIR = "./extracted_images"
DEST_DIR = "./final_cleaned_dataset"  # Only used if MODE = "MOVE"

# 🔴 CHOOSE YOUR MODE:
# "MOVE"   -> Moves VALID images to a new folder (Safe, Recommended).
# "DELETE" -> Deletes INVALID images from the source folder (Destructive/Permanent).
MODE = "MOVE" 

def clean_dataset():
    # 1. Load the Knowledge Base
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        kb_data = json.load(f)
    
    # 2. Extract Valid Filenames
    # We use a 'set' for instant lookup speed
    valid_filenames = {entry['id'] for entry in kb_data}
    print(f"--- 📋 Loaded {len(valid_filenames)} valid files from JSON ---")

    # 3. Scan the Source Directory
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Source directory '{SOURCE_DIR}' not found.")
        return

    all_files = os.listdir(SOURCE_DIR)
    print(f"--- 📂 Scanning '{SOURCE_DIR}' containing {len(all_files)} files ---")

    # Create Dest Dir if Moving
    if MODE == "MOVE":
        os.makedirs(DEST_DIR, exist_ok=True)

    # 4. Process Files
    action_count = 0
    
    for filename in all_files:
        # Construct full path
        file_path = os.path.join(SOURCE_DIR, filename)
        
        # Skip directories, only process files
        if not os.path.isfile(file_path):
            continue

        is_valid = filename in valid_filenames

        if MODE == "MOVE":
            # Action: Move ONLY valid files
            if is_valid:
                shutil.move(file_path, os.path.join(DEST_DIR, filename))
                print(f"✅ Moved Valid Image: {filename}")
                action_count += 1
            else:
                # Optional: Print what is being left behind
                # print(f"   Ignored Junk: {filename}")
                pass

        elif MODE == "DELETE":
            # Action: Delete ONLY invalid files
            if not is_valid:
                os.remove(file_path)
                print(f"🗑️ Deleted Junk Image: {filename}")
                action_count += 1

    # 5. Summary
    print("-" * 40)
    if MODE == "MOVE":
        print(f"🎉 Success! Moved {action_count} valid images to '{DEST_DIR}'.")
        print(f"   (Junk files are left in '{SOURCE_DIR}')")
    else:
        print(f"🎉 Success! Deleted {action_count} junk files from '{SOURCE_DIR}'.")

if __name__ == "__main__":
    clean_dataset()