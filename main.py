import json
import os
import sys

# Import the core engine from backend
backend_path = os.path.abspath("./backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from main import generate_guide_from_rag

if __name__ == "__main__":
    q1 = "How to clean the debris filter on Samsung WA5471ABP?"
    print(f"\n👉 TESTING QUERY: {q1}")
    result = generate_guide_from_rag(q1, model="WA5471ABP", mode="CLOUD")
    
    # Save output to audit trail
    output_filename = "guide.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2))
        print(f"\n--- 💾 Guide output saved to '{output_filename}' ---")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error saving guide: {e}")