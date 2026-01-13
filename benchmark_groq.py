import json
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# 1. Import your existing generation function
# Ensure main1.py is in the same folder
try:
    from main1 import generate_guide_from_rag
except ImportError:
    print("❌ CRITICAL: Could not find 'main1.py'. Make sure it is in this folder.")
    exit()

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Configuration
TEST_DATA_PATH = "benchmark_data.json"
OUTPUT_REPORT = "accuracy_report.csv"

def judge_submission(query, ground_truth, generated_json):
    """
    Uses Groq as a strict Professor to grade the student's answer.
    """
    client = Groq(api_key=API_KEY)
    
    # Format the data for the Judge
    gt_text = "\n".join([f"- {step}" for step in ground_truth])
    
    # Parse your RAG output (handling the JSON structure from main1.py)
    if isinstance(generated_json, dict) and "steps" in generated_json:
        student_text = "\n".join([f"- {s['instruction']}" for s in generated_json['steps']])
    else:
        student_text = str(generated_json)

    # STRICT JUDGE PROMPT
    prompt = f"""
    You are a strict Technical Manual Grader.
    
    ORIGINAL USER QUERY: "{query}"
    
    --- CORRECT ANSWER KEY (Ground Truth) ---
    {gt_text}
    
    --- STUDENT SUBMISSION (Generated Answer) ---
    {student_text}
    
    --- GRADING RUBRIC ---
    1. RECALL (0-5 pts): Did the student include ALL critical steps from the key?
    2. ORDER (0-5 pts): Are the steps in the correct physical sequence?
    3. SAFETY (0-5 pts): Did they include safety warnings (like unplugging) if the key had them?
    
    TASK:
    Compare the Student Submission to the Answer Key. Return a JSON with scores.
    
    OUTPUT JSON FORMAT:
    {{
        "recall_score": 0,
        "order_score": 0,
        "safety_score": 0,
        "total_score": 0, 
        "reasoning": "Briefly explain why points were deducted."
    }}
    """

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "total_score": 0}

def run_benchmark():
    # Load Test Cases
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Error: Create '{TEST_DATA_PATH}' first!")
        return

    with open(TEST_DATA_PATH, "r") as f:
        test_cases = json.load(f)

    results = []
    print(f"--- 📊 Starting Benchmark on {len(test_cases)} Test Cases ---")

    for case in test_cases:
        print(f"\n🔹 Testing Query: {case['query']}")
        
        # 1. Generate Answer (Using your RAG)
        # We wrap this in try/except so one failure doesn't stop the whole test
        try:
            rag_output = generate_guide_from_rag(case['query'])
        except Exception as e:
            print(f"   ❌ RAG Failed: {e}")
            rag_output = {"error": "Generation Failed"}

        # 2. Grade Answer (Using Groq Judge)
        grade = judge_submission(case['query'], case['ground_truth'], rag_output)
        
        print(f"   📝 Score: {grade.get('total_score', 0)}/15")
        print(f"   💡 Critique: {grade.get('reasoning', 'No reasoning')}")

        # 3. Log Data
        results.append({
            "Query": case['query'],
            "Recall (5)": grade.get('recall_score'),
            "Order (5)": grade.get('order_score'),
            "Safety (5)": grade.get('safety_score'),
            "Total (15)": grade.get('total_score'),
            "Judge Reasoning": grade.get('reasoning')
        })

    # Save to Excel/CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_REPORT, index=False)
    print(f"\n✅ Benchmark Complete! Saved to {OUTPUT_REPORT}")
    print(f"🌟 Average Accuracy: {df['Total (15)'].mean():.1f} / 15.0")

if __name__ == "__main__":
    run_benchmark()