import os
import json
import time
import math
from difflib import SequenceMatcher
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings

# --- IMPORT YOUR LOCAL LLM FUNCTION ---
# This imports the code you just shared (saved as main1.py)
try:
    from main_local_llm import generate_guide_from_rag, get_retriever
except ImportError:
    print("❌ Critical Error: 'main1.py' not found.")
    print("   Please save your Local LLM code as 'main1.py' in this folder.")
    exit()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BENCHMARK_FILE = "benchmark_data.json"

# --- METRIC 1: FAITHFULNESS (Hallucination Check) ---
def calculate_faithfulness(answer, context_list):
    """
    Uses Groq (Judge) to check if the Local LLM (Student) is hallucinating.
    """
    client = Groq(api_key=GROQ_API_KEY)
    context_text = "\n".join(context_list)[:15000] 
    
    prompt = f"""
    You are a Fact Checker.
    CONTEXT:
    {context_text}
    STATEMENT:
    "{answer}"
    
    TASK: Return a score (0.0 to 1.0) on how much of the STATEMENT is supported by CONTEXT.
    OUTPUT JSON ONLY: {{ "score": 0.5 }}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content).get("score", 0.0)
    except:
        return 0.0

# --- METRIC 2: RELEVANCY (Topic Check) ---
def calculate_relevancy(query, answer):
    """
    Uses Groq (Judge) to check if the Local LLM (Student) answered the specific question.
    """
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    You are a Relevance Grader.
    USER QUERY: "{query}"
    ANSWER: "{answer}"
    
    TASK: Rate relevance (0.0 to 1.0).
    OUTPUT JSON ONLY: {{ "score": 0.5 }}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content).get("score", 0.0)
    except:
        return 0.0

# --- MAIN JUDGE (Accuracy) ---
def run_llm_judge(query, ground_truth, student_answer):
    client = Groq(api_key=GROQ_API_KEY)
    gt_text = "\n".join([f"- {s}" for s in ground_truth])
    student_text = str(student_answer)
    
    prompt = f"""
    You are a strict QA Auditor.
    QUERY: "{query}"
    EXPECTED ANSWER (Ground Truth):
    {gt_text}
    ACTUAL ANSWER (Student System):
    {student_text}
    
    TASK: Grade accuracy (0-100).
    OUTPUT JSON: {{ "total_score": 0, "reasoning": "Critique" }}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"total_score": 0, "reasoning": "Judge Error"}

# --- HELPER: Find Ground Truth ---
def find_matching_ground_truth(user_query, benchmark_data):
    best_match = None
    highest_score = 0.0
    for entry in benchmark_data:
        score = SequenceMatcher(None, user_query.lower(), entry['query'].lower()).ratio()
        if score > highest_score:
            highest_score = score
            best_match = entry
    return best_match if highest_score > 0.3 else None

def main():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ Error: {BENCHMARK_FILE} missing.")
        return
    with open(BENCHMARK_FILE, "r") as f:
        benchmark_data = json.load(f)

    print("\n" + "="*50)
    print(" 🚀 LOCAL LLM BENCHMARK (Student: phi3.5 | Judge: Llama 3.3)")
    print("="*50)

    user_query = input("👉 Enter your Query: ").strip()
    if not user_query: return

    # 1. Match Ground Truth
    gt_entry = find_matching_ground_truth(user_query, benchmark_data)
    if not gt_entry:
        print("❌ No matching Ground Truth found. Try a query close to your test cases.")
        return
    print(f"✅ Matched Ground Truth ID: {gt_entry.get('id')}")

    # 2. Run Local RAG (This makes the network request to your RTX 3050)
    print(f"⏳ Sending request to Local LLM...")
    start_time = time.time()
    
    try:
        rag_output = generate_guide_from_rag(user_query)
        latency = time.time() - start_time
    except Exception as e:
        print(f"❌ Local LLM Failed: {e}")
        return

    # 3. Fetch Contexts for Fact Checking
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(user_query)
    contexts = [doc.page_content for doc in retrieved_docs]

    # Format output text
    if isinstance(rag_output, dict) and 'steps' in rag_output:
        rag_text = " ".join([s['instruction'] for s in rag_output['steps']])
    else:
        rag_text = str(rag_output)

    # 4. Grading
    print("⚖️  Judge is Grading...")
    llm_grade = run_llm_judge(user_query, gt_entry['ground_truth'], rag_text)
    faith_score = calculate_faithfulness(rag_text, contexts)
    rel_score = calculate_relevancy(user_query, rag_text)

    # 5. Report
    print("\n" + "="*60)
    print(f"📊 REPORT CARD")
    print("="*60)
    print(f"❓ QUERY: {user_query}")
    print("-" * 60)
    print(f"🤖 LOCAL MODEL OUTPUT ({latency:.2f}s):")
    print(f"   {rag_text[:300]}...") 
    print("-" * 60)
    print(f"🏆 SCORES:")
    print(f"   1. ACCURACY (Judge):    {llm_grade['total_score']} / 100")
    print(f"      Critique:            {llm_grade['reasoning']}")
    print("-" * 30)
    print(f"   2. METRICS:")
    print(f"      Faithfulness:        {faith_score:.2f} (1.0 = No Hallucination)")
    print(f"      Relevancy:           {rel_score:.2f} (1.0 = Perfect Answer)")
    print("="*60)

if __name__ == "__main__":
    main()