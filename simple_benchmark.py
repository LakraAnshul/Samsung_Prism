import os
import json
import time
from difflib import SequenceMatcher
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings

# Import your RAG function
try:
    from main_local_llm import generate_guide_from_rag, get_retriever
except ImportError:
    print("❌ Critical: 'main1.py' not found.")
    exit()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BENCHMARK_FILE = "benchmark_data.json"

# --- CUSTOM METRIC 1: FAITHFULNESS (Fact Check) ---
def calculate_faithfulness(answer, context_list):
    """
    Checks if the Answer is derived ONLY from the retrieved Context.
    Returns a score 0.0 to 1.0
    """
    client = Groq(api_key=GROQ_API_KEY)
    context_text = "\n".join(context_list)[:15000] # Limit context size
    
    prompt = f"""
    You are a Fact Checker.
    
    CONTEXT:
    {context_text}
    
    STATEMENT:
    "{answer}"
    
    TASK:
    Analyze the STATEMENT. 
    Return a score (0.0 to 1.0) representing how much of the statement is explicitly supported by the CONTEXT.
    - 1.0: Every claim is in the text.
    - 0.0: The statement is completely hallucinated.
    
    OUTPUT JSON ONLY:
    {{ "score": 0.5, "reasoning": "Explain why" }}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return data.get("score", 0.0)
    except:
        return 0.0

# --- CUSTOM METRIC 2: RELEVANCY (On Topic) ---
def calculate_relevancy(query, answer):
    """
    Checks if the Answer actually addresses the User's Query.
    Returns a score 0.0 to 1.0
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    You are a Relevance Grader.
    
    USER QUERY: "{query}"
    GENERATED ANSWER: "{answer}"
    
    TASK:
    Rate how relevant the answer is to the query (0.0 to 1.0).
    - 1.0: Direct, perfect answer.
    - 0.0: Completely unrelated.
    
    OUTPUT JSON ONLY:
    {{ "score": 0.5 }}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return data.get("score", 0.0)
    except:
        return 0.0

# --- MAIN JUDGE (Your existing logic) ---
def run_llm_judge(query, ground_truth, student_answer):
    client = Groq(api_key=GROQ_API_KEY)
    gt_text = "\n".join([f"- {s}" for s in ground_truth])
    student_text = str(student_answer)
    
    prompt = f"""
    You are a strict QA Auditor.
    QUERY: "{query}"
    EXPECTED ANSWER (Ground Truth):
    {gt_text}
    ACTUAL ANSWER (System):
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
        return {"total_score": 0, "reasoning": "Error"}

def find_matching_ground_truth(user_query, benchmark_data):
    best_match = None
    highest_score = 0.0
    for entry in benchmark_data:
        score = SequenceMatcher(None, user_query.lower(), entry['query'].lower()).ratio()
        if score > highest_score:
            highest_score = score
            best_match = entry
    if highest_score > 0.3: return best_match
    return None

def main():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ Error: {BENCHMARK_FILE} missing.")
        return
    with open(BENCHMARK_FILE, "r") as f:
        benchmark_data = json.load(f)

    print("\n" + "="*50)
    print(" 🚀 SIMPLE BENCHMARK TOOL (STABLE VERSION)")
    print("="*50)

    user_query = input("👉 Enter your Query: ").strip()
    if not user_query: return

    gt_entry = find_matching_ground_truth(user_query, benchmark_data)
    if not gt_entry:
        print("❌ No matching Ground Truth found.")
        return
    
    print(f"✅ Matched Ground Truth ID: {gt_entry.get('id')}")

    print("⏳ Running RAG System...")
    rag_output = generate_guide_from_rag(user_query)
    
    # Get Contexts for Fact Checking
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(user_query)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    if isinstance(rag_output, dict) and 'steps' in rag_output:
        rag_text = " ".join([s['instruction'] for s in rag_output['steps']])
    else:
        rag_text = str(rag_output)

    print("⚖️  Running Grades (Judge + Metrics)...")
    
    # 1. Main Judge (Accuracy)
    llm_grade = run_llm_judge(user_query, gt_entry['ground_truth'], rag_text)
    
    # 2. Custom Metrics (Faithfulness & Relevancy)
    faith_score = calculate_faithfulness(rag_text, contexts)
    rel_score = calculate_relevancy(user_query, rag_text)

    # Final Report
    print("\n" + "="*60)
    print(f"📊 REPORT CARD")
    print("="*60)
    print(f"❓ QUERY: {user_query}")
    print("-" * 60)
    print(f"🤖 SYSTEM OUTPUT (Actual):")
    print(f"   {rag_text[:300]}...") 
    print("-" * 60)
    print(f"🏆 SCORES:")
    print(f"   1. LLM JUDGE:      {llm_grade['total_score']} / 100")
    print(f"      Critique:       {llm_grade['reasoning']}")
    print("-" * 30)
    print(f"   2. METRICS (Ragas-Style):")
    print(f"      Faithfulness:   {faith_score:.2f} (1.0 = No Hallucination)")
    print(f"      Relevancy:      {rel_score:.2f} (1.0 = Perfect Answer)")
    print("="*60)

if __name__ == "__main__":
    main()