import json
from langchain_ollama import ChatOllama

# CONFIGURATION
# 🔴 REPLACE THIS with the IP of the RTX 3050 Laptop
SERVER_IP = "10.159.195.210"  
SERVER_PORT = "11434"

def ask_remote_ai(query):
    print(f"--- 📡 Connecting to GuideWeave Server at {SERVER_IP}... ---")
    
    # We connect to the remote URL instead of localhost
    llm = ChatOllama(
        base_url=f"http://{SERVER_IP}:{SERVER_PORT}",
        model="phi3.5",  # Must match the model pulled on the Server
        temperature=0.1,
        format="json"
    )
    
    prompt = f"""
    You are a helpful AI.
    USER QUERY: {query}
    OUTPUT: JSON format with 'answer' field.
    """
    
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        return {"error": f"Connection Failed. Is the Server Laptop on? Error: {e}"}

if __name__ == "__main__":
    q = "Hello, are you running on the RTX 3050?"
    res = ask_remote_ai(q)
    print(json.dumps(res, indent=2))