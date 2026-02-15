import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_store"

def get_retriever():
    """
    Initializes and returns the ChromaDB retriever.
    """
    # 1. Verify Database Exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        print("   Please run your ingestion script (e.g., ingest.py) first.")
        sys.exit(1)

    # 2. Initialize Embeddings
    # We use the same model used for creating the database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Connect to ChromaDB
    try:
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        
        # Return retriever with search settings (k=7 chunks)
        return vector_db.as_retriever(search_kwargs={"k": 7})
        
    except Exception as e:
        print(f"❌ Error connecting to Database: {e}")
        sys.exit(1)

# Optional: Test block to verify connection when running this file directly
if __name__ == "__main__":
    print("Testing Database Connection...")
    retriever = get_retriever()
    print("✅ Connection Successful!")