import os
import sys
# We use try-except to give friendly error messages if libraries are missing
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings # Updated import for better stability
except ImportError as e:
    print(f"❌ Library Error: {e}")
    print("Run this command: pip install langchain-community langchain-huggingface chromadb pypdf sentence-transformers")
    sys.exit(1)

# CONFIGURATION
PDF_DIRECTORY = "./Knowledge_Base/text"
DB_PATH = "./chroma_db_store"

def create_vector_db():
    print(f"--- 🚀 Starting Knowledge Base Ingestion ---")
    
    # 1. CHECK: Does the directory exist?
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"⚠️ Created folder {PDF_DIRECTORY}.")
        print(f"❌ ACTION REQUIRED: Please put your PDF manuals inside '{PDF_DIRECTORY}' and run this script again.")
        return

    # 2. CHECK: Are there any PDF files?
    files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]
    if not files:
        print(f"❌ No PDF files found in {PDF_DIRECTORY}.")
        print("Please add at least one .pdf file.")
        return

    print(f"✅ Found {len(files)} PDFs: {files}")

    # 3. Load all PDFs
    documents = []
    for pdf_file in files:
        file_path = os.path.join(PDF_DIRECTORY, pdf_file)
        print(f"   📄 Processing: {pdf_file}...")
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata so the AI knows which manual this came from
            for doc in docs:
                doc.metadata["filename"] = pdf_file
                doc.metadata["category"] = "User Manual"
            
            documents.extend(docs)
        except Exception as e:
            print(f"   ⚠️ Warning: Could not read {pdf_file}. Error: {e}")
            continue

    if not documents:
        print("❌ Error: No text could be extracted from the PDFs.")
        return

    # 4. Split Text into Chunks
    print(f"--- ✂️  Splitting text into chunks... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    # 5. Create Embeddings & Store in ChromaDB
    print("--- 🧠 Generating Embeddings (This will take a moment)... ---")
    
    # Using the updated HuggingFace class
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Save to disk
    if os.path.exists(DB_PATH):
        print(f"   ℹ️  Updating existing database at {DB_PATH}...")
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=DB_PATH
    )
    
    print(f"\n--- 🎉 SUCCESS! Knowledge Base saved to {DB_PATH} ---")
    print(f"You can now run 'python main.py'")

if __name__ == "__main__":
    create_vector_db()