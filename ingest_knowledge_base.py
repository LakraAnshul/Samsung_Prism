import os
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ Library Error: {e}")
    print("Run: pip install langchain-community langchain-huggingface chromadb pymupdf sentence-transformers")
    sys.exit(1)

# CONFIGURATION
PDF_DIRECTORY = "./Knowledge_Base/text"
DB_PATH = "./chroma_db_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# KNOWN SAMSUNG WASHING MACHINE MODELS
KNOWN_MODELS = ["WA5471ABP", "WF5M5100AW", "WF350ANR", "DC68", "WW90T504DAN", "WD80T654DBX"]

def extract_model_from_filename(filename: str) -> str:
    """Extracts Samsung model name from filename using known models or regex patterns."""
    for model in KNOWN_MODELS:
        if model.lower() in filename.lower():
            return model
    
    # Generic regex for Samsung appliance models
    match = re.search(r"\b(?:WA|WW|WD|WF|DC|DV|WT)\d+[A-Za-z0-9-]*\b", filename, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    
    return "General"

def extract_documents_from_pdfs(pdf_dir: str):
    """
    Extracts text from all PDFs with page-level tracking and model tagging.
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        print(f"⚠️ Created folder '{pdf_dir}'. Please add PDF manuals and re-run.")
        return []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_dir}'.")
        return []

    print(f"✅ Found {len(pdf_files)} PDF manuals.")
    documents = []

    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        model = extract_model_from_filename(pdf_file)
        print(f"   📄 Processing: {pdf_file} (Model: {model})")

        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text")
                if not page_text.strip():
                    continue

                metadata = {
                    "source": pdf_path,
                    "filename": pdf_file,
                    "model": model,
                    "page_number": page_idx + 1,
                    "total_pages": len(doc),
                    "category": "User Manual"
                }
                documents.append(Document(page_content=page_text, metadata=metadata))
            doc.close()
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to parse '{pdf_file}': {e}")
            continue

    return documents

def create_vector_db():
    print("--- 🚀 Starting Knowledge Base Ingestion ---")
    
    # 1. Extract documents with page-level metadata
    raw_docs = extract_documents_from_pdfs(PDF_DIRECTORY)
    if not raw_docs:
        print("❌ Error: No text could be extracted.")
        return

    # 2. Chunk text preserving metadata
    print("--- ✂️  Splitting text into structured chunks... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✅ Created {len(chunks)} text chunks across {len(raw_docs)} pages.")

    # 3. Generate Embeddings & Store in ChromaDB
    print(f"--- 🧠 Generating Embeddings with '{EMBEDDING_MODEL_NAME}'... ---")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(DB_PATH):
        print(f"   ℹ️  Updating existing database at '{DB_PATH}'...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print(f"\n--- 🎉 SUCCESS! Knowledge Base saved to '{DB_PATH}' with model tagging & page tracking ---")

if __name__ == "__main__":
    create_vector_db()