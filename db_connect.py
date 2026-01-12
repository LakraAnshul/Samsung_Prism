# --- 3. DATABASE CONNECTION ---
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database folder '{DB_PATH}' not found.")
        sys.exit(1)
        
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # CHANGE 1: Increased k from 3 to 7
    # This means it will fetch the top 7 pages relevant to your query
    return vector_db.as_retriever(search_kwargs={"k": 7})

# --- 4. GENERATION PIPELINE ---
def generate_guide_from_rag(query):
    print(f"\n--- 🔍 Step 1: Searching Knowledge Base for: '{query}' ---")
    retriever = get_retriever()
    
    # Retrieve docs
    relevant_docs = retriever.invoke(query)
    
    # --- DEBUGGING: FULL PRINT ---
    print(f"--- 🧐 DEBUG: Retrieved {len(relevant_docs)} chunks from DB ---")
    for i, doc in enumerate(relevant_docs):
        filename = doc.metadata.get('filename', 'Unknown')
        
        # CHANGE 2: Removed [:200]. Now printing the FULL text content.
        print(f"\n[Chunk {i+1} | Source: {filename}]")
        print(f"{'-'*20}")
        print(doc.page_content) 
        print(f"{'-'*20}")
    # -----------------------------------------------

    if not relevant_docs:
        return {"error": "No relevant info found in manuals."}
    
    # ... Rest of the function remains the same ...