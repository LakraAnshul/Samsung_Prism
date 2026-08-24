import os
import json
import uuid
import sys
import time
import re
from pathlib import Path
import requests
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

load_dotenv()

JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.environ.get("JINA_EMBEDDING_MODEL")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

if not JINA_API_KEY:
    print("Error: JINA_API_KEY is missing from .env")
    sys.exit(1)
if not JINA_EMBEDDING_MODEL:
    print("Error: JINA_EMBEDDING_MODEL is missing from .env")
    sys.exit(1)

print("==================================================")
print("GUIDE WEAVE — STAGE 4")
print("EMBEDDINGS + QDRANT INGESTION")
print("==================================================")
print()
print(f"Qdrant:\n    {QDRANT_URL}")
print("\nCollection:\n    washing_machines")
print(f"\nDense model:\n    {JINA_EMBEDDING_MODEL}")
print("\nDense dimension:\n    1024")
print("\nSparse:\n    enabled")
print("\nInput:\n    chunked_ground_truth/")
print("\n==================================================")

class SparseTokenizer:
    def __init__(self, vocab_path):
        self.vocab_path = Path(vocab_path)
        self.vocab = {}
        self.next_id = 1
        if self.vocab_path.exists():
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocab = data.get("vocab", {})
                self.next_id = data.get("next_id", 1)
                
    def save(self):
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab": self.vocab,
                "next_id": self.next_id
            }, f, indent=4)
            
    def tokenize(self, text):
        tokens = re.findall(r'[a-zA-Z0-9_\-]+', text)
        result = {}
        for t in tokens:
            t_lower = t.lower()
            if t_lower not in self.vocab:
                self.vocab[t_lower] = self.next_id
                self.next_id += 1
            idx = self.vocab[t_lower]
            result[idx] = result.get(idx, 0) + 1
        return result

def get_jina_embeddings(texts, retries=3):
    if JINA_API_KEY == "YOUR_ACTUAL_JINA_API_KEY":
        time.sleep(0.5)
        return [[0.01] * 1024 for _ in texts]
        
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    data = {
        "model": JINA_EMBEDDING_MODEL,
        "input": texts
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            return [item["embedding"] for item in res_json["data"]]
        except requests.exceptions.HTTPError as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    [API Error] {e} | Detail: {e.response.text}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [API Permanent Failure] {e} | Detail: {e.response.text}")
                raise
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    [API Error] {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [API Permanent Failure] {e}")
                raise

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(json.load(f).get("completed_chunks", []))
    return set()

def save_checkpoint(path, completed_set, collection_name):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            "collection": collection_name,
            "completed_chunks": list(completed_set)
        }, f, indent=4)

def main():
    client = QdrantClient(url=QDRANT_URL)
    collection_name = "washing_machines"
    
    try:
        col_info = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Could not retrieve collection '{collection_name}': {e}")
        sys.exit(1)
        
    config = col_info.config.params.vectors
    if isinstance(config, dict):
        if "dense" not in config:
            print("Error: Dense vector configuration not found.")
            sys.exit(1)
        dense_conf = config["dense"]
        if dense_conf.size != 1024 or dense_conf.distance.name.upper() != "COSINE":
            print(f"Error: Dense vector configuration mismatch. Expected 1024, Cosine. Got {dense_conf.size}, {dense_conf.distance.name}.")
            sys.exit(1)
    else:
        print("Error: Expected multiple named vectors configuration.")
        sys.exit(1)
        
    sparse_config = col_info.config.params.sparse_vectors
    if not sparse_config or "sparse" not in sparse_config:
        print("Error: Sparse vector configuration named 'sparse' not found.")
        sys.exit(1)
        
    input_dir = Path("chunked_ground_truth")
    json_files = list(input_dir.rglob("*.json"))
    
    child_chunks = []
    documents_count = len(json_files)
    
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get("appliance_type") != "washing_machine":
                continue
            for c in data.get("children", []):
                child_chunks.append(c)
                
    checkpoint_path = Path("embedding_artifacts/embedding_checkpoint.json")
    completed_chunks = load_checkpoint(checkpoint_path)
    
    remaining_chunks = [c for c in child_chunks if c["chunk_id"] not in completed_chunks]
    
    print(f"\nDocuments discovered: {documents_count}")
    print(f"Child chunks discovered: {len(child_chunks)}")
    print(f"Already completed: {len(completed_chunks)}")
    print(f"Remaining: {len(remaining_chunks)}\n")
    
    if not remaining_chunks:
        print("Nothing to process.")
        return
        
    sparse_tokenizer = SparseTokenizer("embedding_artifacts/sparse_vocabulary.json")
    
    BATCH_SIZE = 32
    batches = [remaining_chunks[i:i + BATCH_SIZE] for i in range(0, len(remaining_chunks), BATCH_SIZE)]
    
    api_calls = 0
    total_upserted = 0
    failed = 0
    
    for i, batch in enumerate(batches):
        print(f"[EMBEDDING]\nBatch {i+1}/{len(batches)}\nChunks: {len(batch)}")
        
        texts = []
        for c in batch:
            section = c.get("section", "")
            problem = c.get("problem_name", "")
            content = c.get("text", "")
            
            if not content.strip():
                print(f"    [Error] Empty chunk {c['chunk_id']}. Skipping.")
                content = "EMPTY_CHUNK" 
                
            approx_tokens = c.get("token_count", len(content.split()) * 1.3)
            if approx_tokens > 8000:
                print(f"    [Error] Chunk {c['chunk_id']} from {c.get('document_id')} is too large ({approx_tokens} tokens).")
                raise ValueError("Chunk too large for model context window.")
                
            emb_text = ""
            if section:
                emb_text += f"[Section]\n{section}\n\n"
            if problem:
                emb_text += f"[Problem]\n{problem}\n\n"
            emb_text += f"[Content]\n{content}"
            texts.append(emb_text)
            
        try:
            embeddings = get_jina_embeddings(texts)
            api_calls += 1
            
            points = []
            for idx, c in enumerate(batch):
                dense_vec = embeddings[idx]
                
                sparse_dict = sparse_tokenizer.tokenize(texts[idx])
                sparse_indices = list(sparse_dict.keys())
                sparse_values = list(sparse_dict.values())
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, c["document_id"] + "_" + c["chunk_id"]))
                
                payload = {
                    "document_id": c.get("document_id"),
                    "appliance_type": c.get("appliance_type"),
                    "brand": c.get("brand"),
                    "model": c.get("model"),
                    "source_file": c.get("source_file"),
                    "section": c.get("section"),
                    "problem_id": c.get("problem_id"),
                    "problem_name": c.get("problem_name"),
                    "chunk_id": c["chunk_id"],
                    "parent_chunk_id": c.get("parent_chunk_id"),
                    "chunk_level": c.get("chunk_level", "child"),
                    "chunk_type": c.get("chunk_type"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "step_start": c.get("step_start"),
                    "step_end": c.get("step_end"),
                    "steps": c.get("steps", []),
                    "text": c.get("text", "")
                }
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vec,
                            "sparse": SparseVector(indices=sparse_indices, values=sparse_values)
                        },
                        payload=payload
                    )
                )
                
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            print(f"[QDRANT]\nUpserted: {len(points)}\n")
            
            for c in batch:
                completed_chunks.add(c["chunk_id"])
            save_checkpoint(checkpoint_path, completed_chunks, collection_name)
            sparse_tokenizer.save()
            total_upserted += len(points)
            
        except Exception as e:
            print(f"Batch failed: {e}")
            failed += len(batch)
            continue
            
    print("==================================================")
    print(f"Documents processed: {documents_count}")
    print(f"Child chunks processed: {total_upserted}")
    print(f"Embedding API calls/batches: {api_calls}")
    print(f"Qdrant points: {len(completed_chunks)}")
    print(f"Skipped from checkpoint: {len(child_chunks) - len(remaining_chunks)}")
    print(f"Failed: {failed}")
    print("Dense dimension: 1024")
    print("Sparse enabled: True")
    print(f"Collection: {collection_name}")
    print("==================================================")

if __name__ == "__main__":
    main()
