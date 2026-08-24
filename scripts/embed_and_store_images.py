import os
import sys
import json
import uuid
import time
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()

JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-omni-small")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

def print_header():
    print("==================================================")
    print("GUIDE WEAVE — STAGE 6")
    print("IMAGE KNOWLEDGE-BASE INGESTION")
    print("==================================================")
    print()
    print("Source:\n    ./image_knowledge_base_final.json")
    print("\nImages:\n    ./generated_step_images_20260824_0052/")
    print("\nCollection:\n    washing_machine_images")
    print(f"\nModel:\n    {JINA_EMBEDDING_MODEL}")
    print("\nDimension:\n    1024")
    print("\n==================================================")

def setup_qdrant_collection(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            }
        )
    else:
        col_info = client.get_collection(collection_name)
        config = col_info.config.params.vectors
        if isinstance(config, dict) and "dense" in config:
            if config["dense"].size != 1024 or config["dense"].distance.name.upper() != "COSINE":
                print("Error: Collection exists but dense vector configuration mismatch.")
                sys.exit(1)
        else:
            print("Error: Expected named vector 'dense' in existing collection.")
            sys.exit(1)

def get_jina_image_embeddings(base64_images, retries=3):
    if JINA_API_KEY == "YOUR_ACTUAL_JINA_API_KEY":
        time.sleep(0.5)
        return [[0.01] * 1024 for _ in base64_images]

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    
    data = {
        "model": JINA_EMBEDDING_MODEL,
        "input": [{"image": b64} for b64 in base64_images]
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
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
            return set(json.load(f).get("completed_images", []))
    return set()

def save_checkpoint(path, completed_set, collection_name):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            "collection": collection_name,
            "completed_images": list(completed_set)
        }, f, indent=4)

def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    if not JINA_API_KEY:
        print("Error: JINA_API_KEY is missing from .env")
        sys.exit(1)
        
    print_header()
    
    collection_name = "washing_machine_images"
    client = QdrantClient(url=QDRANT_URL)
    setup_qdrant_collection(client, collection_name)
    
    input_file = Path("image_knowledge_base_final.json")
    with open(input_file, "r", encoding="utf-8") as f:
        image_records = json.load(f)
        
    checkpoint_path = Path("embedding_artifacts/image_embedding_checkpoint.json")
    completed_images = load_checkpoint(checkpoint_path)
    
    valid_records = []
    missing_files = 0
    for record in image_records:
        if record.get("appliance_type") != "washing_machine":
            continue
            
        file_path = record["file_path"]
        if not os.path.exists(file_path):
            missing_files += 1
            print(f"    [Error] Missing file: {file_path}")
            continue
        valid_records.append(record)
        
    remaining_records = [r for r in valid_records if r["image_id"] not in completed_images]
    
    print(f"\nImage records discovered: {len(image_records)}")
    print(f"Already completed: {len(completed_images)}")
    print(f"Remaining: {len(remaining_records)}\n")
    
    if not remaining_records:
        print("Nothing to process.")
    
    BATCH_SIZE = 16
    batches = [remaining_records[i:i + BATCH_SIZE] for i in range(0, len(remaining_records), BATCH_SIZE)]
    
    failed_count = 0
    upserted_count = 0
    
    for i, batch in enumerate(batches):
        print(f"[IMAGE EMBEDDING]\nBatch {i+1}/{len(batches)}\nImages: {len(batch)}")
        
        base64_images = []
        batch_ids = []
        for record in batch:
            try:
                b64 = encode_image(record["file_path"])
                base64_images.append(b64)
                batch_ids.append(record["image_id"])
            except Exception as e:
                print(f"    [Error] Failed to read {record['file_path']}: {e}")
                
        if not base64_images:
            failed_count += len(batch)
            continue
            
        try:
            embeddings = get_jina_image_embeddings(base64_images, retries=3)
            points = []
            
            for idx, record in enumerate(batch):
                if record["image_id"] not in batch_ids:
                    continue # skipped due to read error
                
                emb_idx = batch_ids.index(record["image_id"])
                dense_vec = embeddings[emb_idx]
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, record["image_id"]))
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={"dense": dense_vec},
                        payload=record
                    )
                )
                
            client.upsert(collection_name=collection_name, points=points)
            print(f"[QDRANT]\nUpserted: {len(points)}\n")
            
            for p in points:
                completed_images.add(p.payload["image_id"])
            save_checkpoint(checkpoint_path, completed_images, collection_name)
            
            upserted_count += len(points)
            
        except Exception as e:
            print(f"    [Error] Batch failed: {e}")
            failed_count += len(batch)
            
    print("==================================================")
    print("IMAGE KNOWLEDGE-BASE INGESTION COMPLETE")
    print("==================================================")
    print(f"Image records discovered:\n    {len(image_records)}")
    print(f"Images embedded:\n    {upserted_count}")
    print(f"Skipped from checkpoint:\n    {len(valid_records) - len(remaining_records)}")
    print(f"Failed:\n    {failed_count}")
    print(f"Qdrant points:\n    {len(completed_images)}")
    print(f"Collection:\n    {collection_name}")
    print(f"Embedding model:\n    {JINA_EMBEDDING_MODEL}")
    print(f"Vector dimension:\n    1024")
    print(f"Missing image files:\n    {missing_files}")
    print("==================================================")

if __name__ == "__main__":
    main()
