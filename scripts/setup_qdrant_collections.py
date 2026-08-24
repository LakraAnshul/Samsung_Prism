import sys
from qdrant_client import QdrantClient, models

COLLECTIONS = [
    "washing_machines",
    "refrigerators",
    "air_conditioners",
    "dishwashers",
    "microwave_ovens"
]

def setup_qdrant_collections():
    client = QdrantClient(url="http://localhost:6333")
    
    try:
        # Check connection
        client.get_collections()
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        sys.exit(1)

    for collection_name in COLLECTIONS:
        if client.collection_exists(collection_name):
            print(f"[SKIP] Collection already exists: {collection_name}")
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
            print(f"[CREATED] {collection_name}")

    print("\n--- Verification ---")
    verify_collections(client)

def verify_collections(client):
    try:
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]
    except Exception as e:
        print(f"Failed to get collections for verification: {e}")
        return

    print("Qdrant connection works.")
    
    all_exist = all(c in existing_names for c in COLLECTIONS)
    if all_exist:
        print("All 5 collections exist.")
    else:
        print("Not all collections exist.")
    
    for collection_name in COLLECTIONS:
        if collection_name in existing_names:
            info = client.get_collection(collection_name)
            
            print(f"\nCollection: {collection_name}")
            
            # Dense
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                dense_config = vectors_config["dense"]
                size = getattr(dense_config, 'size', None)
                distance = getattr(dense_config, 'distance', None)
                print(f"  [OK] Dense vector 'dense' found: size={size}, distance={distance}")
                if size == 1024 and distance == models.Distance.COSINE:
                    print("  [OK] Dense vector configuration is correct.")
                else:
                    print("  [ERROR] Dense vector configuration is incorrect.")
            else:
                print(f"  [WARNING] Could not parse dense vector 'dense' from dictionary. Raw: {vectors_config}")

            # Sparse
            sparse_config = info.config.params.sparse_vectors
            if isinstance(sparse_config, dict) and "sparse" in sparse_config:
                print("  [OK] Sparse vector 'sparse' exists.")
            else:
                print(f"  [WARNING] Could not parse sparse vector 'sparse' from dictionary. Raw: {sparse_config}")

if __name__ == "__main__":
    setup_qdrant_collections()
