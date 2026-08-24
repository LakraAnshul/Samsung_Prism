from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

points = client.scroll(
    collection_name="washing_machine_images",
    limit=3,
    with_payload=True,
    with_vectors=True
)[0]

for p in points:
    print(f"Point ID: {p.id}")
    print(f"Vector dim: {len(p.vector['dense'])}")
    payload = p.payload
    print(f"  image_id: {payload.get('image_id')}")
    print(f"  file_path: {payload.get('file_path')}")
    print(f"  model: {payload.get('model')}")
    print(f"  problem_name: {payload.get('problem_name')}")
    print(f"  step_number: {payload.get('step_number')}")
    print(f"  step_id: {payload.get('step_id')}")
    print(f"  dense_caption: {payload.get('dense_caption')[:100]}...\n")
