import os
import sys
import json
import argparse
from pathlib import Path
import requests
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config():
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "backend" / ".env")
    load_dotenv()
    api_key = os.environ.get("JINA_API_KEY")
    model = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-omni-small")
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    
    if not api_key:
        print("Error: JINA_API_KEY is missing from .env")
        sys.exit(1)
        
    return {
        "JINA_API_KEY": api_key,
        "JINA_EMBEDDING_MODEL": model,
        "QDRANT_URL": url
    }

_CLIENT_CACHE = {}
_VALIDATED_COLLECTIONS = set()
_SESSION = requests.Session()
_EMBEDDING_CACHE = {}

def connect_qdrant(url):
    if url not in _CLIENT_CACHE:
        try:
            _CLIENT_CACHE[url] = QdrantClient(url=url)
        except Exception as e:
            print(f"Error: Could not connect to Qdrant at {url}: {e}")
            sys.exit(1)
    return _CLIENT_CACHE[url]

def validate_image_collection(client, collection_name):
    if collection_name in _VALIDATED_COLLECTIONS:
        return
    if not client.collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist.")
        sys.exit(1)
        
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
        print("Error: Expected named vector 'dense' in existing collection.")
        sys.exit(1)
    _VALIDATED_COLLECTIONS.add(collection_name)

def validate_query(query):
    if query is None or not query.strip():
        print("Error: Query cannot be empty.")
        sys.exit(1)
    return query.strip()

def embed_query(query, model, api_key, retries=3):
    cache_key = (query, model)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "input": [query]
    }
    
    import time
    for attempt in range(retries):
        try:
            response = _SESSION.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            emb = response.json()["data"][0]["embedding"]
            _EMBEDDING_CACHE[cache_key] = emb
            return emb
        except requests.exceptions.HTTPError as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    [API Error] HTTP Error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [API Permanent Failure] HTTP Error")
                sys.exit(1)
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    [API Error] {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [API Permanent Failure] {e}")
                sys.exit(1)

def build_filter(appliance_type=None, brand=None, model=None, document_id=None, problem_id=None, step_id=None):
    must_conditions = []
    if appliance_type:
        must_conditions.append(FieldCondition(key="appliance_type", match=MatchValue(value=appliance_type)))
    if brand:
        must_conditions.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    if model:
        must_conditions.append(FieldCondition(key="model", match=MatchValue(value=model)))
    if document_id:
        must_conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    if problem_id:
        must_conditions.append(FieldCondition(key="problem_id", match=MatchValue(value=problem_id)))
    if step_id:
        must_conditions.append(FieldCondition(key="step_id", match=MatchValue(value=step_id)))
        
    if not must_conditions:
        return None
    return Filter(must=must_conditions)

def search_images(client, collection_name, query_vector, q_filter, limit):
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense",
        query_filter=q_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    return results.points

def verify_image_paths(points):
    for p in points:
        file_path = p.payload.get("file_path")
        if file_path:
            p.payload["file_exists"] = os.path.exists(file_path)
    return points

def rank_results(points, requested_step_id=None, exact_step_boost=0.02):
    unique_points = {}
    for p in points:
        image_id = p.payload.get("image_id")
        if not image_id:
            continue
        if image_id not in unique_points or p.score > unique_points[image_id].score:
            unique_points[image_id] = p
            
    ranked = []
    for p in unique_points.values():
        semantic_score = p.score
        step_match = False
        final_score = semantic_score
        
        if requested_step_id and p.payload.get("step_id") == requested_step_id:
            step_match = True
            final_score += exact_step_boost
            
        ranked.append({
            "point": p,
            "semantic_score": semantic_score,
            "step_match": step_match,
            "final_score": final_score
        })
        
    # Sort deterministically
    ranked.sort(key=lambda x: (
        -x["final_score"],
        -x["step_match"],
        x["point"].payload.get("image_id", "")
    ))
    
    return ranked

def retrieve_images(query, appliance_type="washing_machine", brand="Samsung", model=None, 
                    document_id=None, problem_id=None, step_id=None, top_k=5, 
                    search_top_k=20, fallback_enabled=True, min_filtered_results=3,
                    retrieval_mode="model_specific"):
    """
    Semantic image retrieval with image scope tagging.

    Modes:
        - "model_specific": Attempts model-specific retrieval with semantic fallback if needed.
        - "generic": Intentionally generic retrieval across appliance type.
    """
    query = validate_query(query)
    
    if top_k <= 0:
        print("Error: top_k must be positive.")
        sys.exit(1)
        
    config = load_config()
    client = connect_qdrant(config["QDRANT_URL"])
    collection_name = "washing_machine_images"
    validate_image_collection(client, collection_name)
    
    query_vector = embed_query(query, config["JINA_EMBEDDING_MODEL"], config["JINA_API_KEY"])
    
    all_candidates = []
    fallback_used = False
    fallback_reason = None
    
    # Generic Mode: Do not filter by model
    if retrieval_mode == "generic" or not model:
        l_generic_filter = build_filter(appliance_type, brand, None)
        generic_res = search_images(client, collection_name, query_vector, l_generic_filter, search_top_k)
        all_candidates.extend(generic_res)
        default_scope = "generic"
    else:
        # Model Specific Mode:
        default_scope = "model_specific"
        # LEVEL 1: Strict Document/Problem/Step filter (if provided) + Appliance + Model
        if document_id or problem_id or step_id:
            strict_filter = build_filter(appliance_type, brand, model, document_id, problem_id, step_id)
            strict_res = search_images(client, collection_name, query_vector, strict_filter, search_top_k)
            all_candidates.extend(strict_res)
            
            if len(strict_res) < min_filtered_results and fallback_enabled:
                fallback_used = True
                fallback_reason = "Insufficient results for exact document/problem/step"
                # LEVEL 2: Fallback to Appliance + Model only
                l2_filter = build_filter(appliance_type, brand, model)
                l2_res = search_images(client, collection_name, query_vector, l2_filter, search_top_k)
                all_candidates.extend(l2_res)
                
                # If still not enough, LEVEL 3: Fallback to Appliance only
                if len(l2_res) < min_filtered_results and model:
                    fallback_reason = "Insufficient results for exact model and doc/problem/step"
                    l3_filter = build_filter(appliance_type, None, None)
                    l3_res = search_images(client, collection_name, query_vector, l3_filter, search_top_k)
                    all_candidates.extend(l3_res)
        else:
            # LEVEL 2: Appliance + Model
            l2_filter = build_filter(appliance_type, brand, model)
            l2_res = search_images(client, collection_name, query_vector, l2_filter, search_top_k)
            all_candidates.extend(l2_res)
            
            # LEVEL 3: Fallback to Appliance only
            if len(l2_res) < min_filtered_results and model and fallback_enabled:
                fallback_used = True
                fallback_reason = "Insufficient results for exact model"
                l3_filter = build_filter(appliance_type, None, None)
                l3_res = search_images(client, collection_name, query_vector, l3_filter, search_top_k)
                all_candidates.extend(l3_res)
            
    # Verify file paths
    all_candidates = verify_image_paths(all_candidates)
    
    # Rank and dedup
    ranked_candidates = rank_results(all_candidates, requested_step_id=step_id)
    
    final_results = ranked_candidates[:top_k]
    
    results_list = []
    for rank_idx, item in enumerate(final_results, start=1):
        p = item["point"].payload
        img_model = p.get("model")
        
        # Determine image scope
        if retrieval_mode == "generic" or not model:
            image_scope = "generic"
        elif img_model and model and img_model.upper() == model.upper():
            image_scope = "model_specific"
        else:
            image_scope = "generic"

        results_list.append({
            "rank": rank_idx,
            "image_id": p.get("image_id"),
            "file_path": p.get("file_path"),
            "semantic_score": item["semantic_score"],
            "step_match": item["step_match"],
            "final_score": item["final_score"],
            "image_scope": image_scope,
            "document_id": p.get("document_id"),
            "problem_id": p.get("problem_id"),
            "problem_name": p.get("problem_name"),
            "step_id": p.get("step_id"),
            "step_number": p.get("step_number"),
            "step_text": p.get("step_text"),
            "dense_caption": p.get("dense_caption"),
            "detected_objects": p.get("detected_objects", []),
            "appliance_type": p.get("appliance_type"),
            "brand": p.get("brand"),
            "model": p.get("model")
        })
        
    return {
        "query": query,
        "filters": {
            "appliance_type": appliance_type,
            "brand": brand,
            "model": model if retrieval_mode == "model_specific" else None,
            "document_id": document_id,
            "problem_id": problem_id,
            "step_id": step_id
        },
        "retrieval": {
            "search_top_k": search_top_k,
            "final_top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason
        },
        "results": results_list
    }

def print_cli_output(result_obj):
    print("==================================================")
    print("GUIDE WEAVE — STAGE 7B")
    print("SEMANTIC IMAGE RETRIEVAL")
    print("==================================================")
    print(f"\nQuery:\n    {result_obj['query']}")
    print("\nCollection:\n    washing_machine_images")
    
    config = load_config()
    print(f"\nModel:\n    {config['JINA_EMBEDDING_MODEL']}")
    
    print("\nFilters:")
    for k, v in result_obj['filters'].items():
        if v:
            print(f"    {k} = {v}")
            
    r_meta = result_obj['retrieval']
    print(f"\nRetrieval mode:\n    {r_meta.get('retrieval_mode', 'model_specific')}")
    print(f"\nSearch candidates:\n    {r_meta['search_top_k']}")
    
    results = result_obj['results']
    print(f"\nFinal results:\n    {len(results)}")
    
    fallback_used = r_meta['fallback_used']
    print(f"\nFallback used:\n    {str(fallback_used).lower()}")
    if fallback_used and r_meta.get("fallback_reason"):
        print(f"    Reason: {r_meta['fallback_reason']}")
        
    print("\n==================================================")
    
    if not results:
        print("No relevant images found.")
        return
        
    for res in results:
        print("\n--------------------------------------------------")
        print(f"Rank: {res['rank']}\n")
        print(f"Image:\n    {res.get('image_id')}\n")
        print(f"Path:\n    {res.get('file_path')}\n")
        print(f"Scope:\n    {res.get('image_scope')}\n")
        print(f"Semantic score:\n    {res['semantic_score']:.4f}\n")
        print(f"Step match:\n    {str(res['step_match']).lower()}\n")
        print(f"Problem:\n    {res.get('problem_name')}\n")
        print(f"Step:\n    {res.get('step_text')}\n")
        print(f"Step ID:\n    {res.get('step_id')}\n")
        print(f"Caption:\n    {res.get('dense_caption')}\n")
        print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Semantic image retrieval")
    parser.add_argument("--query", type=str, help="The query string")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--appliance-type", type=str, default="washing_machine")
    parser.add_argument("--brand", type=str, default="Samsung")
    parser.add_argument("--model", type=str)
    parser.add_argument("--generic", action="store_true", help="Run generic image retrieval")
    parser.add_argument("--document-id", type=str)
    parser.add_argument("--problem-id", type=str)
    parser.add_argument("--step-id", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-fallback", action="store_true")
    
    args = parser.parse_args()
    
    if args.interactive:
        while True:
            try:
                q = input("\nQuery > ")
            except EOFError:
                break
                
            q = q.strip()
            if q.lower() in ["exit", "quit"]:
                break
            if not q:
                continue
                
            retrieval_mode = "generic" if args.generic else "model_specific"
            res = retrieve_images(q, model=args.model, retrieval_mode=retrieval_mode)
            print_cli_output(res)
    elif args.query:
        fallback_enabled = not args.no_fallback
        retrieval_mode = "generic" if args.generic else "model_specific"
        res = retrieve_images(
            args.query, 
            appliance_type=args.appliance_type,
            brand=args.brand,
            model=args.model,
            document_id=args.document_id,
            problem_id=args.problem_id,
            step_id=args.step_id,
            top_k=args.top_k,
            fallback_enabled=fallback_enabled,
            retrieval_mode=retrieval_mode
        )
        print_cli_output(res)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
