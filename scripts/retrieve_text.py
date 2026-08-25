import os
import sys
import json
import re
import argparse
from pathlib import Path
import requests
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rerank_text import rerank_documents, load_rerank_config
from backend.pipeline_logger import pipeline_logger


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
_TOKENIZER_CACHE = None

def connect_qdrant(url):
    if url not in _CLIENT_CACHE:
        try:
            _CLIENT_CACHE[url] = QdrantClient(url=url)
        except Exception as e:
            print(f"Error: Could not connect to Qdrant at {url}: {e}")
            sys.exit(1)
    return _CLIENT_CACHE[url]

def validate_collection(client, collection_name):
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
            print("Error: Dense vector configuration mismatch. Expected 1024, Cosine.")
            sys.exit(1)
    else:
        print("Error: Expected named vector 'dense' in existing collection.")
        sys.exit(1)
        
    sparse_config = col_info.config.params.sparse_vectors
    if not sparse_config or "sparse" not in sparse_config:
        print("Error: Sparse vector 'sparse' configuration not found.")
        sys.exit(1)
    _VALIDATED_COLLECTIONS.add(collection_name)

class SparseTokenizer:
    def __init__(self, vocab_path=None):
        if vocab_path is None:
            self.vocab_path = PROJECT_ROOT / "embedding_artifacts" / "sparse_vocabulary.json"
        else:
            p = Path(vocab_path)
            if p.is_dir():
                self.vocab_path = p / "sparse_vocabulary.json"
            else:
                self.vocab_path = p
                
        if not self.vocab_path.exists():
            print(f"Error: Sparse vocabulary not found at {self.vocab_path}")
            sys.exit(1)
            
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                self.vocab = data.get("vocab", {})
            except Exception as e:
                print(f"Error: Malformed sparse vocabulary at {self.vocab_path}: {e}")
                sys.exit(1)
                
    def tokenize(self, text):
        if not text or not isinstance(text, str):
            return {}
        tokens = re.findall(r'[a-zA-Z0-9_\-]+', text)
        result = {}
        for t in tokens:
            t_lower = t.lower()
            if t_lower in self.vocab:
                idx = self.vocab[t_lower]
                result[idx] = result.get(idx, 0) + 1
        return result

def embed_query(query, model, api_key, retries=3):
    cache_key = (query, model)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    formatted_query = f"Query:\n{query}"
    
    data = {
        "model": model,
        "input": [formatted_query]
    }
    
    import time
    for attempt in range(retries):
        try:
            response = _SESSION.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            emb = response.json()["data"][0]["embedding"]
            _EMBEDDING_CACHE[cache_key] = emb
            return emb
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"Error: Embedding API failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Details: {e.response.text}")
                sys.exit(1)

def build_filter(appliance_type, brand, model, document_id, problem_id):
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
        
    if not must_conditions:
        return None
        
    return Filter(must=must_conditions)

def dense_search(client, collection_name, query_vector, q_filter, limit):
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

def sparse_search(client, collection_name, sparse_dict, q_filter, limit):
    if not sparse_dict:
        return []
    indices = list(sparse_dict.keys())
    values = list(sparse_dict.values())
    if not indices:
        return []
        
    results = client.query_points(
        collection_name=collection_name,
        query=SparseVector(indices=indices, values=values),
        using="sparse",
        query_filter=q_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    return results.points

def reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60):
    scores = {}
    metadata = {}
    dense_ranks = {}
    sparse_ranks = {}
    dense_scores = {}
    sparse_scores = {}
    
    dense_results = dense_results or []
    sparse_results = sparse_results or []
    
    for rank, hit in enumerate(dense_results, start=1):
        if not hit.payload: continue
        chunk_id = hit.payload.get("chunk_id")
        if not chunk_id: continue
        scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (rrf_k + rank)
        metadata[chunk_id] = hit.payload
        dense_ranks[chunk_id] = rank
        dense_scores[chunk_id] = hit.score
        
    for rank, hit in enumerate(sparse_results, start=1):
        if not hit.payload: continue
        chunk_id = hit.payload.get("chunk_id")
        if not chunk_id: continue
        scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (rrf_k + rank)
        metadata[chunk_id] = hit.payload
        sparse_ranks[chunk_id] = rank
        sparse_scores[chunk_id] = hit.score
        
    # Sort deterministically
    sorted_items = sorted(
        scores.items(),
        key=lambda x: (
            -x[1], # RRF score descending
            dense_ranks.get(x[0], float('inf')), # dense rank ascending
            sparse_ranks.get(x[0], float('inf')), # sparse rank ascending
            x[0] # chunk_id ascending
        )
    )
    
    fused_results = []
    for rank, (chunk_id, rrf_score) in enumerate(sorted_items, start=1):
        payload = metadata[chunk_id]
        fused_results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "parent_chunk_id": payload.get("parent_chunk_id"),
            "document_id": payload.get("document_id"),
            "appliance_type": payload.get("appliance_type"),
            "brand": payload.get("brand"),
            "model": payload.get("model"),
            "problem_id": payload.get("problem_id"),
            "problem_name": payload.get("problem_name"),
            "step_start": payload.get("step_start"),
            "step_end": payload.get("step_end"),
            "steps": payload.get("steps", []),
            "page_start": payload.get("page_start"),
            "page_end": payload.get("page_end"),
            "text": payload.get("text"),
            
            "dense_score": dense_scores.get(chunk_id, None),
            "sparse_score": sparse_scores.get(chunk_id, None),
            "dense_rank": dense_ranks.get(chunk_id, None),
            "sparse_rank": sparse_ranks.get(chunk_id, None),
            "rrf_score": rrf_score
        })
        
    return fused_results

def retrieve_text(query, appliance_type="washing_machine", brand="Samsung", model=None, 
                  document_id=None, problem_id=None, dense_top_k=20, sparse_top_k=20, 
                  final_top_k=8, rrf_k=60, retrieval_mode="model_specific",
                  candidate_top_k=None, rerank=None):
    """
    Hybrid text retrieval with strict retrieval mode support and optional Jina reranking.

    Modes:
        - "model_specific": Filters strictly by model. NEVER removes model filter.
        - "generic": Filters by appliance_type and brand only (model=None).
    """
    if not query or not query.strip():
        print("Error: Query cannot be empty.")
        sys.exit(1)
        
    query = query.strip()
    
    config = load_config()
    rerank_config = load_rerank_config()
    is_rerank_enabled = rerank_config["RERANK_ENABLED"] if rerank is None else rerank
    effective_candidate_k = candidate_top_k or rerank_config["RERANK_CANDIDATE_K"]

    client = connect_qdrant(config["QDRANT_URL"])
    collection_name = "washing_machines"
    validate_collection(client, collection_name)
    
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = SparseTokenizer(PROJECT_ROOT / "embedding_artifacts" / "sparse_vocabulary.json")
    tokenizer = _TOKENIZER_CACHE
    sparse_dict = tokenizer.tokenize(query)
    
    import time
    qdrant_start = time.perf_counter()
    dense_vector = embed_query(query, config["JINA_EMBEDDING_MODEL"], config["JINA_API_KEY"])
    
    # Determine retrieval model filter based on retrieval_mode
    effective_model = model if retrieval_mode == "model_specific" else None
    
    q_filter = build_filter(appliance_type, brand, effective_model, document_id, problem_id)
    
    effective_dense_k = max(dense_top_k, effective_candidate_k) if is_rerank_enabled else dense_top_k
    effective_sparse_k = max(sparse_top_k, effective_candidate_k) if is_rerank_enabled else sparse_top_k

    dense_t0 = time.perf_counter()
    dense_res = dense_search(client, collection_name, dense_vector, q_filter, effective_dense_k)
    dense_time_ms = (time.perf_counter() - dense_t0) * 1000.0

    sparse_t0 = time.perf_counter()
    sparse_res = sparse_search(client, collection_name, sparse_dict, q_filter, effective_sparse_k)
    sparse_time_ms = (time.perf_counter() - sparse_t0) * 1000.0
    
    rrf_t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=rrf_k)
    rrf_time_ms = (time.perf_counter() - rrf_t0) * 1000.0
    qdrant_time_ms = (time.perf_counter() - qdrant_start) * 1000.0

    # Log dense, sparse, and RRF stages
    pipeline_logger.log_dense_retrieval(
        collection=collection_name,
        query=query,
        retrieval_mode=retrieval_mode,
        q_filter=q_filter,
        requested_limit=effective_dense_k,
        points=dense_res,
        latency_ms=dense_time_ms
    )
    
    pipeline_logger.log_sparse_retrieval(
        collection=collection_name,
        query=query,
        retrieval_mode=retrieval_mode,
        q_filter=q_filter,
        requested_limit=effective_sparse_k,
        points=sparse_res,
        matching_tokens_count=len(sparse_dict),
        reason_if_zero="No matching sparse tokens in vocabulary" if not sparse_dict else ("No matching points found in Qdrant" if not sparse_res else None),
        latency_ms=sparse_time_ms
    )

    pipeline_logger.log_rrf_fusion(
        rrf_k=rrf_k,
        dense_count=len(dense_res),
        sparse_count=len(sparse_res),
        fused_results=fused,
        candidate_pool_limit=effective_candidate_k,
        latency_ms=rrf_time_ms
    )
    
    ctx = pipeline_logger.get_context()
    if ctx:
        ctx.qdrant_latency_ms = round(qdrant_time_ms, 2)

    if is_rerank_enabled:
        candidate_pool = fused[:effective_candidate_k]
        pipeline_logger.log_reranker_input_pool(candidates=candidate_pool, top_k=final_top_k)
        final_results, rerank_meta = rerank_documents(
            query=query,
            candidates=candidate_pool,
            top_k=final_top_k,
            enabled=True
        )
    else:
        pipeline_logger.log_reranker_disabled()
        final_results = fused[:final_top_k]
        rerank_meta = {
            "enabled": False,
            "applied": False
        }
    
    retrieval_meta = {
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "candidate_top_k": effective_candidate_k if is_rerank_enabled else None,
        "final_top_k": final_top_k,
        "rrf_k": rrf_k,
        "retrieval_mode": retrieval_mode,
        "fallback_used": False,
        "reranking": rerank_meta,
        "qdrant_latency_ms": round(qdrant_time_ms, 2)
    }

    return {
        "query": query,
        "filters": {
            "appliance_type": appliance_type,
            "brand": brand,
            "model": effective_model,
            "document_id": document_id,
            "problem_id": problem_id
        },
        "retrieval": retrieval_meta,
        "results": final_results
    }

def print_cli_output(result_obj):
    print("==================================================")
    print("GUIDE WEAVE — STAGE 7A")
    print("HYBRID TEXT RETRIEVAL")
    print("==================================================")
    print(f"\nQuery:\n    {result_obj['query']}")
    print("\nFilters:")
    for k, v in result_obj['filters'].items():
        if v:
            print(f"    {k} = {v}")
            
    r_meta = result_obj['retrieval']
    print(f"\nRetrieval mode:\n    {r_meta.get('retrieval_mode', 'model_specific')}")
    print(f"\nDense candidates:\n    {r_meta['dense_top_k']}")
    print(f"\nSparse candidates:\n    {r_meta['sparse_top_k']}")
    if r_meta.get("candidate_top_k"):
        print(f"\nReranker Candidate Pool:\n    {r_meta['candidate_top_k']}")
    print(f"\nRRF K:\n    {r_meta['rrf_k']}")

    rerank_meta = r_meta.get("reranking", {})
    if rerank_meta.get("enabled"):
        print(f"\nReranker:\n    model = {rerank_meta.get('model')}\n    candidates = {rerank_meta.get('candidate_count')}\n    applied = {rerank_meta.get('applied')}")
        if "latency_ms" in rerank_meta:
            print(f"    latency = {rerank_meta.get('latency_ms')} ms")
        if rerank_meta.get("fallback"):
            print(f"    fallback = {rerank_meta.get('fallback')} (reason: {rerank_meta.get('reason')})")
    
    results = result_obj['results']
    print(f"\nFinal results:\n    {len(results)}")
    print("\n==================================================")
    
    if not results:
        print("No relevant troubleshooting chunks found.")
        return
        
    for res in results:
        print("\n--------------------------------------------------")
        print(f"Rank: {res['rank']}\n")
        print(f"Problem:\n    {res.get('problem_name')}\n")
        print(f"Problem ID:\n    {res.get('problem_id')}\n")
        print(f"Chunk ID:\n    {res.get('chunk_id')}\n")
        
        pg_start = res.get('page_start')
        pg_end = res.get('page_end')
        if pg_start and pg_end and pg_start != pg_end:
            print(f"Pages:\n    {pg_start}-{pg_end}\n")
        elif pg_start:
            print(f"Pages:\n    {pg_start}\n")
            
        step_s = res.get('step_start')
        step_e = res.get('step_end')
        if step_s and step_e and step_s != step_e:
            print(f"Step range:\n    {step_s}-{step_e}\n")
        elif step_s:
            print(f"Step range:\n    {step_s}\n")
            
        print(f"Dense rank:\n    {res['dense_rank']}\n")
        print(f"Dense score:\n    {res['dense_score']}\n")
        print(f"Sparse rank:\n    {res['sparse_rank']}\n")
        print(f"Sparse score:\n    {res['sparse_score']}\n")
        print(f"RRF score:\n    {res['rrf_score']:.6f}\n")
        if res.get('rerank_score') is not None:
            print(f"Rerank score:\n    {res['rerank_score']:.6f}\n")
            print(f"Rerank rank:\n    {res.get('rerank_rank')}\n")
            print(f"Original RRF rank:\n    {res.get('retrieval_rank')}\n")
        print(f"Text:\n    {res.get('text')}")
        print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Hybrid text retrieval for Guide Weave")
    parser.add_argument("--query", type=str, help="The query string")
    parser.add_argument("--model", type=str, help="Target appliance model (e.g., WA5471ABP/XAA)")
    parser.add_argument("--generic", action="store_true", help="Run generic retrieval without model filter")
    parser.add_argument("--no-rerank", action="store_true", help="Disable Jina reranking stage")
    parser.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for reranker")
    parser.add_argument("--top-k", type=int, default=8, help="Final top K evidence chunks")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    rerank_flag = False if args.no_rerank else None
    
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
            res = retrieve_text(
                q,
                model=args.model,
                retrieval_mode=retrieval_mode,
                final_top_k=args.top_k,
                candidate_top_k=args.candidate_k,
                rerank=rerank_flag
            )
            print_cli_output(res)
    elif args.query:
        if not args.model and not args.generic:
            print("Error: Specify either --model <MODEL_ID> or --generic for retrieval.")
            sys.exit(1)
        retrieval_mode = "generic" if args.generic else "model_specific"
        res = retrieve_text(
            args.query,
            model=args.model,
            retrieval_mode=retrieval_mode,
            final_top_k=args.top_k,
            candidate_top_k=args.candidate_k,
            rerank=rerank_flag
        )
        print_cli_output(res)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

