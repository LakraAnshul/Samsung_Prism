import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    TOKENIZER_USED = "tiktoken (cl100k_base)"
except ImportError:
    print("Warning: tiktoken not found. Using simple word/character fallback for token counting.")
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3 + 1
    TOKENIZER_USED = "fallback (word count heuristic)"

def extract_blocks_from_pages(pages: List[Dict]) -> List[Dict]:
    blocks = []
    current_block_lines = []
    start_page = None
    last_page = None
    
    def finish_block():
        if current_block_lines:
            text = "\n".join(current_block_lines).strip()
            if text:
                blocks.append({
                    "text": text,
                    "page_start": start_page,
                    "page_end": last_page,
                    "type": "unknown",
                    "step_num": None
                })
            current_block_lines.clear()
            
    for p in pages:
        page_num = p["page_number"]
        text = p.get("text", "")
        if not text:
            continue
            
        lines = text.split('\n')
        for line in lines:
            line_str = line.strip()
            if not line_str:
                finish_block()
                continue
                
            is_heading = bool(re.match(r'^\d+\.\s+[A-Z]', line_str) or re.match(r'^\d+\.\d+\.\s+[A-Z]', line_str))
            is_step = bool(re.match(r'^Step\s+\d+[\.\:]?', line_str, re.IGNORECASE))
            is_warning = bool(re.match(r'^(WARNING|CAUTION|NOTE|SAFETY)', line_str, re.IGNORECASE))
            
            if is_heading or is_step or is_warning:
                finish_block()
                current_block_lines.append(line_str)
                start_page = page_num
                last_page = page_num
            else:
                if not current_block_lines:
                    start_page = page_num
                current_block_lines.append(line_str)
                last_page = page_num
                
    finish_block()
    return blocks

def classify_blocks(blocks: List[Dict]):
    for b in blocks:
        text = b["text"]
        
        if re.match(r'^\d+\.\s+[A-Z]', text):
            b["type"] = "section"
        elif re.match(r'^\d+\.\d+\.\s+[A-Z]', text):
            b["type"] = "subsection"
        elif re.match(r'^Step\s+(\d+)[\.\:]?', text, re.IGNORECASE):
            b["type"] = "step"
            match = re.match(r'^Step\s+(\d+)', text, re.IGNORECASE)
            if match:
                b["step_num"] = int(match.group(1))
        elif re.match(r'^(WARNING|CAUTION|NOTE|SAFETY)', text, re.IGNORECASE):
            b["type"] = "warning"
        elif re.match(r'^(When to escalate|Service):', text, re.IGNORECASE):
            b["type"] = "escalation"
        else:
            b["type"] = "paragraph"

def build_hierarchy(blocks: List[Dict], doc_id: str) -> Dict:
    tree = {"sections": []}
    
    current_section = None
    current_subsection = None
    
    sec_counter = 1
    subsec_counter = 1
    
    for b in blocks:
        b["tokens"] = count_tokens(b["text"])
        
        if b["type"] == "section":
            current_section = {
                "type": "section",
                "title": b["text"].split('\n')[0],
                "id": f"{doc_id}_section_{sec_counter:02d}",
                "subsections": [],
                "blocks": [b]
            }
            sec_counter += 1
            tree["sections"].append(current_section)
            current_subsection = None
            
        elif b["type"] == "subsection":
            if not current_section:
                current_section = {
                    "type": "section",
                    "title": "General",
                    "id": f"{doc_id}_section_{sec_counter:02d}",
                    "subsections": [],
                    "blocks": []
                }
                sec_counter += 1
                tree["sections"].append(current_section)
                
            current_subsection = {
                "type": "subsection",
                "title": b["text"].split('\n')[0],
                "id": f"{doc_id}_problem_{subsec_counter:02d}",
                "blocks": [b]
            }
            subsec_counter += 1
            current_section["subsections"].append(current_subsection)
            
        else:
            if not current_section:
                current_section = {
                    "type": "section",
                    "title": "General",
                    "id": f"{doc_id}_section_{sec_counter:02d}",
                    "subsections": [],
                    "blocks": []
                }
                sec_counter += 1
                tree["sections"].append(current_section)
                
            if not current_subsection:
                current_section["blocks"].append(b)
            else:
                current_subsection["blocks"].append(b)
                
    return tree

def chunk_hierarchy(tree: Dict, doc_meta: Dict, config: Dict):
    parents = []
    children = []
    
    parent_max = config["parent_max_tokens"]
    child_max = config["child_max_tokens"]
    
    parent_counter = 1
    child_counter = 1
    
    def process_blocks(blocks, context):
        nonlocal parent_counter, child_counter
        
        current_parent_blocks = []
        current_parent_tokens = 0
        
        def flush_parent():
            nonlocal parent_counter, child_counter
            if not current_parent_blocks:
                return
            
            p_id = f"{doc_meta['document_id']}_parent_{parent_counter:03d}"
            parent_counter += 1
            
            p_children = chunk_into_children(current_parent_blocks, p_id, context, child_max)
            
            p_text = "\n\n".join(b["text"] for b in current_parent_blocks)
            p_page_start = min(b["page_start"] for b in current_parent_blocks)
            p_page_end = max(b["page_end"] for b in current_parent_blocks)
            
            parent_chunk = {
                "parent_chunk_id": p_id,
                "section": context.get("section_title"),
                "subsection": context.get("subsection_title"),
                "problem_id": context.get("problem_id"),
                "problem_name": context.get("problem_title"),
                "page_start": p_page_start,
                "page_end": p_page_end,
                "chunk_type": "procedure" if any(b["type"] == "step" for b in current_parent_blocks) else "informational",
                "text": p_text,
                "token_count": count_tokens(p_text),
                "child_chunk_ids": [c["chunk_id"] for c in p_children]
            }
            parents.append(parent_chunk)
            children.extend(p_children)
            
            current_parent_blocks.clear()
            
        for b in blocks:
            if current_parent_blocks and current_parent_tokens + b["tokens"] > parent_max:
                flush_parent()
                current_parent_tokens = 0
                
            current_parent_blocks.append(b)
            current_parent_tokens += b["tokens"]
            
        flush_parent()

    def chunk_into_children(blocks, parent_id, context, max_tokens):
        nonlocal child_counter
        kids = []
        
        current_child_blocks = []
        current_child_tokens = 0
        
        def flush_child():
            nonlocal child_counter
            if not current_child_blocks:
                return
                
            header = ""
            if context.get("subsection_title") and not any(b["type"] == "subsection" for b in current_child_blocks):
                header = f"[{context['subsection_title']}]\n"
            elif context.get("section_title") and not any(b["type"] in ["section", "subsection"] for b in current_child_blocks):
                header = f"[{context['section_title']}]\n"
                
            raw_text = "\n\n".join(b["text"] for b in current_child_blocks)
            c_text = header + raw_text
            
            c_id = f"{parent_id}_child_{child_counter:03d}"
            child_counter += 1
            
            c_page_start = min(b["page_start"] for b in current_child_blocks)
            c_page_end = max(b["page_end"] for b in current_child_blocks)
            
            steps = [b["step_num"] for b in current_child_blocks if b.get("step_num") is not None]
            c_step_start = min(steps) if steps else None
            c_step_end = max(steps) if steps else None
            
            toks = count_tokens(c_text)
            if toks > max_tokens:
                sub_kids = recursive_hard_split(c_text, max_tokens, parent_id, context, c_page_start, c_page_end, c_step_start, c_step_end)
                kids.extend(sub_kids)
            else:
                kids.append(create_child_dict(c_id, parent_id, doc_meta, context, c_page_start, c_page_end, c_step_start, c_step_end, c_text))
                
            current_child_blocks.clear()

        for b in blocks:
            if current_child_blocks and current_child_tokens + b["tokens"] > max_tokens:
                flush_child()
                current_child_tokens = 0
                
            current_child_blocks.append(b)
            current_child_tokens += b["tokens"]
            
        flush_child()
        return kids
        
    def recursive_hard_split(text, max_tokens, parent_id, context, p_start, p_end, s_start, s_end):
        nonlocal child_counter
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+|\n', text)
        
        current_text = ""
        for s in sentences:
            if not s.strip():
                continue
            
            if count_tokens(current_text + " " + s) > max_tokens:
                if current_text:
                    c_id = f"{parent_id}_child_{child_counter:03d}"
                    child_counter += 1
                    chunks.append(create_child_dict(c_id, parent_id, doc_meta, context, p_start, p_end, s_start, s_end, current_text.strip(), "recursive_overflow"))
                    current_text = ""
                    
                if count_tokens(s) > max_tokens:
                    words = s.split()
                    temp_word_chunk = ""
                    for w in words:
                        if count_tokens(temp_word_chunk + " " + w) > max_tokens:
                            c_id = f"{parent_id}_child_{child_counter:03d}"
                            child_counter += 1
                            chunks.append(create_child_dict(c_id, parent_id, doc_meta, context, p_start, p_end, s_start, s_end, temp_word_chunk.strip(), "recursive_overflow_words"))
                            temp_word_chunk = w
                        else:
                            temp_word_chunk += (" " if temp_word_chunk else "") + w
                    if temp_word_chunk:
                        current_text = temp_word_chunk
                else:
                    current_text = s
            else:
                current_text += (" " if current_text else "") + s
                
        if current_text.strip():
            c_id = f"{parent_id}_child_{child_counter:03d}"
            child_counter += 1
            chunks.append(create_child_dict(c_id, parent_id, doc_meta, context, p_start, p_end, s_start, s_end, current_text.strip(), "recursive_overflow"))
            
        return chunks

    def create_child_dict(c_id, parent_id, doc_meta, context, p_start, p_end, s_start, s_end, text, split_reason=None):
        d = {
            "chunk_id": c_id,
            "parent_chunk_id": parent_id,
            "document_id": doc_meta["document_id"],
            "appliance_type": doc_meta["appliance_type"],
            "brand": doc_meta["brand"],
            "model": doc_meta["model"],
            "source_file": doc_meta["source"]["file_name"],
            "section": context.get("section_title"),
            "subsection": context.get("subsection_title"),
            "problem_id": context.get("problem_id"),
            "problem_name": context.get("problem_title"),
            "step_start": s_start,
            "step_end": s_end,
            "page_start": p_start,
            "page_end": p_end,
            "chunk_level": "child",
            "chunk_type": "procedure" if s_start else "informational",
            "text": text,
            "token_count": count_tokens(text)
        }
        if split_reason:
            d["split_reason"] = split_reason
        return d

    for sec in tree["sections"]:
        ctx = {
            "section_title": sec["title"],
            "subsection_title": None,
            "problem_id": None,
            "problem_title": None
        }
        if sec["blocks"]:
            process_blocks(sec["blocks"], ctx)
            
        for subsec in sec["subsections"]:
            sub_ctx = {
                "section_title": sec["title"],
                "subsection_title": subsec["title"],
                "problem_id": subsec["id"],
                "problem_title": subsec["title"]
            }
            if subsec["blocks"]:
                process_blocks(subsec["blocks"], sub_ctx)
                
    return parents, children

def generate_hierarchy_output(tree: Dict) -> Dict:
    hierarchy = {"sections": []}
    for sec in tree["sections"]:
        h_sec = {
            "section_id": sec["id"],
            "title": sec["title"],
            "subsections": []
        }
        for subsec in sec["subsections"]:
            h_subsec = {
                "subsection_id": subsec["id"],
                "title": subsec["title"]
            }
            h_sec["subsections"].append(h_subsec)
        hierarchy["sections"].append(h_sec)
    return hierarchy

def validate_chunks(parents, children):
    parent_ids = set()
    for p in parents:
        assert p["parent_chunk_id"] not in parent_ids, "Duplicate parent ID"
        parent_ids.add(p["parent_chunk_id"])
        
    child_ids = set()
    for c in children:
        assert c["chunk_id"] not in child_ids, "Duplicate child ID"
        assert c["parent_chunk_id"] in parent_ids, f"Child {c['chunk_id']} references missing parent"
        assert c["text"].strip(), "Empty child chunk text"
        child_ids.add(c["chunk_id"])
        
    for p in parents:
        for cid in p["child_chunk_ids"]:
            assert cid in child_ids, f"Parent {p['parent_chunk_id']} references missing child {cid}"

def process_file(input_path: Path, output_dir: Path, config: Dict):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    blocks = extract_blocks_from_pages(data["pages"])
    classify_blocks(blocks)
    
    tree = build_hierarchy(blocks, data["document_id"])
    parents, children = chunk_hierarchy(tree, data, config)
    
    validate_chunks(parents, children)
    
    hierarchy = generate_hierarchy_output(tree)
    
    output_data = {
        "document_id": data["document_id"],
        "appliance_type": data["appliance_type"],
        "brand": data["brand"],
        "model": data["model"],
        "source": {
            "file_name": data["source"]["file_name"]
        },
        "chunking_config": config,
        "hierarchy": hierarchy,
        "parents": parents,
        "children": children
    }
    
    output_file = output_dir / input_path.name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    return {
        "name": input_path.name,
        "blocks": len(blocks),
        "parents": len(parents),
        "children": len(children),
        "p_tokens": [p["token_count"] for p in parents],
        "c_tokens": [c["token_count"] for c in children]
    }

def main():
    input_dir = Path("extracted_ground_truth")
    output_dir = Path("chunked_ground_truth")
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory {input_dir} does not exist.")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "strategy": "document_aware_hierarchical_recursive_parent_child",
        "parent_target_tokens": 2200,
        "parent_max_tokens": 3000,
        "child_target_tokens": 600,
        "child_max_tokens": 1000,
        "overlap_tokens": 75,
        "tokenizer": TOKENIZER_USED
    }
    
    json_files = list(input_dir.rglob("*.json"))
    
    print(f"Discovered {len(json_files)} JSON files to chunk.\n")
    
    success_count = 0
    failed_files = []
    
    total_parents = 0
    total_children = 0
    all_c_tokens = []
    
    for jf in json_files:
        try:
            stats = process_file(jf, output_dir, config)
            success_count += 1
            
            p_count = stats["parents"]
            c_count = stats["children"]
            total_parents += p_count
            total_children += c_count
            all_c_tokens.extend(stats["c_tokens"])
            
            p_avg = sum(stats["p_tokens"]) // p_count if p_count > 0 else 0
            c_avg = sum(stats["c_tokens"]) // c_count if c_count > 0 else 0
            c_max = max(stats["c_tokens"]) if c_count > 0 else 0
            
            print(f"[PROCESSING]\n{jf.name}\n")
            print("Detected structure:")
            print(f"    Blocks parsed: {stats['blocks']}")
            print("Chunks:")
            print(f"    Parents: {p_count}")
            print(f"    Children: {c_count}")
            print("Token statistics:")
            print(f"    Parent average: {p_avg}")
            print(f"    Child average: {c_avg}")
            print(f"    Maximum child: {c_max}")
            print("\n[OK]")
            print(f"Output:\n    {output_dir}/{jf.name}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {jf.name}: {e}\n")
            failed_files.append((jf.name, str(e)))
            
    print("==================================================")
    print("CHUNKING COMPLETE")
    print("==================================================")
    print()
    print(f"Documents processed: {len(json_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")
    print()
    print(f"Total parents: {total_parents}")
    print(f"Total children: {total_children}")
    print()
    
    avg_c_per_p = total_children / total_parents if total_parents > 0 else 0
    avg_c_toks = sum(all_c_tokens) // len(all_c_tokens) if all_c_tokens else 0
    max_c_toks = max(all_c_tokens) if all_c_tokens else 0
    
    print(f"Average children per parent: {avg_c_per_p:.1f}")
    print(f"Average child tokens: {avg_c_toks}")
    print(f"Maximum child tokens: {max_c_toks}")
    print("\n==================================================")
    
if __name__ == "__main__":
    main()
