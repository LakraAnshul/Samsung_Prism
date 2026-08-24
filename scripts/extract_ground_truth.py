import fitz
from pathlib import Path
import json
import re
import sys
from typing import Dict, Any, List

def discover_pdfs(input_dir: Path) -> List[Path]:
    """Finds all PDF files in the input directory recursively."""
    return list(input_dir.rglob("*.pdf"))

def build_document_metadata(pdf_path: Path, input_dir: Path) -> Dict[str, Any]:
    """Builds base metadata for a document."""
    brand = "Samsung"
    model = "WA5471ABP/XAA"
    appliance_type = "washing_machine"
    
    # Generate deterministic document_id
    model_prefix = model.split('/')[0]
    doc_id = f"{model_prefix}_{pdf_path.stem}"
    
    # Clean up doc_id just in case
    doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
    
    # Use forward slashes for cross-platform consistency in source file_path
    # Since we are in the project root, input_dir is likely 'ground_truth'
    # we want source path to look like 'ground_truth/file.pdf'
    try:
        relative_path = pdf_path.relative_to(input_dir.parent if input_dir.parent.name else Path("."))
    except ValueError:
        # Fallback if relative_to fails
        relative_path = pdf_path
        
    return {
        "document_id": doc_id,
        "appliance_type": appliance_type,
        "brand": brand,
        "model": model,
        "source": {
            "file_name": pdf_path.name,
            "file_path": str(relative_path).replace('\\', '/'),
            "file_type": "pdf",
            "extraction_method": "pymupdf"
        }
    }

def normalize_text(text: str) -> str:
    """Performs basic text normalization without altering structure."""
    if not text:
        return ""
        
    # Replace carriage returns and inconsistent newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2 consecutive blank lines -> 2 blank lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Rejoin and strip outer whitespace
    return '\n'.join(lines).strip()

def extract_page_text(page: fitz.Page, page_number: int) -> Dict[str, Any]:
    """Extracts text from a single page."""
    raw_text = page.get_text("text")
    text = normalize_text(raw_text)
    
    has_text = bool(text.strip())
    
    return {
        "page_number": page_number,
        "has_text": has_text,
        "text": text
    }

def extract_pdf(pdf_path: Path, input_dir: Path) -> Dict[str, Any]:
    """Extracts all text and metadata from a PDF file."""
    metadata = build_document_metadata(pdf_path, input_dir)
    
    pages = []
    stats = {
        "page_count": 0,
        "pages_with_text": 0,
        "pages_without_text": 0,
        "total_characters": 0,
        "total_words": 0
    }
    
    doc = fitz.open(pdf_path)
    stats["page_count"] = len(doc)
    metadata["page_count"] = len(doc)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_data = extract_page_text(page, page_num + 1)
        pages.append(page_data)
        
        if page_data["has_text"]:
            stats["pages_with_text"] += 1
            stats["total_characters"] += len(page_data["text"])
            stats["total_words"] += len(page_data["text"].split())
        else:
            stats["pages_without_text"] += 1
            
    doc.close()
    
    metadata["pages"] = pages
    metadata["statistics"] = stats
    
    return metadata

def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """Saves dictionary data to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_all_pdfs(input_dir: Path, output_dir: Path) -> None:
    """Processes all PDFs in the input directory and saves to output directory."""
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_paths = discover_pdfs(input_dir)
    
    if not pdf_paths:
        print(f"No PDFs found in '{input_dir}'.")
        return
        
    print(f"Discovered {len(pdf_paths)} PDFs in '{input_dir}'.\n")
    
    success_count = 0
    failed_files = []
    
    total_stats = {
        "pages": 0,
        "pages_with_text": 0,
        "pages_without_text": 0,
        "characters": 0,
        "words": 0
    }
    
    for pdf_path in pdf_paths:
        try:
            relative_path = pdf_path.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix('.json')
            
            data = extract_pdf(pdf_path, input_dir)
            save_json(data, output_file)
            
            stats = data["statistics"]
            print(f"[OK] {pdf_path.name}")
            print(f"     Pages: {stats['page_count']}")
            print(f"     Characters: {stats['total_characters']:,}")
            print(f"     Words: {stats['total_words']:,}")
            print(f"     Output: {output_file.as_posix()}")
            print()
            
            success_count += 1
            total_stats["pages"] += stats["page_count"]
            total_stats["pages_with_text"] += stats["pages_with_text"]
            total_stats["pages_without_text"] += stats["pages_without_text"]
            total_stats["characters"] += stats["total_characters"]
            total_stats["words"] += stats["total_words"]
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path.name}: {e}\n")
            failed_files.append((pdf_path.name, str(e)))
            
    print_summary(input_dir, output_dir, len(pdf_paths), success_count, failed_files, total_stats)

def print_summary(input_dir: Path, output_dir: Path, total_discovered: int, success_count: int, failed_files: List[tuple], stats: Dict[str, int]) -> None:
    """Prints a summary of the batch processing."""
    print("=" * 50)
    print("GROUND TRUTH EXTRACTION COMPLETE")
    print("=" * 50)
    print()
    print(f"Input directory:\n    {input_dir}/")
    print(f"Output directory:\n    {output_dir}/")
    print()
    print(f"PDFs discovered: {total_discovered}")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed: {len(failed_files)}")
    print()
    print(f"Total pages: {stats['pages']}")
    print(f"Pages containing text: {stats['pages_with_text']}")
    print(f"Pages without text: {stats['pages_without_text']}")
    print()
    print(f"Total characters: {stats['characters']:,}")
    print(f"Total words: {stats['words']:,}")
    print("\n" + "=" * 50)
    
    if failed_files:
        print("\nFailed files details:")
        for name, error in failed_files:
            print(f" - {name}: {error}")

def main():
    input_dir = Path("ground_truth")
    output_dir = Path("extracted_ground_truth")
    process_all_pdfs(input_dir, output_dir)

if __name__ == "__main__":
    main()
