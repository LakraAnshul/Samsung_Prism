"""
PHASE A - DISCOVERY SCRIPT
Scans the ground_truth/ folder, detects PDF structure, checks fonts.
Writes DISCOVERY_NOTES.md
"""
import os, re, json, fitz, glob
from pathlib import Path
from datetime import datetime

SOURCE_FOLDER = "./ground_truth"
FONT_DIRS = ["C:/Windows/Fonts"]
OUTPUT_NOTES = "./DISCOVERY_NOTES.md"

NUMBERED_RE = re.compile(
    r"^(\s*(?:Step\s*\d+[\.\:\-]|\d+[\.\)]\s|[•\*\-]\s))",
    re.IGNORECASE | re.MULTILINE,
)

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        return f"ERROR: {e}"

def detect_structure(text):
    lines = text.splitlines()
    matched = [l for l in lines if NUMBERED_RE.match(l)]
    return {"total_lines": len(lines), "structured_matches": len(matched),
            "pattern": "STRUCTURED" if len(matched) >= 2 else "PROSE",
            "sample_matches": matched[:5]}

def collect_ttf_fonts(dirs):
    fonts = []
    for d in dirs:
        if os.path.isdir(d):
            fonts.extend(glob.glob(os.path.join(d, "*.ttf")))
            fonts.extend(glob.glob(os.path.join(d, "*.TTF")))
    return sorted(fonts)

def run_discovery():
    print("=" * 60)
    print("PHASE A - DISCOVERY")
    print("=" * 60)

    pdf_files = sorted([str(p) for p in Path(SOURCE_FOLDER).glob("*.pdf")])
    print(f"\nFound {len(pdf_files)} PDFs in '{SOURCE_FOLDER}':")
    for p in pdf_files:
        print(f"   {Path(p).name}")

    all_analysis = {}
    print("\nAnalysing structure of all PDFs...")
    for pdf_path in pdf_files:
        name = Path(pdf_path).name
        text = extract_text(pdf_path)
        if text.startswith("ERROR"):
            all_analysis[name] = {"pattern": "ERROR", "structured_matches": 0, "error": text}
            print(f"   ERROR: {name}")
            continue
        info = detect_structure(text)
        all_analysis[name] = info
        print(f"   {name[:55]:<55} -> {info['pattern']} ({info['structured_matches']} structured lines)")
        if info["sample_matches"]:
            print(f"      Samples: {info['sample_matches'][:2]}")

    ttf_fonts = collect_ttf_fonts(FONT_DIRS)
    print(f"\nFound {len(ttf_fonts)} TTF fonts.")

    preferred = ["arial.ttf", "arialbd.ttf", "DejaVuSans.ttf"]
    chosen_font = None
    for pref in preferred:
        for font in ttf_fonts:
            if Path(font).name.lower() == pref.lower():
                chosen_font = font
                break
        if chosen_font:
            break
    if not chosen_font and ttf_fonts:
        chosen_font = ttf_fonts[0]
    print(f"Recommended font: {chosen_font}")

    # Detailed text from first 3
    sample_details = []
    for pdf_path in pdf_files[:3]:
        name = Path(pdf_path).name
        text = extract_text(pdf_path)
        info = all_analysis[name]
        sample_details.append({"pdf": name, "snippet": text[:600], "info": info})

    # Write DISCOVERY_NOTES.md
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append("# DISCOVERY NOTES")
    md.append(f"\n**Generated:** {now}  \n**Source folder:** `{SOURCE_FOLDER}`  \n**Total PDFs found:** {len(pdf_files)}\n")
    md.append("---\n## 1. PDF Inventory\n")
    md.append("| # | Filename |")
    md.append("|---|----------|")
    for i, p in enumerate(pdf_files, 1):
        md.append(f"| {i} | `{Path(p).name}` |")
    md.append("\n---\n## 2. Structure Analysis (All PDFs)\n")
    md.append("| PDF | Pattern | Structured-line count |")
    md.append("|-----|---------|----------------------|")
    for name, info in all_analysis.items():
        md.append(f"| `{name}` | **{info['pattern']}** | {info.get('structured_matches', 0)} |")
    md.append("\n---\n## 3. Sample Text Snippets (first 3 PDFs)\n")
    for d in sample_details:
        md.append(f"### `{d['pdf']}`")
        md.append(f"- Pattern: **{d['info']['pattern']}**")
        md.append(f"- Structured matches: {d['info']['structured_matches']}")
        md.append(f"- Sample matches: {d['info'].get('sample_matches', [])[:3]}")
        md.append(f"\n```\n{d['snippet'][:400]}\n```\n")
    md.append("\n---\n## 4. Font Availability\n")
    md.append(f"**Total TTF fonts found:** {len(ttf_fonts)}\n\n**All fonts:**")
    for f in ttf_fonts[:40]:
        md.append(f"- `{f}`")
    md.append(f"\n### Recommended font:\n`{chosen_font}`\n")
    md.append("---\n## 5. Conclusions\n")
    structured_count = sum(1 for v in all_analysis.values() if v.get("pattern") == "STRUCTURED")
    md.append(f"- **{structured_count}/{len(all_analysis)}** PDFs are STRUCTURED (regex-extractable)")
    md.append(f"- **{len(all_analysis)-structured_count}/{len(all_analysis)}** PDFs need LLM fallback")
    md.append(f"- Recommended font path: `{chosen_font}`")

    with open(OUTPUT_NOTES, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nDISCOVERY_NOTES.md written to: {OUTPUT_NOTES}")

    return {"pdf_files": pdf_files, "all_analysis": all_analysis, "chosen_font": chosen_font, "ttf_fonts": ttf_fonts}

if __name__ == "__main__":
    result = run_discovery()
    print("\nSUMMARY:")
    print(f"  Total PDFs: {len(result['pdf_files'])}")
    print(f"  Chosen font: {result['chosen_font']}")
    structured = sum(1 for v in result["all_analysis"].values() if v.get("pattern") == "STRUCTURED")
    print(f"  STRUCTURED: {structured}, PROSE: {len(result['pdf_files']) - structured}")
