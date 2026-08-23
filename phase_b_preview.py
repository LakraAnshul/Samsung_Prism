"""
PHASE B PREVIEW - Extract steps from ONE PDF (for quality check).
Uses the Step N. regex pattern found in Phase A.
"""
import re, json
import pymupdf as fitz
from pathlib import Path

SOURCE_FOLDER = "./ground_truth"
NUMBERED_RE = re.compile(
    r"^(Step\s+\d+[\.\:\-]?\s+.+?)(?=\nStep\s+\d+|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
# Simpler line-by-line extraction
STEP_LINE_RE = re.compile(r"^Step\s+(\d+)[\.\:\-]?\s+(.+)", re.IGNORECASE)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)

def clean_guide_name(filename):
    name = Path(filename).stem  # e.g. "01_Starting_Power_Problems_Detailed"
    return name

def extract_steps_regex(text, guide_name):
    """
    Strategy: collect all lines, group consecutive continuation lines
    after each 'Step N.' line into a single instruction block.
    """
    lines = text.splitlines()
    steps = []
    current_step = None
    current_step_num = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        m = STEP_LINE_RE.match(stripped)
        if m:
            # Save previous step if any
            if current_step is not None and current_lines:
                full_instruction = " ".join(current_lines).strip()
                # Remove trailing boilerplate if too long
                if len(full_instruction) > 600:
                    full_instruction = full_instruction[:600].rsplit(" ", 1)[0] + "..."
                steps.append({
                    "step_number": current_step_num,
                    "step_text": full_instruction,
                    "guide_name": guide_name,
                })
            current_step_num = int(m.group(1))
            current_step = stripped
            current_lines = [m.group(2).strip()]
        else:
            # Continuation line — only attach if it looks like real content
            if current_step is not None and stripped and len(stripped) > 3:
                # Stop collecting if we hit a section header-like line (all caps / very short)
                if not re.match(r"^[A-Z][A-Z\s]{10,}$", stripped):
                    current_lines.append(stripped)

    # Don't forget the last step
    if current_step is not None and current_lines:
        full_instruction = " ".join(current_lines).strip()
        if len(full_instruction) > 600:
            full_instruction = full_instruction[:600].rsplit(" ", 1)[0] + "..."
        steps.append({
            "step_number": current_step_num,
            "step_text": full_instruction,
            "guide_name": guide_name,
        })

    return steps

# --- RUN ON ONE PDF ---
pdf_files = sorted(list(Path(SOURCE_FOLDER).glob("*.pdf")))
test_pdf = pdf_files[0]
guide_name = clean_guide_name(test_pdf.name)
print(f"Testing on: {test_pdf.name}")
print(f"Guide name: {guide_name}\n")

text = extract_text(str(test_pdf))
steps = extract_steps_regex(text, guide_name)

print(f"EXTRACTED {len(steps)} STEPS:\n" + "=" * 60)
for s in steps:
    print(f"\n  Step {s['step_number']:>2}: {s['step_text'][:200]}")
    if len(s['step_text']) > 200:
        print(f"            ... [{len(s['step_text'])} chars total]")

print("\n" + "=" * 60)
print(f"Total steps extracted: {len(steps)}")
print("\nFull JSON of first 3 steps:")
print(json.dumps(steps[:3], indent=2))
