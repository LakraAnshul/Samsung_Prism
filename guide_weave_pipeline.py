"""
GUIDE WEAVE - FINAL PIPELINE (Phases B + C + D)  v2
====================================================
- Phase B: Global step numbering (no duplicates)
- Phase C: 1024x576 Pillow text-card PNGs
- Phase D: metadata JSON
"""

import os, re, json, sys, textwrap, glob
from pathlib import Path
from datetime import datetime
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

SOURCE_FOLDER   = "./ground_truth"
BASE_OUTPUT_DIR = "./generated_step_images"
FONT_PATH       = "C:/Windows/Fonts/arial.ttf"
BOLD_FONT_PATH  = "C:/Windows/Fonts/arialbd.ttf"
CANVAS_W, CANVAS_H = 1024, 576
PADDING         = 60
MIN_FONT_SIZE   = 18
MAX_FONT_SIZE   = 40
METADATA_FILE   = "generated_step_images_metadata.json"
SKIPPED_FILE    = "./SKIPPED_PDFS.md"
STEP_LINE_RE    = re.compile(r"^Step\s+(\d+)[\.\:\-]?\s+(.+)", re.IGNORECASE)

skipped_pdfs = []

def choose_output_dir(base):
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    stamped = f"{base}_{ts}"
    os.makedirs(stamped, exist_ok=True)
    print(f"[WARN] '{base}' exists --> using '{stamped}'")
    return stamped

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        return f"ERROR: {e}"

def extract_steps_global(text, guide_name):
    lines = text.splitlines()
    steps = []
    global_counter = 0
    current_lines = []

    def flush():
        nonlocal global_counter
        if current_lines:
            instruction = " ".join(current_lines).strip()
            cutoff = re.search(r"\s+After completing the steps\b", instruction, re.IGNORECASE)
            if cutoff:
                instruction = instruction[:cutoff.start()].strip()
            if len(instruction) > 500:
                instruction = instruction[:500].rsplit(" ", 1)[0] + "..."
            global_counter += 1
            steps.append({"step_number": global_counter, "step_text": instruction, "guide_name": guide_name})

    in_step = False
    for line in lines:
        stripped = line.strip()
        m = STEP_LINE_RE.match(stripped)
        if m:
            flush()
            current_lines = [m.group(2).strip()]
            in_step = True
        elif in_step and stripped and len(stripped) > 3:
            if not re.match(r"^[A-Z][A-Z\s]{10,}$", stripped):
                current_lines.append(stripped)
    flush()
    return steps

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def render_step_card(step, output_path):
    W, H = CANVAS_W, CANVAS_H
    guide_name = step["guide_name"]
    step_num   = step["step_number"]
    step_text  = step["step_text"]

    BG          = (255, 255, 255)
    STRIPE_BG   = (20, 40, 80)
    STRIPE_TEXT = (255, 255, 255)
    BODY_TEXT   = (30, 30, 30)

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    STRIPE_H = 72
    draw.rectangle([0, 0, W, STRIPE_H], fill=STRIPE_BG)

    header_font = load_font(BOLD_FONT_PATH, 22)
    guide_display = guide_name.replace("_", " ")
    draw.text((PADDING, (STRIPE_H - 22) // 2), guide_display, font=header_font, fill=STRIPE_TEXT)

    badge_text = f"STEP {step_num}"
    badge_font = load_font(BOLD_FONT_PATH, 20)
    bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = bbox[2] - bbox[0]
    draw.text((W - PADDING - bw, (STRIPE_H - 20) // 2), badge_text, font=badge_font, fill=(200, 220, 255))

    draw.line([PADDING, STRIPE_H + 12, W - PADDING, STRIPE_H + 12], fill=(20, 40, 80), width=2)

    body_top    = STRIPE_H + 30
    body_bottom = H - 30
    available_w = W - 2 * PADDING
    available_h = body_bottom - body_top

    best_font  = None
    best_lines = None

    for font_size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        font = load_font(FONT_PATH, font_size)
        avg_char_w = font_size * 0.55
        chars_per_line = max(10, int(available_w / avg_char_w))
        wrapped = textwrap.fill(step_text, width=chars_per_line)
        lines = wrapped.splitlines()
        line_h = font_size + 8
        total_h = len(lines) * line_h
        if total_h <= available_h:
            best_font  = font
            best_lines = lines
            break

    if best_font is None:
        best_font = load_font(FONT_PATH, MIN_FONT_SIZE)
        chars_per_line = max(10, int(available_w / (MIN_FONT_SIZE * 0.55)))
        wrapped = textwrap.fill(step_text, width=chars_per_line)
        best_lines = wrapped.splitlines()

    line_h  = best_font.size + 8
    total_h = len(best_lines) * line_h
    y = body_top + max(0, (available_h - total_h) // 2)

    for line in best_lines:
        draw.text((PADDING, y), line, font=best_font, fill=BODY_TEXT)
        y += line_h

    draw.line([PADDING, H - 20, W - PADDING, H - 20], fill=(200, 200, 200), width=1)
    footer_font = load_font(FONT_PATH, 13)
    draw.text((PADDING, H - 18), "Guide Weave  |  Samsung PRISM", font=footer_font, fill=(180, 180, 180))

    img.save(output_path, "PNG")

if __name__ == "__main__":
    start_time = datetime.now()
    OUTPUT_DIR = choose_output_dir(BASE_OUTPUT_DIR)
    print(f"\n>>> GUIDE WEAVE PIPELINE v2")
    print(f"    Source : {os.path.abspath(SOURCE_FOLDER)}")
    print(f"    Output : {os.path.abspath(OUTPUT_DIR)}")
    print(f"    Font   : {FONT_PATH}\n")

    pdf_files = sorted([str(p) for p in Path(SOURCE_FOLDER).glob("*.pdf")])
    if not pdf_files:
        print("[ERR] No PDFs found. Exiting.")
        sys.exit(1)

    # PHASE B
    print("=" * 60)
    print("PHASE B - STEP EXTRACTION (globally unique numbering)")
    print("=" * 60)
    all_steps = []
    for pdf_path in pdf_files:
        name       = Path(pdf_path).name
        guide_name = Path(pdf_path).stem
        text = extract_text(pdf_path)
        if text.startswith("ERROR"):
            print(f"  [ERR] {name}: {text}")
            skipped_pdfs.append((name, text))
            continue
        steps = extract_steps_global(text, guide_name)
        if len(steps) < 2:
            reason = f"Only {len(steps)} steps extracted."
            print(f"  [SKIP] {name}: {reason}")
            skipped_pdfs.append((name, reason))
            continue
        all_steps.extend(steps)
        print(f"  [OK] {name}: {len(steps)} steps")

    print(f"\nTotal steps: {len(all_steps)}")

    # PHASE C
    print("\n" + "=" * 60)
    print("PHASE C - IMAGE GENERATION")
    print("=" * 60)
    total = len(all_steps)
    for i, step in enumerate(all_steps, 1):
        guide_name = step["guide_name"]
        step_num   = step["step_number"]
        filename   = f"{guide_name}_step{step_num:03d}.png"
        out_path   = os.path.join(OUTPUT_DIR, filename)
        render_step_card(step, out_path)
        if i % 50 == 0 or i == total:
            print(f"  [IMG] [{i}/{total}] Rendered {filename}")

    # PHASE D
    print("\n" + "=" * 60)
    print("PHASE D - METADATA")
    print("=" * 60)
    metadata = []
    for step in all_steps:
        guide_name = step["guide_name"]
        step_num   = step["step_number"]
        filename   = f"{guide_name}_step{step_num:03d}.png"
        rel_path   = f"{OUTPUT_DIR}/{filename}"
        metadata.append({
            "id":              filename,
            "file_path":       rel_path,
            "guide_name":      guide_name,
            "step_number":     step_num,
            "step_text":       step["step_text"],
            "problem_name":    step["step_text"],
            "dense_caption":   f"A text card image displaying the instruction: '{step['step_text']}'.",
            "detected_objects": [],
        })
    meta_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Metadata written: {meta_path}")
    print(f"  [STATS] Total entries: {len(metadata)}")

    # SKIPPED LOG
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# SKIPPED PDFs LOG", "", f"**Generated:** {now}", f"**Source folder:** `{SOURCE_FOLDER}`", ""]
    if not skipped_pdfs:
        lines.append("_No PDFs were skipped. All PDFs processed successfully._")
    else:
        lines.append(f"**Total skipped:** {len(skipped_pdfs)}", )
        lines += ["", "| PDF | Reason |", "|-----|--------|"]
        for name, reason in skipped_pdfs:
            lines.append(f"| `{name}` | {reason} |")
    with open(SKIPPED_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("[OK] PIPELINE COMPLETE")
    print(f"   Steps extracted : {len(all_steps)}")
    print(f"   Images generated: {len(all_steps)}")
    print(f"   Metadata file   : {meta_path}")
    print(f"   Skipped PDFs    : {len(skipped_pdfs)}")
    print(f"   Elapsed time    : {elapsed:.1f}s")
    print("=" * 60)
