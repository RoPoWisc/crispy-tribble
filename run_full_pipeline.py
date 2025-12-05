from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

# --- New Imports for Presentation/Visual Logic ---
import torch
from PIL import Image
from pypdf import PdfReader
from markitdown import MarkItDown
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForCausalLM

from scripts.generate_ad_copy_index import build_index as build_ad_index
from scripts.generate_ad_copy_index import extract_prefix

# --- Constants & Config ---
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
AD_DIR = DATA / "ad_copy_reviews"
WEB_DIR = DATA / "website_reviews"
DOC_CACHE = DATA / ".converted_docx"
OUTPUT_DIR = ROOT / "output" / "text"
AD_INDEX = AD_DIR / "index.json"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Classes ---

class FlorenceEnricher:
    """
    Singleton wrapper for Microsoft Florence-2.
    Handles Object Detection (<OD>) to find charts, then extracts data via OCR/Captioning.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FlorenceEnricher, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        logger.info("Loading Florence-2 Model (this happens once)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        
        # Use 'microsoft/Florence-2-base' for speed, or 'large' for better OCR
        model_id = 'microsoft/Florence-2-large'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Keywords that trigger a deep dive extraction
        self.chart_labels = {"chart", "plot", "graph", "diagram", "infographic", "figure", "table"}

    def detect_and_extract(self, image: Image.Image) -> Optional[str]:
        """
        The "Classifier": 
        1. Runs Object Detection (<OD>) to find 'chart' bounding boxes.
        2. If found, runs OCR and Detailed Captioning.
        """
        # --- Step 1: Fast Classifier (Object Detection) ---
        od_results = self._run_inference("<OD>", image)
        
        # Florence returns {'<OD>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': ['chart', 'person']}}
        labels = od_results.get("<OD>", {}).get("labels", [])
        
        # Check if any detected label is interesting
        has_chart = any(label.lower() in self.chart_labels for label in labels)

        # Fallback: Sometimes OD misses abstract charts. Check simple caption.
        if not has_chart:
            caption = self._run_inference("<CAPTION>", image).get("<CAPTION>", "").lower()
            if any(x in caption for x in ["chart", "graph", "performance", "allocation", "statistics"]):
                has_chart = True

        if not has_chart:
            return None # No visuals detected, rely on MarkItDown text

        # --- Step 2: The "Deep Dive" (If chart detected) ---
        # A. Detailed Description (Trends, colors, layout)
        desc = self._run_inference("<MORE_DETAILED_CAPTION>", image).get("<MORE_DETAILED_CAPTION>", "")
        
        # B. OCR (The raw numbers inside the chart)
        ocr = self._run_inference("<OCR>", image).get("<OCR>", "")

        return (
            f"\n\n### [VISUAL DATA DETECTED]\n"
            f"**Visual Description:** {desc}\n"
            f"**Extracted Data Points:** {ocr}\n"
            f"----------------------------------------\n"
        )

    def _run_inference(self, prompt, image):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer


class PresentationDetector:
    """
    Determines if a file is a presentation (PPTX or Landscape PDF).
    """
    @staticmethod
    def is_presentation(path: Path) -> bool:
        ext = path.suffix.lower()
        if ext == ".pptx":
            return True
        
        if ext == ".pdf":
            try:
                reader = PdfReader(str(path))
                if len(reader.pages) == 0:
                    return False
                
                # Check the first 3 pages (or all if <3)
                # If majority are Landscape (Width > Height), treat as presentation
                landscape_count = 0
                check_pages = reader.pages[:3]
                for page in check_pages:
                    w = float(page.mediabox.width)
                    h = float(page.mediabox.height)
                    if w > h:
                        landscape_count += 1
                
                # Heuristic: If > 50% checked pages are landscape, likely slides
                return landscape_count > (len(check_pages) / 2)
            except Exception:
                return False
        return False

# --- Core Pipeline Logic ---

def build_index(ad_dir: Path, out_path: Path) -> List[Dict[str, str | List[str]]]:
    items = build_ad_index(ad_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote index: {out_path}")
    return items

def ensure_docx(path: Path) -> Path:
    if path.suffix.lower() != ".doc":
        return path
    sibling = path.with_suffix(".docx")
    if sibling.exists():
        return sibling
    DOC_CACHE.mkdir(parents=True, exist_ok=True)
    target = DOC_CACHE / f"{path.stem}.docx"
    if target.exists():
        return target
    try:
        from doc2docx import convert  # type: ignore
    except Exception as exc:
        raise RuntimeError("doc2docx is required to convert .doc files.") from exc
    convert(str(path), str(target))
    if not target.exists():
        raise RuntimeError(f"Conversion failed for {path}")
    return target

def extractor_unstructured(path: Path) -> str:
    from unstructured.partition.auto import partition
    elements = partition(filename=str(path), strategy="fast")
    return "\n\n".join(getattr(el, "text", "") for el in elements if getattr(el, "text", ""))

def extractor_ragparser(path: Path) -> str:
    from ragparser import ChunkingStrategy, ParserConfig, RagParser
    cfg = ParserConfig(chunking_strategy=ChunkingStrategy.FIXED, chunk_size=1500, chunk_overlap=200, enable_ocr=False, extract_tables=False, extract_images=False)
    parser = RagParser(cfg)
    res = parser.parse(str(path))
    if not res.success or not res.document:
        raise RuntimeError(res.error or "Unknown ragparser failure")
    return res.document.content

def extractor_smart_presentation(path: Path) -> str:
    """
    1. Runs MarkItDown to get base text.
    2. Renders pages to images (if PDF).
    3. Runs Florence-2 on visual pages to fill gaps.
    """
    md = MarkItDown()
    enricher = FlorenceEnricher()
    
    # 1. Base Text Extraction (The Skeleton)
    try:
        result = md.convert(str(path))
        base_text = result.text_content
    except Exception as e:
        logger.error(f"MarkItDown failed on {path.name}: {e}")
        base_text = ""

    # 2. Visual Scan (The Muscle)
    # MarkItDown is great for text, but often misses charts in PDFs/PPTXs.
    # We render pages to images and let Florence-2 'watch' the presentation.
    
    visual_insights = []
    
    # We only render PDF to images. If it's a PPTX, MarkItDown usually extracts internal XML charts well.
    # If PPTX has screenshots of charts, we would need to convert PPTX->PDF first to render.
    # For now, we focus the heavy visual lifting on .pdf input.
    if path.suffix.lower() == ".pdf":
        try:
            # Limit page scan to first 15 pages to prevent OOM on massive reports
            images = convert_from_path(str(path), first_page=0, last_page=15)
            
            for i, img in enumerate(images):
                # Classify & Extract
                visual_data = enricher.detect_and_extract(img)
                
                if visual_data:
                    # Append insight with Page Reference
                    visual_insights.append(f"## Page {i+1} Visual Analysis\n{visual_data}")
                    
        except Exception as e:
            logger.warning(f"Visual rendering failed for {path.name}: {e}")

    # 3. Merge
    final_text = base_text + "\n\n" + "\n".join(visual_insights)
    return final_text


def write_text(out_dir: Path, rel_path: Path, method: str, text: str) -> None:
    dest_dir = out_dir / rel_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{rel_path.stem}__{method}.txt"
    with dest.open("w", encoding="utf-8") as f:
        f.write(text)


def process_source(
    source: Path,
    rel_output_path: Path,
    max_docling_pages: int,
    skip_convert_failures: bool,
    skip_existing: bool = True,
    skip_pptx: bool = False,
) -> List[Tuple[str, Optional[str]]]:
    """
    Process one file. logic updated to handle Smart Presentation Detection.
    """
    results: List[Tuple[str, Optional[str]]] = []
    ext = source.suffix.lower()

    if skip_pptx and ext == ".pptx":
        results.append(("skip", "PPTX skipped by flag"))
        return results

    # --- 1. Detect File Type Strategy ---
    is_pres = PresentationDetector.is_presentation(source)
    
    extractors: Dict[str, Any] = {}

    if is_pres:
        # STRATEGY: Presentations (PPTX or Landscape PDFs)
        logger.info(f"Detected Presentation: {source.name}")
        extractors["smart_markitdown"] = lambda p=source: extractor_smart_presentation(p)
        
        # Fallback/Comparison extractors
        if ext == ".pdf":
            extractors["unstructured"] = lambda p=source: extractor_unstructured(p)
    
    elif ext == ".pdf":
        # STRATEGY: Standard Documents
        extractors = {
            "unstructured": lambda p=source: extractor_unstructured(p),
            "ragparser": lambda p=source: extractor_ragparser(p),
        }
    elif ext == ".docx":
        extractors = {
            "unstructured": lambda p=source: extractor_unstructured(p),
        }
    else:
        results.append(("skip", f"Unsupported ext {source.suffix}"))
        return results

    # --- 2. Run Extractors ---
    for name, fn in extractors.items():
        dest = OUTPUT_DIR / rel_output_path.parent / f"{rel_output_path.stem}__{name}.txt"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            text = fn()
            write_text(OUTPUT_DIR, rel_output_path, name, text)
            results.append((name, None))
        except Exception as exc:
            logger.error(f"Error in {name} for {source.name}: {exc}")
            results.append((name, str(exc)))
            
    return results


def process_ad_entry(
    entry: Dict[str, str | List[str]],
    max_docling_pages: int,
    skip_convert_failures: bool,
    skip_existing: bool = True,
    skip_pptx: bool = False,
) -> List[Tuple[str, Optional[str], str]]:
    outcomes: List[Tuple[str, Optional[str], str]] = []
    doc_path = ROOT / str(entry["doc"])
    prefix = extract_prefix(Path(entry["name"]).name) or "unknown"
    base_rel = Path("data") / "ad_copy_reviews" / prefix

    # Handle .doc conversion
    try:
        converted_source = ensure_docx(doc_path)
    except Exception as exc:
        err = f".doc conversion failed: {exc}"
        outcomes.append(("convert", err, "converted"))
        if not skip_convert_failures:
            raise
    else:
        rel_output = base_rel / "converted" / converted_source.name
        for method, error in process_source(
            converted_source,
            rel_output,
            max_docling_pages,
            skip_convert_failures,
            skip_existing=skip_existing,
            skip_pptx=skip_pptx,
        ):
            outcomes.append((method, error, "converted"))

    # Process org files
    converted_rel = str(entry.get("converted", ""))
    doc_stem = Path(entry["name"]).stem
    for org in entry.get("org_files", []):
        org_path = ROOT / org
        if converted_rel and Path(org) == Path(converted_rel):
            continue
        if org_path.stem == doc_stem:
            continue
        rel_output = base_rel / "org_files" / org_path.name
        for method, error in process_source(
            org_path,
            rel_output,
            max_docling_pages,
            skip_convert_failures,
            skip_existing=skip_existing,
            skip_pptx=skip_pptx,
        ):
            outcomes.append((method, error, "org_files"))

    return outcomes


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full data pipeline with Smart Presentation Detection.")
    ap.add_argument("--max-docling-pages", type=int, default=5, help="Limit pages for docling.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of files.")
    ap.add_argument("--skip-convert-failures", action="store_true", help="Continue when .doc conversion fails.")
    ap.add_argument("--skip-pptx", action="store_true", help="Skip PPTX inputs.")
    ap.add_argument("--log-failures", action="store_true", help="Write failures log.")
    ap.add_argument("--sanitize", action="store_true", help="Run PII sanitization.")
    ap.add_argument("--sanitize-output", type=Path, default=ROOT / "output" / "sanitized_text")
    ap.add_argument("--sanitize-limit", type=int, default=None)
    args = ap.parse_args()

    # 1) Build index
    index_items = build_index(AD_DIR, AD_INDEX)
    processed_count = 0
    failures: list[str] = []

    # 2) Process ad copy
    if args.limit:
        index_items = index_items[: args.limit]
    for entry in index_items:
        try:
            outcomes = process_ad_entry(
                entry,
                max_docling_pages=args.max_docling_pages,
                skip_convert_failures=args.skip_convert_failures,
                skip_pptx=args.skip_pptx,
            )
            for method, error, label in outcomes:
                name = Path(entry["name"]).name
                if error:
                    msg = f"[{label}/{method}] {name}: ERROR {error}"
                    print(msg)
                    failures.append(msg)
                else:
                    print(f"[{label}/{method}] {name}: ok")
            processed_count += 1
        except Exception as exc:
            msg = f"[fatal] {entry.get('name')}: {exc}"
            print(msg)
            failures.append(msg)

    # 3) Process website reviews
    web_files = sorted([p for p in WEB_DIR.glob("*") if p.is_file()], key=lambda p: p.name)
    if args.limit:
        web_files = web_files[: max(args.limit - processed_count, 0)]
    for path in web_files:
        rel_output = path.relative_to(ROOT)
        outcomes = process_source(path, rel_output, max_docling_pages=args.max_docling_pages, skip_convert_failures=args.skip_convert_failures)
        for method, error in outcomes:
            print(f"[website/{method}] {path.name}: {'ERROR ' + error if error else 'ok'}")
        processed_count += 1

    # 4) Failures Log
    if args.log_failures and failures:
        fail_path = OUTPUT_DIR / "failures.log"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text("\n".join(failures), encoding="utf-8")

    # 5) Sanitization
    if args.sanitize:
        sanitize_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "sanitize_pii.py"),
            "--input-root", str(OUTPUT_DIR / "data" / "ad_copy_reviews"),
            "--output-root", str(args.sanitize_output),
        ]
        if args.sanitize_limit:
            sanitize_cmd += ["--limit", str(args.sanitize_limit)]
        subprocess.run(sanitize_cmd, check=True)

    print(f"Processed {processed_count} files.")

if __name__ == "__main__":
    main()