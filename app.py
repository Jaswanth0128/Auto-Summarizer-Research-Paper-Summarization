from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import json

from upload import save_uploaded_file
from extract import run_extraction
from extractive import run_extractive_summarization
from polisher import run_semantic_polishing
from rewriter import run_rewriting

app = FastAPI(title="AI Research Summarizer v3")

app.mount("/shared_data", StaticFiles(directory="shared_data"), name="shared_data")
app.mount("/frontend",    StaticFiles(directory="frontend"),    name="frontend")

@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


# ── Directory setup ──────────────────────────────────────────
SHARED_PDF_DIR       = "shared_data/input_pdfs"
SHARED_IMG_DIR       = "shared_data/extracted_images"
SHARED_JSON_DIR      = "shared_data/intermediate_jsons"
SHARED_EXTRACTIVE_DIR = "shared_data/extractive_jsons"
SHARED_POLISHED_DIR  = "shared_data/polished_jsons"
SHARED_FINAL_DIR     = "shared_data/final_summaries"

for folder in [
    SHARED_PDF_DIR, SHARED_IMG_DIR, SHARED_JSON_DIR,
    SHARED_EXTRACTIVE_DIR, SHARED_POLISHED_DIR, SHARED_FINAL_DIR
]:
    os.makedirs(folder, exist_ok=True)


# ── Cleanup helper ───────────────────────────────────────────
def cleanup_temp_files(*file_paths):
    """Delete intermediate files after the response has been sent."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"[Cleanup] Removed: {path}")
        except Exception as e:
            print(f"[Cleanup] Error removing {path}: {e}")


# ── Main pipeline ────────────────────────────────────────────
@app.post("/api/process-paper")
async def process_paper(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    print(f"\n{'='*50}")
    print(f"PIPELINE START: {file.filename}")
    print(f"{'='*50}")

    # ── Phase 1: Upload ──────────────────────────────────────
    file_bytes = await file.read()
    pdf_path, uuid_prefix = save_uploaded_file(
        file_bytes, file.filename, upload_dir=SHARED_PDF_DIR
    )
    if not pdf_path:
        raise HTTPException(status_code=400, detail="Upload failed. Ensure file is a valid PDF.")

    # All downstream filenames share the same UUID prefix — prevents collisions
    # when two users upload files with the same name at the same time
    safe_name      = f"{uuid_prefix}_{file.filename.replace(' ', '_').replace('.pdf', '.json')}"
    extraction_path = os.path.join(SHARED_JSON_DIR,       safe_name)
    extractive_path = os.path.join(SHARED_EXTRACTIVE_DIR, f"ext_{safe_name}")
    polished_path   = os.path.join(SHARED_POLISHED_DIR,   f"pol_{safe_name}")
    final_path      = os.path.join(SHARED_FINAL_DIR,      f"fin_{safe_name}")

    # ── Phase 2: Extract (PDF → structured JSON) ─────────────
    print("\n[Phase 1] Extracting text, tables, and images from PDF...")
    if not run_extraction(pdf_path, extraction_path, SHARED_IMG_DIR):
        raise HTTPException(status_code=500, detail="Phase 1: Extraction failed.")

    # ── Phase 3: Extractive compression (BM25 + bucket mapping) ─
    print("\n[Phase 2] Running extractive summarization...")
    if not run_extractive_summarization(extraction_path, extractive_path, summary_ratio=0.3):
        raise HTTPException(status_code=500, detail="Phase 2: Extractive summarization failed.")

    # ── Phase 4: Semantic polishing (dedup, clean, transitions) ──
    print("\n[Phase 3] Running semantic polishing...")
    if not run_semantic_polishing(extractive_path, polished_path):
        raise HTTPException(status_code=500, detail="Phase 3: Semantic polishing failed.")

    # ── Phase 5: LLM rewriting (Phi-3 → numbered points) ─────
    print("\n[Phase 4] Running Phi-3 Mini rewriting...")
    if not run_rewriting(polished_path, final_path):
        raise HTTPException(status_code=500, detail="Phase 4: LLM rewriting failed.")

    # ── Read final result ─────────────────────────────────────
    try:
        with open(final_path, "r", encoding="utf-8") as f:
            final_report = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read final report: {e}")

    # ── Cleanup intermediate files in background ──────────────
    # final_path is kept so the frontend can re-fetch if needed.
    # Remove: pdf, extraction JSON, extractive JSON, polished JSON.
    background_tasks.add_task(
        cleanup_temp_files,
        pdf_path,
        extraction_path,
        extractive_path,
        polished_path,
    )

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE: {file.filename}")
    print(f"{'='*50}\n")

    return {
        "status": "success",
        "paper_title": final_report.get("Paper Title"),
        "summary": final_report.get("Final_Summary")
    }

# Pre-load Phi-3 at startup so the first request doesn't time out
@app.on_event("startup")
async def preload_models():
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    print("[Startup] Pre-loading Phi-3 Mini in background...")
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, _preload_phi3)

def _preload_phi3():
    try:
        from rewriter.phi3_rewriter import _get_pipeline
        _get_pipeline()
        print("[Startup] Phi-3 Mini loaded and ready.")
    except Exception as e:
        print(f"[Startup] Warning: Could not pre-load Phi-3: {e}")