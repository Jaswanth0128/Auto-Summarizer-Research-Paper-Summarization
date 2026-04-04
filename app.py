import asyncio
import json
import os
import threading
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from extract import run_extraction
from extractive import run_extractive_summarization
from polisher import run_semantic_polishing
from rewriter import run_rewriting
from upload import save_uploaded_file

app = FastAPI(title="AI Research Summarizer v3")

app.mount("/shared_data", StaticFiles(directory="shared_data"), name="shared_data")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


SHARED_PDF_DIR = "shared_data/input_pdfs"
SHARED_IMG_DIR = "shared_data/extracted_images"
SHARED_JSON_DIR = "shared_data/intermediate_jsons"
SHARED_EXTRACTIVE_DIR = "shared_data/extractive_jsons"
SHARED_POLISHED_DIR = "shared_data/polished_jsons"
SHARED_FINAL_DIR = "shared_data/final_summaries"

for folder in [
    SHARED_PDF_DIR,
    SHARED_IMG_DIR,
    SHARED_JSON_DIR,
    SHARED_EXTRACTIVE_DIR,
    SHARED_POLISHED_DIR,
    SHARED_FINAL_DIR,
]:
    os.makedirs(folder, exist_ok=True)


jobs = {}
jobs_lock = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_temp_files(*file_paths):
    """Delete temporary input files after the response has been sent."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"[Cleanup] Removed: {path}")
        except Exception as exc:
            print(f"[Cleanup] Error removing {path}: {exc}")


def update_job(job_id: str, **fields):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = utc_now_iso()


def serialize_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return {}
        return {
            "job_id": job_id,
            "status": job["status"],
            "step": job.get("step"),
            "paper_title": job.get("paper_title"),
            "summary": job.get("summary"),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "completed_at": job.get("completed_at"),
        }


def _run_pipeline_job(job_id: str, pdf_path: str, original_filename: str, uuid_prefix: str):
    print(f"\n{'=' * 50}")
    print(f"PIPELINE START [{job_id}]: {original_filename}")
    print(f"{'=' * 50}")

    safe_name = f"{uuid_prefix}_{original_filename.replace(' ', '_').replace('.pdf', '.json')}"
    extraction_path = os.path.join(SHARED_JSON_DIR, safe_name)
    extractive_path = os.path.join(SHARED_EXTRACTIVE_DIR, f"ext_{safe_name}")
    polished_path = os.path.join(SHARED_POLISHED_DIR, f"pol_{safe_name}")
    final_path = os.path.join(SHARED_FINAL_DIR, f"fin_{safe_name}")

    try:
        update_job(job_id, status="processing", step="Extracting text, tables, and images from PDF.")
        if not run_extraction(pdf_path, extraction_path, SHARED_IMG_DIR):
            raise RuntimeError("Phase 1: Extraction failed.")

        update_job(job_id, step="Running extractive summarization.")
        if not run_extractive_summarization(extraction_path, extractive_path, summary_ratio=0.3):
            raise RuntimeError("Phase 2: Extractive summarization failed.")

        update_job(job_id, step="Running semantic polishing.")
        if not run_semantic_polishing(extractive_path, polished_path):
            raise RuntimeError("Phase 3: Semantic polishing failed.")

        update_job(job_id, step="Running Qwen rewriting.")
        if not run_rewriting(polished_path, final_path):
            raise RuntimeError("Phase 4: LLM rewriting failed.")

        update_job(job_id, step="Loading final report.")
        try:
            with open(final_path, "r", encoding="utf-8") as file_obj:
                final_report = json.load(file_obj)
        except Exception as exc:
            raise RuntimeError(f"Failed to read final report: {exc}") from exc

        cleanup_temp_files(pdf_path)
        update_job(
            job_id,
            status="completed",
            step="Completed.",
            paper_title=final_report.get("Paper Title"),
            summary=final_report.get("Final_Summary"),
            completed_at=utc_now_iso(),
        )

        print(f"\n{'=' * 50}")
        print(f"PIPELINE COMPLETE [{job_id}]: {original_filename}")
        print(f"{'=' * 50}\n")

    except Exception as exc:
        print(f"[Pipeline] Job {job_id} failed: {exc}")
        update_job(
            job_id,
            status="failed",
            step="Failed.",
            error=str(exc),
            completed_at=utc_now_iso(),
        )


@app.post("/api/process-paper")
async def process_paper(file: UploadFile = File(...)):
    file_bytes = await file.read()
    pdf_path, uuid_prefix = save_uploaded_file(file_bytes, file.filename, upload_dir=SHARED_PDF_DIR)
    if not pdf_path:
        raise HTTPException(status_code=400, detail="Upload failed. Ensure file is a valid PDF.")

    job_id = uuid.uuid4().hex
    now = utc_now_iso()
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "step": "Queued for processing.",
            "paper_title": None,
            "summary": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }

    asyncio.create_task(asyncio.to_thread(_run_pipeline_job, job_id, pdf_path, file.filename, uuid_prefix))
    return {"job_id": job_id, "status": "queued", "step": "Queued for processing."}


@app.get("/api/process-paper/{job_id}")
async def get_process_paper_status(job_id: str):
    payload = serialize_job(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found.")
    return payload


@app.on_event("startup")
async def preload_models():
    print("[Startup] Pre-loading Qwen in background...")
    asyncio.create_task(asyncio.to_thread(_preload_qwen))


def _preload_qwen():
    try:
        from rewriter.qwen_rewriter import _get_pipeline

        _get_pipeline()
        print("[Startup] Qwen loaded and ready.")
    except Exception as exc:
        print(f"[Startup] Warning: Could not pre-load Qwen: {exc}")
