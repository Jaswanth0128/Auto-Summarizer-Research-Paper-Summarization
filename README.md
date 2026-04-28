# 🧠 Auto-Summarizer — AI Research Paper Summarization

> Upload any research paper (PDF) and receive a structured, AI-generated technical summary — organized into **Objective, Methodology, Results, and Conclusion** — with extracted figures and tables, powered by an async job pipeline.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo Flow](#live-demo-flow)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Frontend Features](#frontend-features)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Overview

**Auto-Summarizer** is a full-stack AI application that processes academic research papers (PDFs) through a 4-phase NLP pipeline and returns a clean, structured summary organized into four semantic buckets:

| Bucket | What it captures |
|---|---|
| **Objective** | Research problem, motivation, goals, background |
| **Methodology** | System design, model architecture, training approach, datasets |
| **Results** | Metrics, accuracy values, benchmarks, performance comparisons, table data |
| **Conclusion** | Takeaways, limitations, future directions |

Each section is presented as numbered points with an optional toggle to view the raw extracted sentences. Extracted figures and tables are displayed alongside the relevant section.

The backend uses an **async job queue** — upload returns a `job_id` instantly, and the frontend polls for live step-by-step progress updates while the pipeline runs in the background.

---

## Live Demo Flow

```
User uploads PDF
      ↓
POST /api/process-paper  →  returns job_id immediately
      ↓
Frontend polls GET /api/process-paper/{job_id} every 3 seconds
      ↓
Phase 1 — PDF Extraction       (pymupdf4llm → structured JSON + PNG tables/charts)
      ↓
Phase 2 — Extractive Summary   (BM25 + semantic bucket classification → compressed JSON)
      ↓
Phase 3 — Semantic Polishing   (dedup, cleaning, pronoun grounding, transition injection)
      ↓
Phase 4 — LLM Rewriting        (Qwen 2.5 1.5B → numbered bullet points per section)
      ↓
Job status → "completed"  →  Frontend renders the final report with figures
```

---

## Architecture

```
auto-summarizer/
│
├── app.py                    # FastAPI server — async job queue, pipeline orchestration
│
├── upload/
│   └── uploader.py           # File validation (extension + magic bytes), UUID namespacing
│
├── extract/
│   └── extractor.py          # PDF → Markdown + images → structured JSON (pymupdf4llm)
│                             # Includes decorative image filtering
│
├── extractive/
│   └── BM25_extractive.py    # Section classification + BM25 sentence ranking
│                             # Metric-rich sentence prioritization for Results bucket
│
├── polisher/
│   └── semantic_polisher.py  # Sentence cleaning, dedup, transitions (sentence-transformers)
│
├── rewriter/
│   └── qwen_rewriter.py      # LLM rewriting to numbered points (Qwen 2.5 1.5B-Instruct)
│                             # Results bucket gets enhanced metric-preservation prompt rules
│
├── frontend/
│   ├── index.html            # Upload UI with drag-and-drop
│   ├── script.js             # Job submission, polling loop, render logic
│   └── style.css             # Clean, responsive UI
│
└── shared_data/              # Runtime I/O directories (auto-created)
    ├── input_pdfs/
    ├── extracted_images/
    ├── intermediate_jsons/
    ├── extractive_jsons/
    ├── polished_jsons/
    └── final_summaries/
```

---

## Pipeline Stages

### Phase 1 — PDF Extraction (`extract/extractor.py`)

Converts a PDF into a structured dictionary of sections using [`pymupdf4llm`](https://github.com/pymupdf/RAG).

- **Table extraction**: Uses `pymupdf` to locate table bounding boxes per page and saves high-resolution (300 DPI) PNG screenshots of each table.
- **Text & chart extraction**: Converts the full PDF to Markdown (including embedded chart images) via `pymupdf4llm.to_markdown()`.
- **Decorative image filtering** *(new in v3)*: A post-processing pass scans every extracted image and removes decorative assets — header/footer dividers, tiny logos, repeated line-like separators — based on aspect ratio, pixel dimensions, and repeat frequency across pages. Table screenshots (`table_page*`) are always preserved.
- **Section parsing**: Splits the Markdown on heading patterns (Roman numerals, `#` headers, bold numbered sections) to build a `{section_name: content}` dictionary.
- **Output**: A structured JSON file saved to `shared_data/intermediate_jsons/`.

---

### Phase 2 — Extractive Summarization (`extractive/BM25_extractive.py`)

Maps every section of the paper to one of the four semantic buckets, then selects the most relevant sentences using BM25.

**Bucket classification** uses two signals:
1. **Semantic similarity** — each paragraph chunk is embedded with `all-MiniLM-L6-v2` and compared against prototype descriptions for each bucket via cosine similarity.
2. **Header bonuses** — section headers like "Abstract", "Methodology", "Results" boost the score for the corresponding bucket by +0.25.

**BM25 ranking** uses an external query derived from each bucket's prototype description (not the document's own top terms). This avoids circular self-scoring.

**Results bucket enhancements** *(new in v3)*:
- A `_is_metric_rich()` detector identifies sentences and table rows containing keywords like `mAP`, `IoU`, `F1`, `accuracy`, `latency`, `FPS`, `TensorRT`, `Jetson` combined with numeric values. These receive a **2.0× quality multiplier**.
- Table rows within the Results section are extracted line-by-line and individually scored if metric-rich.
- Up to 5 metric-rich sentences are **guaranteed inclusion** in the top-N selection before filling remaining slots by BM25 score. This prevents narrative sentences from outranking actual benchmark numbers.

Additional features:
- Literature/related work sections are skipped entirely to reduce noise.
- Section weights applied: Abstract = 1.5×, References = 0.0×.
- Image paths and captions are extracted and attached to their respective bucket.
- Output: `shared_data/extractive_jsons/`

---

### Phase 3 — Semantic Polishing (`polisher/semantic_polisher.py`)

Cleans and refines the extracted sentences before they reach the LLM.

Steps applied per bucket:
1. **HTML & token stripping** — removes `<strong>`, `<|end|>`, `&amp;` and other artifacts.
2. **Math & formula filtering** — removes LaTeX, Greek symbols, and sentences with high symbol density.
3. **Sentence segmentation & blacklist filtering** — rejects references, figure captions, navigation sentences.
4. **Intra-bucket deduplication** — encodes sentences and drops those with cosine similarity > 0.72 to any already-kept sentence.
5. **Cross-bucket deduplication** — compares new bucket sentences against a global embedding pool built from all previously processed buckets.
6. **Pronoun grounding** — replaces leading "They" with "The researchers".
7. **Transition injection** — detects low-similarity sentence pairs and inserts transitions like "Turning to another aspect," at semantic topic shifts (cooldown of 8 sentences between injections).
8. **Paragraph formatting** — groups sentences into paragraphs based on embedding similarity.

Output: `shared_data/polished_jsons/`

---

### Phase 4 — LLM Rewriting (`rewriter/qwen_rewriter.py`)

Uses **Qwen 2.5 1.5B-Instruct** (CPU inference) to rewrite each polished section as numbered bullet points.

- Model loaded once at startup and cached as a singleton.
- **Results bucket enhanced prompt** *(new in v3)*:
  - Rule 8: Every concrete metric value (latency, FPS, throughput, table values) must be preserved exactly.
  - Rule 9: Exact numbers must never be replaced with vague comparatives like "better", "higher", or "improved".
  - Rule 10: Metric names and their values must be kept together as a unit in the output.
- **Output point cap raised** *(new in v3)*: Up to **12–15 points** per section (previously 4–7) to avoid losing key facts in information-dense sections.
- `dataset names` added to the preservation list alongside model names and metric values.
- Both the rewritten points and the polished raw text are saved so the frontend can toggle between them.
- Output: `shared_data/final_summaries/`

---

## Project Structure

```
.
├── app.py                      # FastAPI entrypoint — async job queue
├── upload/
│   ├── __init__.py
│   └── uploader.py             # PDF validation + UUID-namespaced saving
├── extract/
│   ├── __init__.py
│   └── extractor.py            # PDF → JSON + image extraction + decorative filter
├── extractive/
│   ├── __init__.py
│   └── BM25_extractive.py      # BM25 + semantic classification + metric prioritization
├── polisher/
│   ├── __init__.py
│   └── semantic_polisher.py    # Sentence cleaning + dedup + transitions
├── rewriter/
│   ├── __init__.py
│   └── qwen_rewriter.py        # Qwen 2.5 LLM rewriting + Results-enhanced prompt
├── frontend/
│   ├── index.html
│   ├── script.js               # Job submission + polling loop
│   └── style.css
└── shared_data/                # Auto-created at runtime
    ├── input_pdfs/
    ├── extracted_images/
    ├── intermediate_jsons/
    ├── extractive_jsons/
    ├── polished_jsons/
    └── final_summaries/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | [FastAPI](https://fastapi.tiangolo.com/) — async, job-queue architecture |
| **Async Runtime** | Python `asyncio` + `threading.Lock` for thread-safe job state |
| **PDF Parsing** | [pymupdf](https://pymupdf.readthedocs.io/) + [pymupdf4llm](https://github.com/pymupdf/RAG) |
| **Sentence Embeddings** | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| **BM25 Ranking** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) |
| **NLP / Spacy** | [spacy](https://spacy.io/) (`en_core_web_sm`) |
| **LLM** | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) via HuggingFace Transformers |
| **ML Backend** | [PyTorch](https://pytorch.org/) (CPU inference) |
| **Frontend** | Vanilla HTML/CSS/JS — polling-based job UI |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- ~4 GB disk space (for the Qwen 2.5 model, downloaded automatically on first run)
- No GPU required — runs entirely on CPU

### 1. Clone the repository

```bash
git clone https://github.com/Jaswanth0128/Auto-Summarizer-Research-Paper-Summarization.git
cd Auto-Summarizer-Research-Paper-Summarization
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn python-multipart
pip install pymupdf pymupdf4llm
pip install sentence-transformers rank-bm25
pip install spacy numpy
pip install torch transformers accelerate
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. (First run only) Download Qwen 2.5

The model (~3 GB) is downloaded automatically from HuggingFace on first run and cached locally. Ensure you have an internet connection on first startup.

---

## Running the App

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open your browser at:

```
http://localhost:8000
```

> **Note:** The first request after startup may take 30–90 seconds while Qwen 2.5 loads into memory. Subsequent requests are faster since the model is cached as a singleton.

---

## API Reference

### `POST /api/process-paper`

Submits a PDF for processing. Returns immediately with a job ID — the pipeline runs in the background.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `File` | A valid PDF file |

**Response:**
```json
{
  "job_id": "a3f9c21b4d...",
  "status": "queued",
  "step": "Queued for processing."
}
```

---

### `GET /api/process-paper/{job_id}`

Polls the status of a running or completed job. Poll every 3 seconds from the frontend.

**Response — while processing:**
```json
{
  "job_id": "a3f9c21b4d...",
  "status": "processing",
  "step": "Running extractive summarization.",
  "paper_title": null,
  "summary": null,
  "error": null,
  "created_at": "2025-01-01T10:00:00+00:00",
  "updated_at": "2025-01-01T10:00:15+00:00",
  "completed_at": null
}
```

**Response — on completion:**
```json
{
  "job_id": "a3f9c21b4d...",
  "status": "completed",
  "step": "Completed.",
  "paper_title": "Detected Title of the Paper",
  "summary": {
    "Objective":    { "points": ["..."], "raw_text": "...", "images": [] },
    "Methodology":  { "points": ["..."], "raw_text": "...", "images": [] },
    "Results":      { "points": ["..."], "raw_text": "...", "images": [] },
    "Conclusion":   { "points": ["..."], "raw_text": "...", "images": [] }
  },
  "created_at": "2025-01-01T10:00:00+00:00",
  "updated_at": "2025-01-01T10:01:05+00:00",
  "completed_at": "2025-01-01T10:01:05+00:00"
}
```

**Job status values:**

| Status | Meaning |
|---|---|
| `queued` | Job created, pipeline not yet started |
| `processing` | Pipeline is actively running |
| `completed` | All phases finished successfully |
| `failed` | A pipeline phase raised an error — see `error` field |

**HTTP error codes:**

| Code | Reason |
|---|---|
| `400` | File is not a valid PDF (extension or magic byte check failed) |
| `404` | Job ID not found |

---

## Frontend Features

- **Drag-and-drop upload** — drop a PDF onto the upload zone or click to browse.
- **Async job polling** — upload returns instantly; the page polls every 3 seconds for live step updates without blocking or timing out on slow machines.
- **Live step label** — displays the exact pipeline step currently running (`Extracting text...`, `Running semantic polishing.`, etc.) pulled directly from the job state on every poll.
- **Structured report rendering** — displays each bucket (Objective, Methodology, Results, Conclusion) as a card with numbered points.
- **Raw sentence toggle** — each section has a collapsible panel showing the original polished sentences before LLM rewriting.
- **Figure grid** — extracted tables and charts displayed in a responsive image grid with captions.
- **Resilient to page refresh** — because results are stored server-side by `job_id`, the page can reload without losing the result.

---

## Design Decisions

**Async job queue architecture** — the pipeline is fully decoupled from the HTTP request. `POST /api/process-paper` returns a `job_id` immediately. The pipeline runs in a background thread via `asyncio.to_thread()`. A thread-safe `jobs` dictionary protected by a `threading.Lock` stores all job state. This eliminates HTTP timeout issues on slow machines and handles concurrent uploads correctly.

**Job lifecycle timestamps** — every job tracks `created_at`, `updated_at`, and `completed_at` as UTC ISO strings, enabling processing duration calculation and per-job audit trails.

**Decorative image filtering** — Phase 1 post-processes extracted images using pixel dimensions and aspect ratio heuristics. Line-like separators (aspect ratio ≥ 14.0, short edge ≤ 60px), images smaller than 220×220px, and images repeated across pages (header/footer dividers) are automatically removed and deleted from disk. Table screenshots are always preserved.

**Metric-rich sentence prioritization** — the Results bucket uses a `_is_metric_rich()` detector that checks for metric keywords combined with numeric values. These sentences receive a 2.0× quality multiplier and up to 5 are guaranteed inclusion in the extractive selection, preventing narrative sentences from outranking actual benchmark data in BM25 scoring.

**Results-enhanced LLM prompt** — three additional rules in the Qwen 2.5 prompt ensure every concrete metric value is preserved verbatim and never replaced with vague comparatives. The output point cap is raised to 12–15 for information-dense sections.

**UUID-prefixed filenames** — every uploaded file is prefixed with an 8-character UUID hex to prevent collisions when concurrent users upload identically named files.

**Magic byte validation** — the uploader checks for `%PDF` magic bytes in addition to the file extension, rejecting renamed non-PDF files.

**External BM25 queries** — the BM25 query is derived from each bucket's prototype description, not the document's own top terms. Self-referential queries cause BM25 to re-score frequent document words rather than selecting semantically relevant sentences.

**Singleton model loading** — `all-MiniLM-L6-v2`, spaCy, and Qwen 2.5 are each loaded once and reused across all requests via module-level globals. This prevents multi-second reload penalties on every upload.

---

## Known Limitations

- **CPU-only inference** — Qwen 2.5 runs on CPU, meaning rewriting takes 30–90 seconds per paper. A GPU would reduce this to a few seconds.
- **In-memory job store** — the `jobs` dictionary lives in process memory. Restarting the server clears all job history. A persistent store (Redis, SQLite) would be needed for production deployments.
- **English only** — the pipeline is tuned for English-language papers. Non-English PDFs will produce degraded output.
- **Multi-column PDFs** — some two-column academic PDFs may have sentence order scrambled during Markdown extraction.
- **Heavy math papers** — papers dominated by equations (e.g., pure mathematics or signal processing) may produce sparse summaries after math filtering.
- **Model download on first run** — the Qwen 2.5 model (~3 GB) requires a one-time internet connection to download.

---

## Future Work

- [ ] Add GPU support via `device_map="cuda"` for faster LLM inference
- [ ] Persist job store to SQLite or Redis for server-restart resilience
- [ ] Support batch processing of multiple papers in a single session
- [ ] Add export options (PDF report, Markdown download)
- [ ] Citation extraction and reference linking
- [ ] Support for additional languages via multilingual embedding models
- [ ] Docker container for one-command deployment
- [ ] User session management and summary history

---

## License

This project is intended for educational and research purposes.

---

## Author

**Jaswanth** — [@Jaswanth0128](https://github.com/Jaswanth0128)
