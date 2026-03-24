import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer, util

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[Extractive] Error: 'rank_bm25' not installed. Run: pip install rank-bm25")

ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'we', 'our', 'the', 'is', 'in', 'and', 'of',
    'to', 'a', 'for', 'with', 'on', 'as', 'by', 'this', 'that',
    'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
    'it', 'its', 'at', 'an', 'or', 'but', 'not', 'from', 'also'
}

# ─────────────────────────────────────────────────────────────
# Semantic bucket definitions
# These serve TWO roles now:
#   1. Prototype text for embedding-based classification (same as v2)
#   2. BM25 query source — FIX: query now comes from here, not the document itself
# ─────────────────────────────────────────────────────────────
BUCKET_PROTOTYPES = {
    "Objective": (
        "This section describes the research problem, motivation, goals, hypothesis, "
        "background context, gap in the literature, and the objective the authors aim to achieve. "
        "It introduces why the research was conducted and what problem it solves."
    ),
    "Methodology": (
        "This section explains the proposed method, system architecture, algorithm design, "
        "dataset preparation, model training process, implementation details, and technical approach. "
        "It describes how the system works, preprocessing steps, and the tools or frameworks used."
    ),
    "Results": (
        "This section presents experimental results, evaluation metrics, accuracy numbers, "
        "performance comparisons, test outcomes, detection rates, precision, recall, F1 scores, "
        "training accuracy, validation accuracy, and quantitative analysis of how well the system performed."
    ),
    "Conclusion": (
        "This section summarizes what the paper achieved, discusses practical limitations, "
        "suggests future work and improvements, and provides the overall conclusion. "
        "It reflects on the significance of the results and potential real-world applications."
    ),
}

# Pre-tokenized BM25 queries derived from BUCKET_PROTOTYPES (not from the document)
# This is the key fix from v2 — the query is now an external signal, not self-referential
BUCKET_BM25_QUERIES = {
    name: [
        w for w in re.findall(r'\b\w+\b', proto.lower())
        if w not in ENGLISH_STOPWORDS and len(w) > 3
    ]
    for name, proto in BUCKET_PROTOTYPES.items()
}

HEADER_BONUS = {
    "Objective":    ["abstract", "introduction", "background", "objective", "problem statement", "existing system", "front matter"],
    "Methodology":  ["methodology", "method", "approach", "proposed system", "system overview", "architecture", "dataset", "training", "implementation"],
    "Results":      ["result", "evaluation", "experiment", "performance", "ablation", "output"],
    "Conclusion":   ["conclusion", "future work", "discussion", "summary", "limitations"],
}

HEADING_PATTERN = re.compile(
    r'\n(?:[IVXLC]+\.?\s*)?(ABSTRACT|INTRODUCTION|BACKGROUND|OBJECTIVE|EXISTING SYSTEM|'
    r'PROPOSED SYSTEM|SYSTEM OVERVIEW|METHODOLOGY|METHODS?|APPROACH|DATASET PREPARATION|'
    r'MODEL TRAINING|EVALUATION|EXPERIMENTS?|RESULTS?|DISCUSSION|CONCLUSION|FUTURE WORK)\b',
    flags=re.IGNORECASE
)

# Singleton model — loaded once, reused across all requests
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[Extractive] Loading sentence-transformers model (one-time)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Extractive] Embedding model ready.")
    return _embed_model


def _classify_chunk(chunk_embedding, prototype_embeddings: dict, header: str = "") -> Tuple[str, float]:
    """Classify a single chunk embedding against all bucket prototypes."""
    scores = {}
    for bucket_name, proto_emb in prototype_embeddings.items():
        sim = float(util.cos_sim(chunk_embedding, proto_emb)[0][0])
        if header:
            header_lower = header.lower()
            if any(kw in header_lower for kw in HEADER_BONUS.get(bucket_name, [])):
                sim += 0.25
        scores[bucket_name] = sim
    best_bucket = max(scores, key=scores.get)
    return best_bucket, scores[best_bucket]


def _split_into_paragraphs(text: str, sentences_per_chunk: int = 4) -> List[str]:
    """Split text into paragraph-sized chunks for classification."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 4]
    if not sentences:
        return [text] if text.strip() else []
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if len(chunk.split()) > 10:
            paragraphs.append(chunk)
    return paragraphs if paragraphs else [text]


def _get_section_weight(header: str) -> float:
    """Section importance weight applied to BM25 scores."""
    h = header.lower()
    if any(k in h for k in ["abstract", "conclusion", "future work", "summary"]):      return 1.5
    if any(k in h for k in ["methodology", "method", "proposed system",
                              "architecture", "approach", "implementation"]):           return 1.4
    if any(k in h for k in ["introduction"]):                                          return 1.3
    if any(k in h for k in ["result", "evaluation", "experiment", "performance"]):     return 1.1
    if any(k in h for k in ["background"]):                                            return 0.9
    if any(k in h for k in ["literature", "related work", "existing system"]):         return 0.6
    if any(k in h for k in ["reference", "bibliography"]):                             return 0.0
    return 1.0


def map_to_4_buckets(extracted_data: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
    buckets = {"Objective": [], "Methodology": [], "Results": [], "Conclusion": []}

    model = _get_embed_model()
    prototype_embeddings = {
        name: model.encode(desc, convert_to_tensor=True)
        for name, desc in BUCKET_PROTOTYPES.items()
    }

    for raw_header, text in extracted_data.items():
        if raw_header == "Paper Title" or not text.strip():
            continue
        if re.search(r'\b(references?|bibliography)\b', raw_header, re.IGNORECASE):
            continue
        if re.search(r'\b(literature|related\s+work|survey|prior\s+work)\b', raw_header, re.IGNORECASE):
            print(f"  [Bucket] Skipping '{raw_header}' (literature/related work)")
            continue

        # ── Front Matter / Abstract ──────────────────────────────────────────
        # FIX from v2: merged the two identical classification loops into one pass
        if re.search(r'\b(front\s*matter|abstract)\b', raw_header, re.IGNORECASE):
            paragraphs = _split_into_paragraphs(text, sentences_per_chunk=3)
            if paragraphs:
                para_embeddings = model.encode(paragraphs, convert_to_tensor=True)
                para_buckets = {}

                # Single loop — classifies AND logs in one pass (was two identical loops in v2)
                for para_text, para_emb in zip(paragraphs, para_embeddings):
                    scores = {}
                    for bn, proto_emb in prototype_embeddings.items():
                        sim = float(util.cos_sim(para_emb, proto_emb)[0][0])
                        if bn == "Objective":
                            sim += 0.20   # abstract bias
                        scores[bn] = sim
                    best = max(scores, key=scores.get)
                    buckets[best].append((para_text, 1.5))
                    para_buckets[best] = para_buckets.get(best, 0) + 1

                dist = ", ".join(f"{k}: {v}" for k, v in sorted(para_buckets.items()))
                print(f"  [Bucket] '{raw_header}' ({len(paragraphs)} chunks, abstract-biased) → {dist}")
            continue

        # ── Hard-cut references embedded inside a section ────────────────────
        ref_split = re.split(
            r'\n(?:[IVXLC]+\.?\s*)?(?:REFERENCES|BIBLIOGRAPHY)\b',
            text, maxsplit=1, flags=re.IGNORECASE
        )
        text = ref_split[0]

        # ── Split by embedded headings if present ────────────────────────────
        matches = list(HEADING_PATTERN.finditer("\n" + text))
        sub_sections = []
        if len(matches) > 1:
            for i in range(len(matches)):
                start_idx = matches[i].end()
                end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sub_header = matches[i].group(1)
                sub_text = text[start_idx:end_idx].strip()
                if len(sub_text.split()) > 10:
                    sub_sections.append((sub_header, sub_text))
        else:
            sub_sections.append((raw_header, text))

        # ── Classify each sub-section paragraph by paragraph ─────────────────
        for header, section_text in sub_sections:
            weight = _get_section_weight(header)
            paragraphs = _split_into_paragraphs(section_text, sentences_per_chunk=4)
            if not paragraphs:
                continue

            para_embeddings = model.encode(paragraphs, convert_to_tensor=True)
            para_buckets = {}
            for para_text, para_emb in zip(paragraphs, para_embeddings):
                bucket_name, _ = _classify_chunk(para_emb, prototype_embeddings, header)
                buckets[bucket_name].append((para_text, weight))
                para_buckets[bucket_name] = para_buckets.get(bucket_name, 0) + 1

            dist = ", ".join(f"{k}: {v}" for k, v in sorted(para_buckets.items()))
            print(f"  [Bucket] '{header}' ({len(paragraphs)} chunks) → {dist}")

    return buckets


def extract_image_metadata(text: str):
    image_list = []
    for match in re.finditer(r'!\[\]\((.*?)\)', text):
        img_path = match.group(1)
        context = text[match.end():match.end() + 150]
        caption_match = re.search(
            r'(Fig(?:ure)?\.?\s*\d+.*?|Table\s*\d+.*?)(\n|$)', context, re.IGNORECASE
        )
        caption = caption_match.group(1).strip('_* ') if caption_match else "Figure/Table"
        image_list.append({"path": img_path, "caption": caption, "tag": match.group(0)})
    return image_list


_FILLER_PATTERNS = [
    re.compile(r'^it is worth noting that', re.IGNORECASE),
    re.compile(r'^this is (why|because|actually)', re.IGNORECASE),
    re.compile(r'^they all work together', re.IGNORECASE),
    re.compile(r'rather than following all the patterns', re.IGNORECASE),
    re.compile(r'^this makes us obtain', re.IGNORECASE),
    re.compile(r'^learning here is not only', re.IGNORECASE),
    re.compile(r'^you will be able to notice', re.IGNORECASE),
    re.compile(r'constructs a useful structure', re.IGNORECASE),
]


def _sentence_quality(text: str) -> float:
    """Quality multiplier [0.0–1.3]. Penalises vague/filler, rewards metric sentences."""
    words = text.split()
    if len(words) <= 8:
        return 0.3
    for pat in _FILLER_PATTERNS:
        if pat.search(text):
            return 0.1
    has_numbers = bool(re.search(
        r'\d+\.?\d*\s*(%|per\s*cent|accuracy|precision|recall|F1)', text, re.IGNORECASE
    ))
    if has_numbers:
        return 1.3
    return 1.0


def extract_top_sentences(chunks_with_weights: List[Tuple[str, float]],
                          bucket_name: str,
                          summary_ratio: float = 0.3) -> str:
    """
    Select top sentences using BM25.

    FIX from v2: BM25 query now comes from BUCKET_BM25_QUERIES[bucket_name]
    (words from the bucket's prototype description) instead of the document's
    own most-frequent terms. This gives BM25 an external, meaningful signal
    rather than circular self-scoring.
    """
    if not chunks_with_weights:
        return "Data not found for this section."

    all_sentences = []
    for chunk_text, weight in chunks_with_weights:
        clean_text = re.sub(r'!\[\]\(.*?\)', '', chunk_text)
        for s in re.split(r'(?<=[.!?])\s+', clean_text):
            s = s.strip()
            if len(s.split()) > 5:
                all_sentences.append({"text": s, "weight": weight})

    if len(all_sentences) <= 5:
        return " ".join(s["text"] for s in all_sentences)

    tokenized = [re.findall(r'\b\w+\b', s["text"].lower()) for s in all_sentences]

    # KEY FIX: use the bucket's prototype words as query, not document top-terms
    query = BUCKET_BM25_QUERIES.get(bucket_name, [])
    if not query:
        # Fallback: top document terms (old behaviour), only if prototype query is empty
        all_tokens = [t for seq in tokenized for t in seq if t not in ENGLISH_STOPWORDS]
        query = [term for term, _ in Counter(all_tokens).most_common(20)]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query)

    scored = []
    for i, s_data in enumerate(all_sentences):
        quality = _sentence_quality(s_data["text"])
        final_score = scores[i] * s_data["weight"] * quality
        scored.append({"text": s_data["text"], "score": final_score, "index": i})

    scored.sort(key=lambda x: x["score"], reverse=True)
    max_sentences = min(12, max(4, int(len(all_sentences) * summary_ratio)))
    top_n = scored[:max_sentences]

    # Restore original order (preserve paper's logical sequence)
    top_n.sort(key=lambda x: x["index"])

    return " ".join(s["text"] for s in top_n)


def run_extractive_summarization(input_json_path: str,
                                  output_json_path: str,
                                  summary_ratio: float = 0.3) -> bool:
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)

        structured_buckets = map_to_4_buckets(extracted_data)
        compressed_summary = {
            "Paper Title": extracted_data.get("Paper Title", "Unknown Title"),
            "Extractive_Summary": {}
        }

        for name, chunks in structured_buckets.items():
            if not chunks:
                compressed_summary["Extractive_Summary"][name] = {
                    "text": "Data not found for this section.", "images": []
                }
                continue

            full_raw_text = "\n\n".join(c[0] for c in chunks)
            images = extract_image_metadata(full_raw_text)

            # Pass bucket_name so BM25 uses the right external query
            summary_text = extract_top_sentences(chunks, bucket_name=name,
                                                  summary_ratio=summary_ratio)
            compressed_summary["Extractive_Summary"][name] = {
                "text": summary_text, "images": images
            }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(compressed_summary, f, ensure_ascii=False, indent=4)
        return True

    except Exception as e:
        print(f"[Extractive] Error: {e}")
        import traceback; traceback.print_exc()
        return False
