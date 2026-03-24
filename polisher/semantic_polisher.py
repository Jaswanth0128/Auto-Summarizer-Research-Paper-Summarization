import json
import os
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

# ── Singleton loaders ────────────────────────────────────────
_embed_model = None
_nlp = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def _get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── HTML / entity / token cleaner ───────────────────────────
def _strip_html(text: str) -> str:
    """
    Remove HTML tags, HTML entities, and model control tokens
    from any text string. Applied at sentence level so nothing
    leaks into downstream stages.
    """
    # Remove HTML tags: <strong>, </strong>, <b>, <br />, etc.
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    # Remove model control tokens: <|end|>, <|user|>, <|cut|>, etc.
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return text.strip()


# ── Step 1: Clean raw text ───────────────────────────────────
def _clean_text(text: str) -> str:
    if not text or text == "Data not found for this section.":
        return text

    # Strip HTML first before any other processing
    text = _strip_html(text)

    text = re.sub(r'_\*\*|\*\*_|_\*|\*_', '', text)
    text = re.sub(r'\*{1,2}|_{1,2}', '', text)
    text = re.sub(r'\bABSTRACT\s*:\s*', '', text)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\[a-z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-z]+', '', text)

    text = re.sub(r'\[\s*[\+\-\=\(\)\[\]\{\}/\\<>]\s*\]', '', text)
    text = re.sub(r'\[\s*[a-z]\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\u2200-\u22FF]', '', text)
    text = re.sub(r'[\u2300-\u23FF]', '', text)
    text = re.sub(r'[\u0370-\u03FF]', '', text)
    text = re.sub(r'[\u2100-\u214F]', '', text)
    text = re.sub(r'[√∥⟨⟩≤≥±∓∞∑∏∫∂∇≈≠≡∝∧∨¬←→↔⇒⇐⇔]', '', text)
    text = re.sub(r'\b[a-zA-Z]\s*[\(\[]\s*[a-zA-Z]\s*[\)\]]\s*=', '', text)
    text = re.sub(r'(?:^|\s)[a-zA-Z]{1,2}\s*=\s*[^.]{0,50}(?=[.;,]|$)', ' ', text, flags=re.MULTILINE)

    text = re.sub(r'\b\d{1,3}\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\b', '', text)
    text = re.sub(r'\[\d+(?:\s*[,\-]\s*\d+)*\]', '', text)
    text = re.sub(r'\([A-Za-z\s]+(?:et\s+al\.?)?,\s*\d{4}\)', '', text)
    text = re.sub(r'!\[\]\(.*?\)', '', text)
    text = re.sub(r'\|[^\n]*\|', ' ', text)
    text = re.sub(r'[•·]', '', text)
    text = re.sub(r'\[\s*[A-Z]\s*\]', '', text)
    text = re.sub(r'---+', ' ', text)
    text = re.sub(r'~~[^~]*~~', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Step 2: Split into sentences and filter junk ─────────────
BLACKLIST = ["et al.", "vol.", "proc.", "ieee", "arxiv", "journal", "doi", "issn"]

_PAGE_HEADER_RE = re.compile(
    r'\b[A-Z][A-Za-z\s]+(?:System|Detection|Method|Approach|Framework|Network|Model|Analysis|Learning)\s+\d{1,3}\s+(?:Figure|Fig|Table)',
    re.IGNORECASE
)
_NAVIGATION_RE = re.compile(
    r'(?:we\s+)?(?:describe|provide|present|discuss|outline|detail|give)\s+(?:an?\s+)?'
    r'(?:in-depth\s+)?(?:description|overview|details?|summary)\s+(?:of\s+)?.*?(?:in\s+)?section\s+\d',
    re.IGNORECASE
)

def _is_math_heavy(s: str) -> bool:
    math_chars = sum(1 for c in s if c in '=+<>{}()[]^/\\~')
    single_letter_vars = len(re.findall(r'(?<![A-Za-z])[a-zA-Z](?![A-Za-z])', s))
    total_words = len(s.split())
    if total_words == 0:
        return True
    if math_chars / max(len(s), 1) > 0.06:
        return True
    if single_letter_vars / max(total_words, 1) > 0.20:
        return True
    if re.search(r'(?:Eq\.?\s*\(\d+\)|\bwhere\b.*=|[a-z]\s*\(\s*[a-z]\s*\)\s*=)', s, re.IGNORECASE):
        return True
    if re.match(r'^[\.,;:\d\s\-\+\=\(\)]+\s*(?:where|and|for)', s):
        return True
    return False

def _extract_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    valid = []
    for s in sentences:
        # Strip HTML from every individual sentence before validation
        s = _strip_html(s).strip()
        if len(s.split()) <= 5:
            continue
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.55:
            continue
        if any(bad in s.lower() for bad in BLACKLIST):
            continue
        if re.match(r'^(fig(?:ure)?|table)\s*[:\.\d]', s, re.IGNORECASE):
            continue
        if re.match(r'^keywords\s*:', s, re.IGNORECASE):
            continue
        if s.count('|') >= 2:
            continue
        if _is_math_heavy(s):
            continue
        if _PAGE_HEADER_RE.search(s):
            continue
        if _NAVIGATION_RE.search(s):
            continue
        if s and s[0].islower() and not s.startswith('i '):
            continue
        if s and not s[0].isalpha():
            continue
        valid.append(s)
    return valid


# ── Step 3: Remove redundant sentences ──────────────────────
def _remove_redundancy(sentences: list, threshold: float = 0.72) -> list:
    if len(sentences) <= 1:
        return sentences

    model = _get_embed_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    keep = []
    keep_embeddings = []

    for sent, emb in zip(sentences, embeddings):
        if not keep_embeddings:
            keep.append(sent)
            keep_embeddings.append(emb)
            continue
        sims = util.cos_sim(emb.unsqueeze(0), _torch_stack(keep_embeddings))[0]
        if float(sims.max()) < threshold:
            keep.append(sent)
            keep_embeddings.append(emb)

    return keep


def _torch_stack(tensors):
    import torch
    return torch.stack(list(tensors))


# ── Step 4: Pronoun grounding ────────────────────────────────
def _ground_pronouns(sentences: list) -> list:
    result = []
    for sent in sentences:
        sent = re.sub(r'^They\b', 'The researchers', sent)
        result.append(sent)
    return result


# ── Step 5: Transition injection ────────────────────────────
_TRANSITION_RE = re.compile(
    r'^(However|Moreover|Additionally|Furthermore|In contrast|Specifically|'
    r'Consequently|Therefore|Meanwhile|Notably|In particular|'
    r'On the other hand|As a result|In addition|For instance|'
    r'Similarly|Conversely|Nevertheless|Accordingly)',
    re.IGNORECASE
)

TOPIC_SHIFT_TRANSITIONS = [
    "Turning to another aspect,", "On a related note,", "In a different vein,",
    "From a different perspective,", "Regarding another dimension,",
    "Looking at this differently,", "Considering another factor,",
]

def _inject_transitions(sentences: list, used_transitions: set = None) -> list:
    if used_transitions is None:
        used_transitions = set()
    if len(sentences) <= 3:
        return sentences

    model = _get_embed_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)

    sims = [
        float(util.cos_sim(embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0))[0][0])
        for i in range(1, len(sentences))
    ]
    if not sims:
        return sentences

    sim_threshold = float(np.percentile(sims, 10))
    rng = np.random.default_rng(42)
    available = [t for t in TOPIC_SHIFT_TRANSITIONS if t not in used_transitions]
    if not available:
        return sentences

    result = [sentences[0]]
    last_injection_idx = -10

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sim = sims[i - 1]
        already_has = bool(_TRANSITION_RE.match(sent))
        cooldown_ok = (i - last_injection_idx) >= 8

        if not already_has and cooldown_ok and sim <= sim_threshold and available:
            trans = rng.choice(available)
            sent = f"{trans} {sent[0].lower()}{sent[1:]}"
            last_injection_idx = i
            available.remove(trans)
            used_transitions.add(trans)

        result.append(sent)

    return result


# ── Step 6: Format into paragraphs ──────────────────────────
def _format_paragraphs(sentences: list) -> str:
    if not sentences:
        return "Data not found for this section."
    if len(sentences) <= 3:
        return " ".join(sentences)

    model = _get_embed_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)

    paragraphs = []
    current_para = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = float(util.cos_sim(
            embeddings[i-1].unsqueeze(0), embeddings[i].unsqueeze(0)
        )[0][0])
        if sim < 0.35 and len(current_para) >= 2:
            paragraphs.append(" ".join(current_para))
            current_para = [sentences[i]]
        else:
            current_para.append(sentences[i])

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


# ── Cross-bucket deduplication ───────────────────────────────
def _cross_bucket_dedup(sentences: list, global_embeddings: list,
                         threshold: float = 0.72) -> list:
    if not global_embeddings or not sentences:
        return sentences
    model = _get_embed_model()
    new_embs = model.encode(sentences, convert_to_tensor=True)
    global_stack = _torch_stack(global_embeddings)
    keep = []
    for sent, emb in zip(sentences, new_embs):
        sims = util.cos_sim(emb.unsqueeze(0), global_stack)[0]
        if float(sims.max()) < threshold:
            keep.append(sent)
    return keep


# ── Master pipeline per bucket ───────────────────────────────
def polish_bucket(bucket_name: str, raw_text: str,
                  used_transitions: set = None,
                  global_embeddings: list = None) -> str:
    """
    Clean → strip HTML → split → intra-dedup → cross-dedup → ground → transitions → format
    """
    if not raw_text or raw_text == "Data not found for this section.":
        return raw_text

    if used_transitions is None:
        used_transitions = set()
    if global_embeddings is None:
        global_embeddings = []

    print(f"  [Polisher] Processing {bucket_name}...")

    cleaned = _clean_text(raw_text)
    sentences = _extract_sentences(cleaned)
    if not sentences:
        return raw_text
    print(f"    Sentences extracted: {len(sentences)}")

    sentences = _remove_redundancy(sentences, threshold=0.72)
    print(f"    After intra-bucket dedup: {len(sentences)}")

    if global_embeddings:
        sentences = _cross_bucket_dedup(sentences, global_embeddings, threshold=0.72)
        print(f"    After cross-bucket dedup: {len(sentences)}")

    if not sentences:
        return raw_text

    model = _get_embed_model()
    new_embs = model.encode(sentences, convert_to_tensor=True)
    global_embeddings.extend(new_embs)

    sentences = _ground_pronouns(sentences)
    sentences = _inject_transitions(sentences, used_transitions)
    result = _format_paragraphs(sentences)

    return result


# ── Entry point ──────────────────────────────────────────────
BUCKET_PRIORITY = ["Objective", "Results", "Conclusion", "Methodology"]

def run_semantic_polishing(input_json_path: str, output_json_path: str) -> bool:
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        final_output = {
            "Paper Title": data.get("Paper Title", "Unknown Title"),
            "Polished_Summary": {}
        }

        extractive = data.get("Extractive_Summary", {})
        used_transitions = set()
        global_embeddings = []

        for bucket_name in BUCKET_PRIORITY:
            content = extractive.get(bucket_name)
            if not content:
                final_output["Polished_Summary"][bucket_name] = {
                    "text": "Data not found for this section.", "images": []
                }
                continue

            raw_text = content.get("text", "")
            polished_text = polish_bucket(
                bucket_name, raw_text,
                used_transitions=used_transitions,
                global_embeddings=global_embeddings
            )
            final_output["Polished_Summary"][bucket_name] = {
                "text": polished_text,
                "images": content.get("images", [])
            }

        for bucket_name, content in extractive.items():
            if bucket_name not in BUCKET_PRIORITY:
                final_output["Polished_Summary"][bucket_name] = {
                    "text": content.get("text", ""),
                    "images": content.get("images", [])
                }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        return True

    except Exception as e:
        print(f"[Polisher] Critical Error: {e}")
        import traceback; traceback.print_exc()
        return False