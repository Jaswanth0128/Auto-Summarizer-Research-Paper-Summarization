import json
import os
import re

# ── Singleton model loader ────────────────────────────────────
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("[Rewriter] Loading Qwen 2.5 1.5B (first run downloads ~3 GB, cached after)...")
        try:
            import torch
            from transformers import pipeline, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                trust_remote_code=False
            )

            _pipeline = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                tokenizer=tokenizer,
                trust_remote_code=False,
                dtype=torch.float32,
                device_map="cpu",
            )

            torch.set_num_threads(8)
            print("[Rewriter] Qwen 2.5 1.5B ready.")

        except Exception as e:
            print(f"[Rewriter] Failed to load model: {e}")
            print("[Rewriter] Run: pip install transformers accelerate torch")
            raise

    return _pipeline


# ── HTML and token cleaner ────────────────────────────────────
def _clean_text(text: str) -> str:
    """Strip HTML tags, HTML entities, and model control tokens from any text."""
    # Remove HTML tags like <strong>, </strong>, <b>, <br>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    # Remove model control tokens like <|end|>, <|user|>, <|assistant|>, <|cut|>
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return text.strip()


# ── Prompt builder ────────────────────────────────────────────
BUCKET_CONTEXT = {
    "Objective":   "the research problem, motivation, and goals of the paper",
    "Methodology": "the methods, system design, and technical approach used",
    "Results":     "the experimental results, metrics, and performance findings",
    "Conclusion":  "the conclusions, limitations, and future work",
}

def _build_prompt(bucket_name: str, sentences: str) -> str:
    context = BUCKET_CONTEXT.get(bucket_name, "the content of this section")
    # Strip HTML tags and control tokens from input sentences before building prompt
    sentences = _clean_text(sentences)
    return (
        f"<|user|>\n"
        f"You are a strict technical summarizer for a research paper section about {context}.\n\n"
        f"RULES:\n"
        f"1. Only use the sentences provided below. Do NOT add anything new.\n"
        f"2. Do NOT change any numbers, percentages, model names, or metric values. Copy them exactly as they appear.\n"
        f"3. Cover all important facts — do not skip any metrics, accuracy values, method names, or dataset details.\n"
        f"4. Each point must be unique — do not repeat the same fact in different words.\n"
        f"5. Write in simple, clear English. One fact per point.\n"
        f"6. Generate between 4 to 7 numbered points. Condense the content to fit this range without losing any key facts.\n"
        f"7. Output numbered points only. No intro, no explanation, nothing else.\n\n"
        f"Sentences:\n{sentences}\n\n"
        f"Numbered summary:\n"
        f"<|end|>\n"
        f"<|assistant|>\n"
    )


# ── Output parser ─────────────────────────────────────────────
def _parse_numbered_points(raw_output: str, prompt: str) -> list:
    """
    Extract numbered points from model output.
    Strips prompt echo, control tokens, HTML, and double-numbering.
    Returns a clean list of strings.
    """
    # Strip the prompt from the output (model echoes it back)
    response = raw_output
    if "<|assistant|>" in raw_output:
        response = raw_output.split("<|assistant|>")[-1]
    elif prompt in raw_output:
        response = raw_output[len(prompt):]

    # Clean control tokens and HTML from the response
    response = _clean_text(response)
    response = response.strip()

    # Match lines starting with "1." "2." "1)" "2)" etc.
    pattern = re.compile(r'^\s*(\d+)[.)]\s+(.+)', re.MULTILINE)
    matches = pattern.findall(response)

    if matches:
        points = [text.strip() for _, text in matches]
        # Final clean pass on each point — strip any leftover tags or tokens
        points = [_clean_text(p) for p in points]
        # Filter out empty or very short points
        points = [p for p in points if len(p.split()) > 4]
        return points

    # Fallback: split by newlines
    lines = [_clean_text(l.strip()) for l in response.split('\n') if len(l.strip().split()) > 4]
    return lines[:7]


# ── Per-bucket rewriter ───────────────────────────────────────
def rewrite_bucket(bucket_name: str, polished_text: str) -> list:
    """
    Takes the polished paragraph text for one bucket,
    rewrites it as numbered points using the LLM.
    Returns a clean list of point strings.
    """
    if not polished_text or polished_text == "Data not found for this section.":
        return []

    pipe = _get_pipeline()
    prompt = _build_prompt(bucket_name, polished_text)

    try:
        output = pipe(
            prompt,
            max_new_tokens=400,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            return_full_text=True,
        )
        raw_text = output[0]["generated_text"]
        points = _parse_numbered_points(raw_text, prompt)

        if not points:
            print(f"  [Rewriter] Warning: no points parsed for {bucket_name}, using fallback.")
            sentences = re.split(r'(?<=[.!?])\s+', polished_text)
            return [_clean_text(s.strip()) for s in sentences if len(s.strip().split()) > 4][:7]

        print(f"  [Rewriter] {bucket_name}: {len(points)} points generated.")
        return points

    except Exception as e:
        print(f"  [Rewriter] Error on {bucket_name}: {e}")
        sentences = re.split(r'(?<=[.!?])\s+', polished_text)
        return [_clean_text(s.strip()) for s in sentences if len(s.strip().split()) > 4][:7]


# ── Entry point ───────────────────────────────────────────────
def run_rewriting(input_json_path: str, output_json_path: str) -> bool:
    """
    Phase 4: LLM rewriting.
    Reads polished JSON, rewrites each bucket as numbered points,
    writes final JSON with both the rewritten points AND the original
    polished text (so the frontend can show a toggle).
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        final_output = {
            "Paper Title": data.get("Paper Title", "Unknown Title"),
            "Final_Summary": {}
        }

        polished = data.get("Polished_Summary", {})

        for bucket_name, content in polished.items():
            raw_text = content.get("text", "")
            images = content.get("images", [])

            if raw_text == "Data not found for this section." or not raw_text.strip():
                final_output["Final_Summary"][bucket_name] = {
                    "points": [],
                    "raw_text": raw_text,
                    "images": images
                }
                continue

            points = rewrite_bucket(bucket_name, raw_text)

            final_output["Final_Summary"][bucket_name] = {
                "points": points,
                "raw_text": raw_text,
                "images": images
            }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print("[Rewriter] All buckets rewritten successfully.")
        return True

    except Exception as e:
        print(f"[Rewriter] Critical Error: {e}")
        import traceback; traceback.print_exc()
        return False