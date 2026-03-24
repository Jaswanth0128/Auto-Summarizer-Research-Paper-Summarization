import pymupdf4llm
import pymupdf
import re
import os
import json
from collections import Counter


_MD_IMAGE_TAG_PATTERN = re.compile(r'!\[\]\((.*?)\)')
_PAGE_NO_PATTERN = re.compile(r'-(\d{4})-\d{2}\.png$', re.IGNORECASE)


def _parse_page_number_from_name(filename: str):
    match = _PAGE_NO_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1))


def _remove_markdown_image_tags(md_text: str, removed_paths: set[str]) -> str:
    def _replace(match):
        path = match.group(1)
        return "" if path in removed_paths else match.group(0)

    cleaned = _MD_IMAGE_TAG_PATTERN.sub(_replace, md_text)
    # Compact excessive blank lines after tag removal.
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned


def _filter_decorative_images(md_text: str, image_output_dir: str, pdf_path: str, dpi: int = 300) -> str:
    """
    Remove small / line-like decorative assets (commonly header/footer separators)
    while keeping substantive figures and explicit table crops.
    """
    tags = list(_MD_IMAGE_TAG_PATTERN.finditer(md_text))
    if not tags:
        return md_text

    page_count = 0
    try:
        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)
    except Exception:
        page_count = 0

    image_info = []
    for m in tags:
        rel_path = m.group(1)
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(os.getcwd(), rel_path)
        basename = os.path.basename(rel_path)

        # Keep explicit table screenshots generated in Part A.
        if basename.lower().startswith("table_page"):
            image_info.append({"rel_path": rel_path, "keep": True})
            continue

        if not os.path.exists(abs_path):
            image_info.append({"rel_path": rel_path, "keep": True})
            continue

        try:
            pix = pymupdf.Pixmap(abs_path)
            width = int(pix.width)
            height = int(pix.height)
            del pix
        except Exception:
            image_info.append({"rel_path": rel_path, "keep": True})
            continue

        min_edge = min(width, height)
        max_edge = max(width, height)
        area = width * height
        aspect = (max_edge / min_edge) if min_edge > 0 else float("inf")
        page_no = _parse_page_number_from_name(basename)

        image_info.append({
            "rel_path": rel_path,
            "abs_path": abs_path,
            "page_no": page_no,
            "width": width,
            "height": height,
            "area": area,
            "aspect": aspect,
            "keep": True,
        })

    # Repeated tiny line-like assets are usually decorative separators.
    dims_counter = Counter(
        (item.get("width"), item.get("height"))
        for item in image_info
        if "width" in item
    )
    repeated_threshold = max(3, page_count // 3) if page_count else 3

    removed_paths = set()
    for item in image_info:
        if "width" not in item:
            continue

        width = item["width"]
        height = item["height"]
        area = item["area"]
        aspect = item["aspect"]
        dims = (width, height)

        too_small = width < 220 or height < 220 or area < 90000
        line_like = (aspect >= 14.0 and min(width, height) <= 60)
        repeated_decorative = dims_counter[dims] >= repeated_threshold and (line_like or too_small)

        if line_like or repeated_decorative:
            item["keep"] = False
            removed_paths.add(item["rel_path"])
            try:
                os.remove(item["abs_path"])
            except Exception:
                pass

    if removed_paths:
        print(f"[Extractor] Filtered decorative images: {len(removed_paths)} removed.")
        return _remove_markdown_image_tags(md_text, removed_paths)
    return md_text

def run_extraction(pdf_path: str, output_json_path: str, image_output_dir: str) -> bool:
    """
    Extracts text, Markdown tables, and takes high-res screenshots of 
    visual tables and charts, packaging everything into a JSON dictionary.
    """
    if not os.path.exists(pdf_path):
        print(f"[Extractor] Error: Could not find '{pdf_path}'.")
        return False

    os.makedirs(image_output_dir, exist_ok=True)
    print(f"[Extractor] Processing: '{pdf_path}'...")
    
    # ---------------------------------------------------------
    # PART A: Extract Tables as Crisp PNG Images
    # ---------------------------------------------------------
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables.tables):
                bbox = table.bbox 
                # Take a high-res screenshot of just the table's bounding box
                pix = page.get_pixmap(clip=bbox, dpi=300)
                
                table_filename = f"table_page{page_num + 1}_{table_index + 1}.png"
                table_filepath = os.path.join(image_output_dir, table_filename)
                pix.save(table_filepath)
                
        print(f"[Extractor] Visual table extraction complete.")
    except Exception as e:
        print(f"[Extractor] Warning: Table screenshot extraction encountered an issue: {e}")

    # ---------------------------------------------------------
    # PART B: Extract Text, Charts, and Markdown structure
    # ---------------------------------------------------------
    try:
        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            write_images=True,
            image_path=image_output_dir,
            image_format="png",
            dpi=300,
            header=False,
            footer=False
        )
        md_text = _filter_decorative_images(
            md_text,
            image_output_dir=image_output_dir,
            pdf_path=pdf_path,
            dpi=300,
        )
    except Exception as e:
        print(f"[Extractor] Error during Markdown extraction: {e}")
        return False
    
    # ---------------------------------------------------------
    # PART C: Parse Markdown into a Structured Dictionary
    # ---------------------------------------------------------
    header_pattern = r'(?m)^(#+\s+.*|[\*\_]+[\d\.]+(?:[\*\_]|\s)+[^\n]*|[\*\_]+(?:Acknowledgments|References)[\*\_]*[^\n]*|[IVXLC]+\.?\s+[A-Z][A-Z ]+)'
    parts = re.split(header_pattern, md_text, flags=re.IGNORECASE)
    paper_dict = {}
    
    if len(parts) > 1:
        # Map the Paper Title
        raw_title = parts[1]
        title_content = parts[2].strip() if 2 < len(parts) else ""
        clean_title = re.sub(r'^#+\s+', '', raw_title).replace('**', '').replace('_', '').strip()
        paper_dict["Paper Title"] = clean_title
        
        # Map the Front Matter (Authors, Abstract, etc.)
        if title_content:
            paper_dict["Front Matter"] = title_content
            
        # Loop through remaining sections
        for i in range(3, len(parts), 2):
            raw_header = parts[i]
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            
            clean_header = re.sub(r'^[IVXLC]+\.?\s+', '', raw_header)
            clean_header = re.sub(r'^#+\s+', '', clean_header).replace('**', '').replace('_', '').strip()
            
            if clean_header in paper_dict:
                paper_dict[clean_header] += "\n\n" + content
            else:
                paper_dict[clean_header] = content
                
    # ---------------------------------------------------------
    # PART D: Export to JSON
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    try:
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(paper_dict, json_file, ensure_ascii=False, indent=4)
        print(f"[Extractor] Success! Data structured and saved to '{output_json_path}'.")
        return True
    except Exception as e:
        print(f"[Extractor] Failed to write JSON: {e}")
        return False