import os
import uuid


def save_uploaded_file(file_content: bytes, original_filename: str, upload_dir: str = "shared_data/input_pdfs"):
    """
    Validates and saves an uploaded PDF file.
    Returns (file_path, uuid_prefix) on success, or (None, None) on failure.

    Changes from v2:
    - Added magic byte check: verifies file is actually a PDF, not just named .pdf
    - Now returns the uuid_prefix so app.py can use it for all downstream filenames,
      preventing race conditions when two users upload files with the same name.
    """

    # 1. Check file extension
    if not original_filename.lower().endswith('.pdf'):
        print(f"[Uploader] Error: '{original_filename}' is not a PDF file.")
        return None, None

    # 2. Check magic bytes — real PDF files always start with %PDF
    #    This catches renamed files (e.g. virus.exe renamed to paper.pdf)
    if not file_content[:4] == b'%PDF':
        print(f"[Uploader] Error: '{original_filename}' failed magic byte check. Not a real PDF.")
        return None, None

    # 3. Create upload directory if needed
    os.makedirs(upload_dir, exist_ok=True)

    # 4. Generate UUID prefix — returned to caller so ALL downstream files share it
    uuid_prefix = uuid.uuid4().hex[:8]
    clean_name = original_filename.replace(" ", "_")
    unique_filename = f"{uuid_prefix}_{clean_name}"
    file_path = os.path.join(upload_dir, unique_filename)

    # 5. Save to disk
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        print(f"[Uploader] Saved '{original_filename}' → '{file_path}'")
        return file_path, uuid_prefix

    except Exception as e:
        print(f"[Uploader] Critical error saving file: {e}")
        return None, None
