from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

STORAGE_DIR = BACKEND_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
CHROMA_DIR = STORAGE_DIR / "chroma_db"
PAGE_INDEX_DIR = STORAGE_DIR / "page_index"

for d in (UPLOADS_DIR, CHROMA_DIR, PAGE_INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHROMA_COLLECTION_NAME = "slides"

# Spec: chỉ sub-chunk khi 1 trang vượt ngưỡng token, giữ nguyên page metadata.
MAX_TOKENS_PER_CHUNK = 500
