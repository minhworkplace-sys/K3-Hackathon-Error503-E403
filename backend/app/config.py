import os
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

# M3: cho phép override tên collection qua env var — để test suite dùng
# collection riêng ("slides_test", set trong tests/conftest.py), không đụng
# vào collection thật (đã bind embedding function thật, gemini_embedding_001,
# khác dimension/config với offline_hashing mock dùng trong test).
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME") or "slides"

# Spec: chỉ sub-chunk khi 1 trang vượt ngưỡng token, giữ nguyên page metadata.
MAX_TOKENS_PER_CHUNK = 500

# M10: ngưỡng lọc trang cần gọi Gemini Vision lúc ingest — xem
# app/ingestion/vision.py và PROGRESS.md (mục M10) để biết lý do chọn 2
# ngưỡng này (đã test thực tế trên PDF thật trong project, so sánh 4 cách
# lọc khác nhau trước khi chốt).
VISION_MIN_IMAGE_AREA_RATIO = 0.05  # ảnh nhúng chiếm >5% diện tích trang
# >15 (nghĩa là >=16) vector shape/trang — đúng bằng ngưỡng đã test & báo cáo
# ("n_drawings > 15"). Lưu ý: dùng 16 ở đây (so sánh strict >) sẽ LỌT mất
# trang có đúng 16 drawings — đã tự bắt lỗi off-by-one này lúc test thật.
VISION_MIN_DRAWINGS = 15
VISION_RENDER_DPI = 150
