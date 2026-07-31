"""M3: script migration 1 lần — xoá ChromaDB collection cũ (đang bind với
`offline_hashing`, 256-dim) và re-index lại toàn bộ doc đã upload bằng
embedding thật (`gemini_embedding_001`). Bắt buộc chạy với LLM_MODE=gemini
(báo lỗi rõ nếu không).

Đọc text từ `page_index/{doc_id}.json` đã có sẵn trên đĩa (không re-parse
PDF) — để KHÔNG mất phần augment ảnh/biểu đồ (M10) đã lưu sẵn trong đó cho
các doc đã test trước. filename gốc không được lưu riêng nên dùng lại
"{doc_id}.pdf" cho metadata (chỉ ảnh hưởng hiển thị, không ảnh hưởng
retrieval/citation).

Chạy: cd backend && LLM_MODE=gemini .venv/Scripts/python.exe scripts/reindex_embeddings.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHROMA_COLLECTION_NAME, PAGE_INDEX_DIR  # noqa: E402
from app.ingestion.chunker import chunk_by_slide  # noqa: E402
from app.ingestion.indexer import _client, index_chunks  # noqa: E402
from app.ingestion.pdf_parser import ParsedPage  # noqa: E402


def main() -> None:
    mode = os.environ.get("LLM_MODE", "mock").strip().lower()
    if mode != "gemini":
        print(f"LLM_MODE hiện là '{mode}', cần 'gemini' để reindex bằng embedding thật. Dừng.")
        sys.exit(1)

    # Xoá qua API ChromaDB (không rmtree cả thư mục) — tránh lỗi file-lock
    # trên Windows (sqlite file có thể bị process khác giữ handle).
    try:
        _client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Đã xoá collection cũ: {CHROMA_COLLECTION_NAME}")
    except Exception as exc:  # noqa: BLE001 - chua co collection cung khong sao
        print(f"(Không xoá được collection cũ, có thể chưa tồn tại: {exc})")

    doc_files = sorted(PAGE_INDEX_DIR.glob("*.json"))
    # Chỉ reindex doc_id truyền qua argv nếu có — mặc định reindex-all rất
    # tốn quota Embedding API khi đã có nhiều doc_id trùng nội dung từ các
    # lần test trước (26 doc_id trên đĩa, phần lớn là cùng 1 file test).
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        doc_files = [p for p in doc_files if p.stem in wanted]
    if not doc_files:
        print("Không có doc_id nào để reindex (kiểm tra lại tham số truyền vào).")
        return

    total_chunks = 0
    for path in doc_files:
        doc_id = path.stem
        page_map: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        pages = [
            ParsedPage(page_number=int(p), text=text)
            for p, text in sorted(page_map.items(), key=lambda kv: int(kv[0]))
        ]
        chunks = chunk_by_slide(pages)
        n = index_chunks(doc_id, f"{doc_id}.pdf", chunks)
        total_chunks += n
        print(f"  {doc_id}: {len(pages)} trang → {n} chunk đã index (embedding thật)")

    print(f"Xong. {len(doc_files)} doc, tổng {total_chunks} chunk.")


if __name__ == "__main__":
    main()
