"""Ghi log mọi tương tác /chat vào eval/qa_log.jsonl (thư mục eval/ ở root
project, không phải trong backend/) — dữ liệu thô để dùng cho script đo
accuracy locate_extract ở M7. Lỗi ghi log không được làm hỏng response
chính của /chat.

M9: thêm field `response.keywords` (chỉ thêm, không đổi field cũ nào) — log
cũ (trước M9) không có field này, vẫn đọc được bình thường vì mỗi dòng JSONL
độc lập, code đọc log cũ không bắt buộc phải có field mới.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.config import BACKEND_DIR
from app.models.schemas import ChatRequest, ChatResponse

EVAL_DIR = BACKEND_DIR.parent / "eval"
EVAL_LOG_PATH = EVAL_DIR / "qa_log.jsonl"

EVAL_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(req: ChatRequest, res: ChatResponse) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_id": req.doc_id,
        "request": {
            "message": req.message or None,
            "selected_text": req.selected_text,
            "page_hint": req.page_hint,
            "up_to_page": req.up_to_page,
        },
        "response": {
            "mode": res.mode,
            "needs_clarification": res.needs_clarification,
            "answer": res.answer,
            "keywords": [kw.model_dump() for kw in res.keywords] if res.keywords else None,
            "citation_page": res.citation_page,
        },
    }
    try:
        with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[eval_logger] Không ghi được log: {exc}")
