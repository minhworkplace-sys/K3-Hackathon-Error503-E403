"""M2: /chat — summarize_selection gọi qua LLMProvider (Gemini thật hoặc
Mock tuỳ LLM_MODE). locate_extract vẫn dùng keyword search thật trên
page_index (chưa phải hybrid BM25/semantic — đó là M3); summarize_progress
vẫn là placeholder đếm trang (map-reduce thật là M5).
"""

import re

from fastapi import APIRouter, HTTPException

from app.eval_logger import log_interaction
from app.ingestion.indexer import read_page_index
from app.llm_provider import LLMProvider, LLMProviderError, get_provider
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()

_provider: LLMProvider | None = None


def _get_provider() -> LLMProvider:
    """Khởi tạo provider lười (lazy) và cache lại — tránh crash lúc app
    start nếu LLM_MODE=gemini nhưng thiếu key, và tránh tạo lại genai.Client
    mỗi request."""
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


_AMBIGUOUS_MARKERS = ["đoạn này", "cái này", "chỗ này", "phần này", "giải thích"]

# Các từ chức năng tiếng Việt xuất hiện dày đặc trên mọi trang, không mang
# nhiều tín hiệu chủ đề — loại khỏi keyword search để tránh match sai trang
# (vd. "dùng"/"làm" xuất hiện ở page khác nhiều hơn từ khoá thật sự liên quan).
_STOPWORDS = {
    "va", "la", "co", "cua", "cho", "theo", "dung", "lam", "de", "cac",
    "mot", "nhung", "nay", "do", "khi", "thi", "ma", "hay", "hoac",
    "khong", "duoc", "voi", "trong", "ngoai", "tren", "duoi", "tai",
    "den", "tu", "ra", "vao", "len", "xuong", "di", "ve", "he", "so",
    "và", "là", "có", "của", "cho", "theo", "dùng", "làm", "để", "các",
    "một", "những", "này", "đó", "khi", "thì", "mà", "hay", "hoặc",
    "không", "được", "với", "trong", "ngoài", "trên", "dưới", "tại",
    "đến", "từ", "ra", "vào", "lên", "xuống", "đi", "về", "hệ", "sao",
}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    page_index = read_page_index(req.doc_id)
    if not page_index:
        raise HTTPException(status_code=404, detail="doc_id không tồn tại, hãy upload trước")

    response = _route(req, page_index)
    log_interaction(req, response)
    return response


def _route(req: ChatRequest, page_index: dict[str, str]) -> ChatResponse:
    if req.selected_text and req.selected_text.strip():
        return _summarize_selection(req.selected_text.strip())

    if req.up_to_page:
        return _summarize_progress(req.up_to_page, page_index)

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message rỗng")

    if not req.page_hint and _looks_ambiguous(message):
        return ChatResponse(
            mode="clarification",
            needs_clarification=True,
            answer=(
                "Câu hỏi của bạn hơi mơ hồ — bạn có thể bôi đen đoạn cụ thể trên "
                "slide, hoặc cho biết trang/số slide bạn đang hỏi không?"
            ),
        )

    return _locate_extract(message, req.page_hint, page_index)


def _summarize_selection(selected_text: str) -> ChatResponse:
    try:
        provider = _get_provider()
        answer = "".join(provider.summarize_stream(selected_text))
    except LLMProviderError as exc:
        return ChatResponse(
            mode="summarize_selection",
            answer=f"Không thể tóm tắt lúc này ({exc}). Vui lòng thử lại sau.",
        )
    return ChatResponse(mode="summarize_selection", answer=answer)


def _summarize_progress(up_to_page: int, page_index: dict[str, str]) -> ChatResponse:
    """Tóm tắt đơn giản (1 call Gemini trên toàn bộ text gộp từ trang 1..N).

    Chưa phải map-reduce từng trang + cache theo (doc_id, up_to_page) như
    CLAUDE.md mô tả cho bản MVP cuối cùng (việc đó thuộc M5) — đây là bản
    rút gọn để demo lệnh "tóm tắt tài liệu" chạy được với Gemini thật.
    """
    n = min(up_to_page, len(page_index))
    pages_text = "\n\n".join(
        page_index[str(p)] for p in range(1, n + 1) if page_index.get(str(p))
    )
    if not pages_text.strip():
        return ChatResponse(mode="summarize_progress", answer="Chưa có nội dung để tóm tắt.")

    try:
        provider = _get_provider()
        answer = "".join(provider.summarize_stream(pages_text))
    except LLMProviderError as exc:
        return ChatResponse(
            mode="summarize_progress",
            answer=f"Không thể tóm tắt lúc này ({exc}). Vui lòng thử lại sau.",
        )
    return ChatResponse(mode="summarize_progress", answer=answer)


def _looks_ambiguous(message: str) -> bool:
    lowered = message.lower()
    return len(message) < 40 and any(marker in lowered for marker in _AMBIGUOUS_MARKERS)


def _locate_extract(message: str, page_hint: int | None, page_index: dict[str, str]) -> ChatResponse:
    if page_hint and str(page_hint) in page_index:
        page = page_hint
    else:
        page = _keyword_search(message, page_index)

    if page is None:
        return ChatResponse(
            mode="locate_extract",
            answer="[Demo] Không tìm thấy nội dung khớp với câu hỏi trong tài liệu.",
        )

    return ChatResponse(
        mode="locate_extract",
        citation_page=page,
        answer=f"[Demo] Nội dung liên quan tới câu hỏi của bạn nằm ở trang {page}.",
    )


def _keyword_search(message: str, page_index: dict[str, str]) -> int | None:
    words = [
        w.lower()
        for w in re.findall(r"\w+", message)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]
    if not words:
        return None
    best_page, best_score = None, 0
    for page_str, text in page_index.items():
        lowered = text.lower()
        score = sum(lowered.count(w) for w in words)
        if score > best_score:
            best_score, best_page = score, int(page_str)
    return best_page
