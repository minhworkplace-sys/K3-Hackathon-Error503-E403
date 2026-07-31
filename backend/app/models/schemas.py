from typing import Literal

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    num_vision_pages: int = 0  # M10: so trang duoc goi Gemini Vision augment


class PageText(BaseModel):
    page_number: int
    text: str


class PagesResponse(BaseModel):
    doc_id: str
    pages: list[PageText]


class ChatRequest(BaseModel):
    doc_id: str
    message: str = ""
    selected_text: str | None = None
    page_hint: int | None = None
    up_to_page: int | None = None


class KeywordChip(BaseModel):
    text: str
    category: Literal["problem", "solution", "metric", "definition", "none"] = "none"
    connector: bool = False  # True = nối bằng mũi tên "->" với chip ngay truoc no


class RelatedPage(BaseModel):
    page: int
    similarity: float  # [0,1], tu semantic embedding that - khong qua LLM generate


class ChatResponse(BaseModel):
    mode: str  # "locate_extract" | "summarize_selection" | "summarize_progress" | "range" | "related" | "clarification"
    needs_clarification: bool = False
    answer: str  # ban flatten (text-only) cua keywords, hoac cau van cho clarification/error
    keywords: list[KeywordChip] | None = None  # co gia tri cho locate_extract/summarize_*/range
    related_pages: list[RelatedPage] | None = None  # co gia tri cho mode="related"
    citation_page: int | None = None


class DetailRequest(BaseModel):
    doc_id: str
    page: int


class DetailResponse(BaseModel):
    page: int
    detail: str
