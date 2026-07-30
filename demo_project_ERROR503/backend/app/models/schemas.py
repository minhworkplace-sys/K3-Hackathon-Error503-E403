from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int


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


class ChatResponse(BaseModel):
    mode: str  # "locate_extract" | "summarize_selection" | "summarize_progress" | "clarification"
    needs_clarification: bool = False
    answer: str
    citation_page: int | None = None
