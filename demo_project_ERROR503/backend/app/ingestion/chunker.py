"""Chunk theo đơn vị slide/trang tự nhiên.

Mỗi trang PDF = 1 chunk. Chỉ sub-chunk khi 1 trang vượt MAX_TOKENS_PER_CHUNK
token, nhưng mọi sub-chunk vẫn giữ đúng page_number của trang gốc (không suy
diễn page từ vị trí chunk).
"""

from dataclasses import dataclass

from app.config import MAX_TOKENS_PER_CHUNK
from app.ingestion.pdf_parser import ParsedPage


@dataclass
class Chunk:
    page_number: int
    chunk_index: int  # thứ tự sub-chunk trong cùng 1 trang, 0 nếu trang không bị tách
    text: str


def _count_tokens(text: str) -> int:
    # Xấp xỉ token bằng word-count (đủ dùng để quyết định ngưỡng sub-chunk,
    # không cần thêm dependency tokenizer cho demo).
    return len(text.split())


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + max_tokens])
        for i in range(0, len(words), max_tokens)
    ]


def chunk_by_slide(
    pages: list[ParsedPage], max_tokens: int = MAX_TOKENS_PER_CHUNK
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text:
            continue
        if _count_tokens(page.text) <= max_tokens:
            chunks.append(Chunk(page_number=page.page_number, chunk_index=0, text=page.text))
        else:
            for idx, part in enumerate(_split_by_tokens(page.text, max_tokens)):
                chunks.append(Chunk(page_number=page.page_number, chunk_index=idx, text=part))
    return chunks
