"""Parse PDF, giữ đúng page number gốc từ file (ground-truth từ fitz.Page.number)."""

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed, lấy từ page.number + 1 (KHÔNG suy từ enumerate chunk)
    text: str


def parse_pdf(file_path: str) -> list[ParsedPage]:
    doc = fitz.open(file_path)
    try:
        pages = []
        for page in doc:
            page_number = page.number + 1
            text = page.get_text("text").strip()
            pages.append(ParsedPage(page_number=page_number, text=text))
        return pages
    finally:
        doc.close()
