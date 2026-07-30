"""Parse PDF, giữ đúng page number gốc từ file (ground-truth từ fitz.Page.number)."""

from dataclasses import dataclass

from pathlib import Path

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


def render_pdf_to_images(file_path: str, output_dir: Path) -> None:
    """Render từng trang PDF thành file ảnh PNG tương ứng (1 page = 1 slide image)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(file_path)
    try:
        for page in doc:
            page_number = page.number + 1
            pix = page.get_pixmap(dpi=150)
            image_path = output_dir / f"page_{page_number}.png"
            pix.save(str(image_path))
    finally:
        doc.close()
