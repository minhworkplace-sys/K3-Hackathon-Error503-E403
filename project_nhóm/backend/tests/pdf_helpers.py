"""Helper tạo PDF giả lập có marker biết trước theo từng trang, dùng để verify
page number không bị lệch qua toàn bộ pipeline (parse -> chunk -> index)."""

import fitz


def page_marker(n: int) -> str:
    return f"PAGE_MARKER_{n}"


def make_test_pdf(path: str, num_pages: int = 4) -> None:
    doc = fitz.open()
    rect = fitz.Rect(50, 50, 550, 750)
    for i in range(num_pages):
        page = doc.new_page()
        n = i + 1
        text = f"{page_marker(n)}\nNoi dung trang so {n}."
        page.insert_textbox(rect, text, fontsize=14)
    doc.save(path)
    doc.close()


def make_long_page_text(word_count: int = 600) -> str:
    return " ".join(f"word{i}" for i in range(word_count))
