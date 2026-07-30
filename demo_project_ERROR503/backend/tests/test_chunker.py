from app.ingestion.chunker import chunk_by_slide
from app.ingestion.pdf_parser import ParsedPage
from tests.pdf_helpers import make_long_page_text


def test_short_page_becomes_single_chunk_with_correct_page_number():
    pages = [ParsedPage(page_number=1, text="short slide content")]

    chunks = chunk_by_slide(pages, max_tokens=500)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_long_page_is_split_but_keeps_same_page_number():
    long_text = make_long_page_text(600)
    pages = [ParsedPage(page_number=1, text=long_text)]

    chunks = chunk_by_slide(pages, max_tokens=500)

    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_mixed_pages_do_not_leak_page_number_across_boundaries():
    # Trang 1 ngắn, trang 2 dài (bị tách), trang 3 ngắn — chunk của trang 3
    # không được vô tình mang page_number của trang 2 do lỗi suy diễn từ
    # vị trí/enumerate của danh sách chunk tổng.
    pages = [
        ParsedPage(page_number=1, text="slide mo dau"),
        ParsedPage(page_number=2, text=make_long_page_text(600)),
        ParsedPage(page_number=3, text="slide ket luan"),
    ]

    chunks = chunk_by_slide(pages, max_tokens=500)

    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    page3_chunks = [c for c in chunks if c.page_number == 3]

    assert len(page1_chunks) == 1
    assert len(page2_chunks) > 1
    assert len(page3_chunks) == 1
    assert "ket luan" in page3_chunks[0].text
