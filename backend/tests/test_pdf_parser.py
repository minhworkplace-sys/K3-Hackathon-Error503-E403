from app.ingestion.pdf_parser import parse_pdf
from tests.pdf_helpers import make_test_pdf, page_marker


def test_page_number_matches_marker(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_test_pdf(str(pdf_path), num_pages=5)

    pages = parse_pdf(str(pdf_path))

    assert len(pages) == 5
    for n, page in enumerate(pages, start=1):
        assert page.page_number == n
        assert page_marker(n) in page.text
        # Marker của các trang khác không được lẫn vào trang này
        for other in range(1, 6):
            if other != n:
                assert page_marker(other) not in page.text


def test_page_numbers_are_sequential_from_one(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_test_pdf(str(pdf_path), num_pages=3)

    pages = parse_pdf(str(pdf_path))

    assert [p.page_number for p in pages] == [1, 2, 3]
