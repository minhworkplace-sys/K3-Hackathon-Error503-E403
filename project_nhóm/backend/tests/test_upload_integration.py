from fastapi.testclient import TestClient

from app.ingestion.indexer import get_collection, read_page_index
from app.main import app
from tests.pdf_helpers import make_test_pdf, page_marker

client = TestClient(app)


def test_upload_end_to_end_page_numbers_not_shifted(tmp_path):
    pdf_path = tmp_path / "lecture.pdf"
    make_test_pdf(str(pdf_path), num_pages=4)

    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": ("lecture.pdf", f, "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["num_pages"] == 4
    assert body["num_chunks"] == 4  # slide ngan -> 1 chunk / trang
    doc_id = body["doc_id"]

    # 1. page_index: text moi trang phai dung marker tuong ung, khong lech
    page_index = read_page_index(doc_id)
    assert set(page_index.keys()) == {"1", "2", "3", "4"}
    for n in range(1, 5):
        assert page_marker(n) in page_index[str(n)]

    # 2. ChromaDB: metadata page_number cua tung chunk phai khop voi noi dung
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    assert len(result["ids"]) == 4
    for doc_text, metadata in zip(result["documents"], result["metadatas"]):
        expected_marker = page_marker(metadata["page_number"])
        assert expected_marker in doc_text


def test_upload_rejects_non_pdf():
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
