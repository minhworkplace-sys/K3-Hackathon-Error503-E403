import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import UPLOADS_DIR
from app.ingestion.chunker import chunk_by_slide
from app.ingestion.indexer import index_chunks, write_page_index
from app.ingestion.pdf_parser import parse_pdf
from app.models.schemas import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile) -> UploadResponse:
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File phải là PDF hợp lệ")

    doc_id = uuid.uuid4().hex
    dest_path = UPLOADS_DIR / f"{doc_id}.pdf"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File rỗng")
    dest_path.write_bytes(contents)

    try:
        pages = parse_pdf(str(dest_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không parse được PDF: {exc}") from exc

    if not pages:
        raise HTTPException(status_code=400, detail="PDF không có trang nào")

    chunks = chunk_by_slide(pages)

    write_page_index(doc_id, pages)
    num_indexed = index_chunks(doc_id, file.filename, chunks)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_pages=len(pages),
        num_chunks=num_indexed,
    )
