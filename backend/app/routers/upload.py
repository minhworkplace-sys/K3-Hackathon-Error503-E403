import logging
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import UPLOADS_DIR
from app.ingestion.chunker import chunk_by_slide
from app.ingestion.indexer import index_chunks, write_page_index
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.vision import describe_pages_with_images
from app.llm_provider import LLMProviderError, get_provider
from app.models.schemas import UploadResponse

logger = logging.getLogger(__name__)

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

    # M10: augment page co bieu do/hinh anh dang ke bang Gemini Vision, CHI
    # 1 lan luc ingest (khong goi lai khi user hoi lai cung trang - ket qua
    # da nam san trong page_index/chunk duoi day). Loi Vision (thieu key,
    # quota...) khong duoc lam hong upload - fallback ve pages goc, chi mat
    # phan augment anh, van con day du text.
    n_vision_pages = 0
    try:
        provider = get_provider()
        pages, n_vision_pages = describe_pages_with_images(str(dest_path), pages, provider)
    except LLMProviderError as exc:
        logger.warning("Bỏ qua augmentation Vision cho doc_id=%s: %s", doc_id, exc)

    chunks = chunk_by_slide(pages)

    write_page_index(doc_id, pages)
    num_indexed = index_chunks(doc_id, file.filename, chunks)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_pages=len(pages),
        num_chunks=num_indexed,
        num_vision_pages=n_vision_pages,
    )
