from fastapi import APIRouter, HTTPException

from app.ingestion.indexer import read_page_index
from app.models.schemas import PagesResponse, PageText

router = APIRouter()


@router.get("/doc/{doc_id}/pages", response_model=PagesResponse)
def get_pages(doc_id: str) -> PagesResponse:
    page_index = read_page_index(doc_id)
    if not page_index:
        raise HTTPException(status_code=404, detail="doc_id không tồn tại, hãy upload trước")

    pages = [
        PageText(page_number=int(k), text=v)
        for k, v in sorted(page_index.items(), key=lambda kv: int(kv[0]))
    ]
    return PagesResponse(doc_id=doc_id, pages=pages)
