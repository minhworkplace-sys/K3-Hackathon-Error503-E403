"""Dual storage bắt buộc: ChromaDB (semantic search) + page_index (exact lookup).

page_index format theo CLAUDE.md: {doc_id: {page_number: text}} — lưu full
text từng trang, phục vụ exact lookup theo trang và map-reduce summarization
(M3/M4), tách biệt khỏi việc chunk cho vector search.
"""

import json

import chromadb

from app.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, PAGE_INDEX_DIR
from app.ingestion.chunker import Chunk
from app.ingestion.embeddings import OfflineHashingEmbeddingFunction
from app.ingestion.pdf_parser import ParsedPage

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_embedding_function = OfflineHashingEmbeddingFunction()


def get_collection():
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_embedding_function,
    )


def index_chunks(doc_id: str, filename: str, chunks: list[Chunk]) -> int:
    """Index chunks vào ChromaDB, metadata mang page_number gốc cho citation."""
    if not chunks:
        return 0
    collection = get_collection()
    ids = [f"{doc_id}_p{c.page_number}_c{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def write_page_index(doc_id: str, pages: list[ParsedPage]) -> None:
    """Ghi page_index dạng {page_number: text} ra JSON riêng cho từng doc_id."""
    page_map = {str(p.page_number): p.text for p in pages}
    path = PAGE_INDEX_DIR / f"{doc_id}.json"
    path.write_text(json.dumps(page_map, ensure_ascii=False, indent=2), encoding="utf-8")


def read_page_index(doc_id: str) -> dict[str, str]:
    path = PAGE_INDEX_DIR / f"{doc_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
