"""M3: hybrid retrieval (BM25 + semantic) — CHỈ dùng khi câu hỏi không có
page_hint rõ ràng. Thứ tự ưu tiên vẫn đúng CLAUDE.md: page_hint exact lookup
từ page_index TRƯỚC (xem chat.py._locate_extract), hybrid search ở đây chỉ
là fallback khi không có/miss page_hint — KHÔNG được đảo thứ tự này.

Confidence score: weighted-fusion, final_score = 0.5*bm25_component +
0.5*semantic_component (giữ đúng trọng số 0.5/0.5 đã chốt).

QUAN TRỌNG — đã SỬA cách tính từng component sau khi test thật ở M11 phát
hiện bug: bản đầu dùng min-max normalize BM25/semantic TRONG tập kết quả
của từng câu hỏi. Bug: với câu hỏi HOÀN TOÀN ngoài phạm vi tài liệu (VD
"Cách nấu phở bò"), min-max vẫn kéo trang tốt nhất TƯƠNG ĐỐI lên tới 1.0
dù tuyệt đối nó chẳng liên quan gì — đo thật cho confidence NGOÀI-phạm-vi
tới 1.000, cao hơn cả một số câu ĐÚNG phạm vi (0.71) → guardrail sẽ vô
dụng. Sửa: bỏ min-max, dùng scale TUYỆT ĐỐI cho từng bên:
- BM25: saturating transform `score / (score + _BM25_SATURATION_K)` — giữ
  được thông tin "điểm tuyệt đối cao hay thấp" thay vì chỉ xếp hạng tương
  đối (0 điểm luôn → 0, không bị kéo lên).
- Semantic: dùng THẲNG cosine similarity (đã tự nhiên nằm trong khoảng có
  ý nghĩa tuyệt đối, không cần min-max nữa).
Đã đo thật trên 5 câu ĐÚNG (fused 0.657-0.716) vs 5 câu SAI/ngoài phạm vi
(fused 0.296-0.520) — có khoảng trống rõ ràng, chọn threshold ở giữa. Xem
PROGRESS.md mục M11 để biết số liệu đầy đủ.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from rank_bm25 import BM25Okapi

from app.ingestion.embeddings import embed_query, embed_similarity
from app.ingestion.indexer import get_collection
from app.llm_provider import LLMProviderError

_RELATED_TOP_K_RAW = 15  # lay du chunk truoc khi loai trang neo + dedupe theo trang
_RELATED_MAX_PAGES = 5  # gioi han CUNG - khong duoc vuot (theo yeu cau M11)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Từ chức năng tiếng Việt xuất hiện dày đặc trên mọi trang, không mang nhiều
# tín hiệu chủ đề — loại khỏi BM25 tokenize để tránh match sai trang.
STOPWORDS = {
    "va", "la", "co", "cua", "cho", "theo", "dung", "lam", "de", "cac",
    "mot", "nhung", "nay", "do", "khi", "thi", "ma", "hay", "hoac",
    "khong", "duoc", "voi", "trong", "ngoai", "tren", "duoi", "tai",
    "den", "tu", "ra", "vao", "len", "xuong", "di", "ve", "he", "so",
    "và", "là", "có", "của", "cho", "theo", "dùng", "làm", "để", "các",
    "một", "những", "này", "đó", "khi", "thì", "mà", "hay", "hoặc",
    "không", "được", "với", "trong", "ngoài", "trên", "dưới", "tại",
    "đến", "từ", "ra", "vào", "lên", "xuống", "đi", "về", "hệ", "sao",
}

_SEMANTIC_TOP_K = 10

# "Nửa bão hoà" cho saturating transform BM25 (score=K -> component=0.5).
# Chọn dựa trên phân bố thật đo được: câu ĐÚNG phạm vi có top BM25 raw
# 5.3-10.1, câu SAI/ngoài phạm vi có top BM25 raw 0-3.7 (nhiều câu = 0 vì
# không trùng từ khoá nào cả) — K=5 nằm giữa 2 nhóm, đủ steep để phân biệt.
_BM25_SATURATION_K = 5.0


def _tokenize(text: str) -> list[str]:
    return [
        w for w in _TOKEN_RE.findall(text.lower())
        if len(w) > 2 and w not in STOPWORDS
    ]


def _bm25_scores(message: str, page_index: dict[str, str]) -> dict[int, float]:
    pages = sorted(page_index.items(), key=lambda kv: int(kv[0]))
    corpus = [_tokenize(text) for _, text in pages]
    query_tokens = _tokenize(message)
    if not corpus or not query_tokens:
        return {}
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    return {int(page_str): float(score) for (page_str, _), score in zip(pages, scores)}


def _semantic_scores(message: str, doc_id: str) -> dict[int, float]:
    collection = get_collection()
    query_embedding = embed_query(message)
    result = collection.query(
        query_embeddings=[query_embedding],
        where={"doc_id": doc_id},
        n_results=_SEMANTIC_TOP_K,
    )
    scores: dict[int, float] = {}
    ids = result.get("ids") or [[]]
    distances = result.get("distances") or [[]]
    metadatas = result.get("metadatas") or [[]]
    for _id, dist, meta in zip(ids[0], distances[0], metadatas[0]):
        page = meta["page_number"]
        # ChromaDB cosine space: distance = 1 - cosine_similarity
        similarity = 1.0 - dist
        # 1 trang có thể có nhiều chunk (trang dài, sub-chunk >500 token) —
        # lấy similarity cao nhất trong các chunk cùng trang.
        if page not in scores or similarity > scores[page]:
            scores[page] = similarity
    return scores


def _bm25_saturate(score: float) -> float:
    """score=0 -> 0; score=K -> 0.5; score→∞ -> tiệm cận 1. Giữ thông tin
    tuyệt đối (khác min-max — xem docstring đầu file lý do đổi)."""
    return score / (score + _BM25_SATURATION_K) if score > 0 else 0.0


class SearchResult(NamedTuple):
    page: int | None
    confidence: float  # fused score cua trang tot nhat, [0,1]
    runner_up_page: int | None
    margin: float  # confidence - runner_up_confidence; 1.0 neu khong co runner-up (khong mo ho)


def hybrid_search(message: str, doc_id: str, page_index: dict[str, str]) -> SearchResult:
    """M11: trả về SearchResult (page, confidence, runner_up_page, margin) —
    margin dùng để phát hiện tình huống MƠ HỒ (2 trang cùng liên quan, VD
    trang 28/29 "divider" vs "chi tiết" cùng chủ đề — xem chat.py._locate_extract
    và PROGRESS.md mục M11 phần disambiguation)."""
    bm25_raw = _bm25_scores(message, page_index)
    try:
        semantic_raw = _semantic_scores(message, doc_id)
    except LLMProviderError:
        # Lỗi Gemini Embedding API (quota, timeout...) — fallback về BM25
        # thuần thay vì crash cả request; vẫn tôn trọng nguyên tắc CLAUDE.md
        # "không để lỗi 500 trần trụi lên frontend".
        semantic_raw = {}

    all_pages = set(bm25_raw) | set(semantic_raw)
    if not all_pages:
        return SearchResult(None, 0.0, None, 1.0)

    fused = {
        p: 0.5 * _bm25_saturate(bm25_raw.get(p, 0.0)) + 0.5 * semantic_raw.get(p, 0.0)
        for p in all_pages
    }
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    best_page, best_score = ranked[0]
    if len(ranked) == 1:
        return SearchResult(best_page, best_score, None, 1.0)
    runner_up_page, runner_up_score = ranked[1]
    return SearchResult(best_page, best_score, runner_up_page, best_score - runner_up_score)


def find_related_pages(
    doc_id: str, anchor_page: int, page_index: dict[str, str]
) -> list[tuple[int, float]]:
    """M11 Phần A #4: tìm tối đa 5 trang liên quan nhất tới `anchor_page`,
    CHỈ trong cùng `doc_id` (where filter — không search xuyên tài liệu
    khác). Dùng embed_similarity (task_type=SEMANTIC_SIMILARITY, khác
    RETRIEVAL_QUERY của hybrid_search — đây là so sánh trang-với-trang
    ngang hàng, không phải query-tìm-document).

    Trả về list (page_number, similarity) đã sort giảm dần, tối đa
    _RELATED_MAX_PAGES phần tử. Không dùng LLM generate — thuần retrieval,
    không có rủi ro bịa nội dung.
    """
    anchor_text = page_index.get(str(anchor_page), "")
    if not anchor_text.strip():
        return []

    try:
        anchor_embedding = embed_similarity(anchor_text)
    except LLMProviderError:
        return []

    collection = get_collection()
    result = collection.query(
        query_embeddings=[anchor_embedding],
        where={"doc_id": doc_id},
        n_results=_RELATED_TOP_K_RAW,
    )
    distances = result.get("distances") or [[]]
    metadatas = result.get("metadatas") or [[]]

    scores: dict[int, float] = {}
    for dist, meta in zip(distances[0], metadatas[0]):
        page = meta["page_number"]
        if page == anchor_page:
            continue  # khong tinh chinh trang neo la "lien quan" toi chinh no
        similarity = 1.0 - dist
        if page not in scores or similarity > scores[page]:
            scores[page] = similarity

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:_RELATED_MAX_PAGES]
