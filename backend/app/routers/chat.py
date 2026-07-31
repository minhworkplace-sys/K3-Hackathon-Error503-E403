"""M9: /chat — locate_extract, summarize_selection, summarize_progress đều trả
lời dạng keyword-chip (3-6 cụm từ ngắn, có category để tô màu, connector để nối
mũi tên khi có quan hệ nhân quả) thay cho định dạng "1 câu + bullet" cũ. Dùng
chung `LLMProvider.classify(prompt, schema)` (structured JSON output) cho cả 3
chỗ — xem _KEYWORDS_SCHEMA/_keywords_prompt_for_*.

M3 (tối giản, làm trước M11): locate_extract khi KHÔNG có page_hint giờ dùng
`app.retrieval.hybrid_search` (BM25 + semantic Gemini Embedding thật, weighted-
fusion 0.5/0.5), thay cho `_keyword_search` (substring đếm thô) trước đây.
`_keyword_search`/`_STOPWORDS` GIỮ LẠI trong file này (không xoá) chỉ để script
eval so sánh accuracy cũ/mới — không còn được gọi trong luồng request thật.

M11: mở rộng phạm vi truy vấn (range, related) + guardrail chống bịa —
milestone ưu tiên độ chính xác/trung thực TUYỆT ĐỐI, cao hơn tốc độ/chi phí.
- Range "từ trang X đến Y": bắt bằng regex thủ công (_RANGE_PATTERN, deterministic,
  không cần LLM).
- related: semantic similarity thuần (không qua LLM generate — 0 rủi ro bịa),
  neo vào trang đang xem (req.page_hint, tái dùng checkbox "dùng trang đang
  xem" có sẵn từ UI), giới hạn CỨNG 5 trang, chỉ trong doc_id hiện tại.
- Guardrail: confidence < _CONFIDENCE_THRESHOLD (từ hybrid_search) → từ chối
  thay vì đoán. Grounding-check: 1 LLM call thứ 2 verify câu trả lời
  locate_extract có thật sự dựa trên context hay không, trước khi trả về.

Bugfix (sau M11): classify_intent mở rộng 3→5 loại (single_page/range/
full_document/related/general_meta) — 3 bug thật (agent bị hỏi "giúp được
gì" thì guardrail từ chối sai; "tóm tắt toàn bộ" 44 trang bị nhồi vào range
nhỏ ra chip vô nghĩa; câu hỏi tổng quan "mục đích tài liệu" lexical-mismatch
với retrieval) đều do thiếu 2 loại route này, không phải bug retrieval.
- general_meta: câu hỏi về CHÍNH AGENT — trả lời TĨNH (không gọi LLM, tránh
  bịa năng lực không thật), KHÔNG qua guardrail (không cần grounding từ tài
  liệu vì không hỏi nội dung tài liệu).
- full_document: tách biệt "range". Câu hỏi tổng quan (mục đích/chủ đề) →
  fast-path dùng trang 1 làm proxy metadata, không guardrail, không map-
  reduce. Yêu cầu tóm tắt toàn bộ thật sự → map-reduce THẬT (M5, trước đây
  chưa từng build — chỉ có bản rút gọn nối-text-1-call cho range nhỏ):
  Map song song tối đa 5 trang/lúc (ThreadPoolExecutor — codebase hiện sync,
  không dùng asyncio.Semaphore như CLAUDE.md gốc mô tả, xem hằng số
  _MAP_REDUCE_MAX_CONCURRENCY), Reduce gộp theo đúng thứ tự trang, cache
  theo doc_id.
- "range" giờ CHỈ còn là case mơ hồ không xác định được biên cụ thể (số
  trang tường minh đã bị regex bắt sớm hơn; muốn toàn bộ tài liệu đã tách
  sang full_document) — route sang clarification hỏi lại biên, KHÔNG còn
  mặc định (1, len(page_index)) như trước (đó chính là nguồn bug #2).

/chat/detail giữ nguyên câu văn đầy đủ (summarize_stream), không đổi sang chip
— dùng cho nút "Xem chi tiết hơn".
"""

import concurrent.futures
import re

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.eval_logger import log_interaction
from app.ingestion.indexer import read_page_index
from app.llm_provider import LLMProvider, LLMProviderError, get_provider
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DetailRequest,
    DetailResponse,
    KeywordChip,
    RelatedPage,
)
from app.retrieval import find_related_pages, hybrid_search

router = APIRouter()

_provider: LLMProvider | None = None


def _get_provider() -> LLMProvider:
    """Khởi tạo provider lười (lazy) và cache lại — tránh crash lúc app
    start nếu LLM_MODE=gemini nhưng thiếu key, và tránh tạo lại genai.Client
    mỗi request."""
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


# Bắt số trang gõ tường minh trong câu hỏi (VD: "Trang 26 nói về điều gì?",
# "slide 12", "page5") — ưu tiên cao nhất, phải tôn trọng tuyệt đối, chạy
# TRƯỚC khi xét tới checkbox "dùng trang đang xem" hay keyword-matching mock.
_PAGE_HINT_PATTERN = re.compile(r"(?:trang|slide|page)\s*#?\s*(\d+)", re.IGNORECASE)


def _extract_page_hint_from_text(message: str) -> int | None:
    match = _PAGE_HINT_PATTERN.search(message)
    return int(match.group(1)) if match else None


# M11: bắt "từ trang X đến Y" / "trang X-Y" / "trang X tới Y" / "slide X→Y".
# Chạy TRƯỚC _PAGE_HINT_PATTERN trong _route (2 số + connector là tín hiệu
# range rõ ràng hơn, không để _PAGE_HINT_PATTERN chỉ bắt được số đầu tiên).
# Có cả biến thể KHÔNG DẤU (den/toi) — tự test phát hiện thiếu bản không dấu
# làm rớt nhầm về single_page (VD "trang 5 den 12" chỉ bắt được số 5 nếu
# thiếu "den" trong pattern, giống hệt kiểu bug page-hint mà cả dự án này
# được thiết kế để tránh).
_RANGE_PATTERN = re.compile(
    r"(?:trang|slide|page)\s*#?\s*(\d+)\s*"
    r"(?:-|–|—|đến|den|tới|toi|->|→)\s*"
    r"(?:trang|slide|page)?\s*#?\s*(\d+)",
    re.IGNORECASE,
)


def _extract_range_from_text(message: str) -> tuple[int, int] | None:
    match = _RANGE_PATTERN.search(message)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start > end:
        start, end = end, start
    return start, end


_AMBIGUOUS_MARKERS = ["đoạn này", "cái này", "chỗ này", "phần này", "giải thích"]

# Các từ chức năng tiếng Việt xuất hiện dày đặc trên mọi trang, không mang
# nhiều tín hiệu chủ đề — loại khỏi keyword search để tránh match sai trang
# (vd. "dùng"/"làm" xuất hiện ở page khác nhiều hơn từ khoá thật sự liên quan).
_STOPWORDS = {
    "va", "la", "co", "cua", "cho", "theo", "dung", "lam", "de", "cac",
    "mot", "nhung", "nay", "do", "khi", "thi", "ma", "hay", "hoac",
    "khong", "duoc", "voi", "trong", "ngoai", "tren", "duoi", "tai",
    "den", "tu", "ra", "vao", "len", "xuong", "di", "ve", "he", "so",
    "và", "là", "có", "của", "cho", "theo", "dùng", "làm", "để", "các",
    "một", "những", "này", "đó", "khi", "thì", "mà", "hay", "hoặc",
    "không", "được", "với", "trong", "ngoài", "trên", "dưới", "tại",
    "đến", "từ", "ra", "vào", "lên", "xuống", "đi", "về", "hệ", "sao",
}


# --- Keyword-chip: schema + prompt (dùng chung cho locate_extract, --------
# --- summarize_selection, summarize_progress qua LLMProvider.classify) ----

_KEYWORDS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["problem", "solution", "metric", "definition", "none"],
                    },
                    "connector": {"type": "boolean"},
                },
                "required": ["text", "category", "connector"],
            },
        },
    },
    "required": ["keywords"],
}

_KEYWORDS_INSTRUCTIONS = """Bạn là trợ lý học tập, cần rút gọn nội dung sau thành 3-6 CỤM TỪ NGẮN (mỗi cụm khoảng 2-4 từ, KHÔNG phải câu văn), giúp học viên đọc lướt hiểu ngay trong ~10 giây khi đang nghe giảng trực tiếp.

Quy tắc bắt buộc:
1. Mỗi cụm phải MANG NỘI DUNG cụ thể (số liệu, tên khái niệm, hành động cụ thể) — KHÔNG được là từ chung chung một mình như "Vấn đề", "Giải pháp", "Kết luận".
2. Mỗi cụm KHÔNG được là một câu văn đầy đủ cắt ngắn — phải là cụm từ/nhãn (label), không chủ ngữ-vị ngữ dài dòng.
3. Chỉ đặt connector=true cho một cụm khi nó THẬT SỰ có quan hệ nhân quả/trình tự rõ ràng với cụm ngay trước nó (sẽ hiển thị mũi tên nối 2 cụm). Phần lớn trường hợp connector=false là bình thường, không ép mọi cụm phải nối nhau.
4. category: "problem" (vấn đề/bottleneck) | "solution" (giải pháp/hướng xử lý) | "metric" (số liệu/chỉ số đo lường) | "definition" (khái niệm/định nghĩa) | "none" (không thuộc loại nào rõ ràng).
5. CHỈ được dùng thông tin CÓ TRONG nội dung được cung cấp bên dưới — TUYỆT ĐỐI KHÔNG dùng kiến thức nền ngoài tài liệu, KHÔNG suy diễn/bịa thêm chi tiết không xuất hiện trong nội dung. Nếu nội dung không đủ để trả lời câu hỏi, hãy tạo 1 cụm duy nhất: {"text": "Không tìm thấy thông tin này trong tài liệu", "category": "none", "connector": false}.

Ví dụ ĐÚNG:
Input: "Bottleneck: Mỗi ticket cần tra cứu 4-5 hệ thống và tóm tắt lại cho khách. Impact: Thời gian xử lý trung bình 8 phút; 40% ticket vượt SLA 5 phút."
Output: {"keywords": [
  {"text": "Bottleneck: tra cứu 4-5 hệ thống/ticket", "category": "problem", "connector": false},
  {"text": "Impact: 40% vượt SLA (5 phút)", "category": "metric", "connector": true}
]}

Ví dụ SAI (quá chung chung — KHÔNG được làm vậy):
{"keywords": [{"text": "Vấn đề", "category": "problem", "connector": false}, {"text": "Tác động", "category": "metric", "connector": true}]}

Ví dụ SAI (vẫn là câu văn cắt ngắn — KHÔNG được làm vậy):
{"keywords": [{"text": "Mỗi ticket cần tra cứu từ 4 đến 5 hệ thống khác nhau và tóm tắt lại thông tin cho khách hàng", "category": "problem", "connector": false}]}
"""


def _keywords_prompt_for_question(question: str, context: str) -> str:
    return (
        f"{_KEYWORDS_INSTRUCTIONS}\n\n"
        f"Câu hỏi của học viên: {question}\n\n"
        f"Nội dung liên quan trong tài liệu:\n{context}\n\n"
        "Trả về đúng JSON theo schema, dựa trên câu hỏi và nội dung trên."
    )


def _keywords_prompt_for_summary(text: str) -> str:
    return (
        f"{_KEYWORDS_INSTRUCTIONS}\n\n"
        f"Nội dung cần tóm tắt thành từ khoá:\n{text}\n\n"
        "Trả về đúng JSON theo schema, tóm tắt nội dung trên."
    )


# --- M11/bugfix: classify_intent (single_page/range/full_document/related/ -
# --- general_meta) ---------------------------------------------------------
#
# Bugfix (phiên sau M11): 3 bug thật phát hiện qua test thủ công đều bắt
# nguồn từ 1 root cause — classify_intent CHỈ có 3 loại, thiếu route cho
# câu hỏi về CHÍNH AGENT và câu hỏi về TOÀN BỘ tài liệu:
# 1. "bạn giúp được gì?" bị route vào single_page → hybrid_search không tìm
#    thấy gì liên quan (đúng, vì đây không phải nội dung tài liệu) →
#    guardrail từ chối SAI (không phải thiếu dữ liệu, mà hỏi sai loại).
# 2. "tóm tắt toàn bộ tài liệu" (44 trang) bị route vào "range" → _route cũ
#    mặc định range không có số tường minh = (1, len(page_index)) → nhồi
#    NGUYÊN VĂN 44 trang vào 1 lệnh classify() duy nhất (thiết kế cho range
#    NHỎ ≤8 trang) → chip vô nghĩa (model phải nén quá nhiều nội dung khác
#    nhau vào 3-6 cụm, mất hết tính cụ thể).
# 3. "tài liệu này có mục đích gì?" — không có 1 trang cụ thể nào "là" câu
#    trả lời (đây là câu hỏi TỔNG QUAN), lexical mismatch với hybrid_search
#    giống hệt case trang 28/29 trước đây → route sai/bị nhồi vào range.
#
# Sửa: thêm "full_document" (tách biệt "range" — range vẫn còn nhưng giờ
# chỉ là leftover case mơ hồ không xác định được biên, route sang
# clarification thay vì đoán = (1, len(page_index)) như cũ) và
# "general_meta" (câu hỏi về NĂNG LỰC agent, không phải nội dung tài liệu
# — trả lời tĩnh, KHÔNG gọi LLM sinh tự do để tránh bịa năng lực không có
# thật, KHÔNG qua guardrail vì không cần grounding từ tài liệu).

_INTENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["single_page", "range", "full_document", "related", "general_meta"],
        },
    },
    "required": ["intent"],
}

_VALID_INTENTS = {"single_page", "range", "full_document", "related", "general_meta"}


def _intent_prompt(message: str) -> str:
    return (
        "Phân loại câu hỏi sau của học viên vào ĐÚNG 1 trong 5 loại:\n"
        '- "single_page": hỏi về 1 nội dung/khái niệm cụ thể, có thể tìm ở '
        "1 trang duy nhất trong tài liệu. DÙNG CẢ CHO câu hỏi về chủ đề "
        "KHÔNG liên quan gì tới tài liệu (VD hỏi thời tiết, giá cổ phiếu, "
        "thể thao) — những câu đó VẪN LÀ single_page (hệ thống sẽ tự tìm "
        "không thấy và từ chối đúng cách), TUYỆT ĐỐI KHÔNG phải general_meta.\n"
        '- "range": yêu cầu tóm tắt một phạm vi NHIỀU TRANG NHƯNG CÓ GIỚI '
        'HẠN (không phải toàn bộ tài liệu), VD "tóm tắt phần vừa học", mà '
        "không nói rõ số trang cụ thể.\n"
        '- "full_document": (a) yêu cầu tóm tắt/tổng hợp TOÀN BỘ tài liệu '
        '("tóm tắt toàn bộ", "tóm tắt cả bài", "từ đầu đến giờ" theo nghĩa '
        'toàn bộ), HOẶC (b) hỏi TỔNG QUAN về tài liệu nói chung — mục đích, '
        'chủ đề, nội dung chính là gì (VD "tài liệu này nói về gì?", "mục '
        'đích của bài giảng là gì?") — cả 2 đều không gắn với 1 trang cụ '
        "thể mà là CẢ tài liệu.\n"
        '- "related": hỏi có còn nội dung/trang nào KHÁC liên quan tới trang/'
        'chủ đề đang xem hay không (VD "còn phần nào khác liên quan không?").\n'
        '- "general_meta": CHỈ khi câu hỏi nói về CHÍNH CÔNG CỤ/TRỢ LÝ này '
        "đang dùng — nó LÀ GÌ, LÀM ĐƯỢC GÌ (chức năng), hoặc lời chào xã "
        'giao thuần tuý (VD "bạn là ai?", "bạn/app/công cụ này giúp được '
        'gì?", "hướng dẫn cách dùng", "xin chào", "cảm ơn"). KHÔNG dùng '
        "loại này cho bất kỳ câu hỏi nào có thể liên quan tới NỘI DUNG "
        "(kể cả khi không chắc tài liệu có đề cập hay không) — nếu phân "
        "vân, LUÔN chọn single_page thay vì general_meta.\n\n"
        "Ví dụ:\n"
        '"Giá cổ phiếu Apple hôm nay bao nhiêu?" → single_page (câu hỏi về '
        "SỰ KIỆN/THÔNG TIN, không phải hỏi về công cụ này).\n"
        '"5 câu hỏi nên hỏi stakeholder trong discovery interview là gì?" → '
        "single_page (đang hỏi về NỘI DUNG kiến thức, dù dùng chữ 'câu hỏi' "
        "cũng không phải hỏi về chính agent).\n"
        '"Bạn giúp được gì cho tôi?" → general_meta (hỏi thẳng về chức năng '
        "của CHÍNH công cụ này).\n\n"
        f"Câu hỏi: {message}\n\n"
        "Trả về đúng JSON theo schema."
    )


def _classify_query_intent(message: str) -> str:
    """Trả về 1 trong 5 loại ở _VALID_INTENTS. Lỗi provider hoặc kết quả
    không hợp lệ → mặc định "single_page" (an toàn nhất — vẫn đi qua
    hybrid_search + guardrail, không im lặng bịa)."""
    try:
        provider = _get_provider()
        result = provider.classify(_intent_prompt(message), _INTENT_SCHEMA, model_tier="light")
        intent = result.get("intent") if isinstance(result, dict) else None
    except LLMProviderError:
        return "single_page"
    return intent if intent in _VALID_INTENTS else "single_page"


# --- M11 Phần B: guardrail chống bịa + grounding-check ---------------------

# Ngưỡng đo thật (sau khi sửa bug min-max ở retrieval.py — xem docstring đầu
# file đó): 5 câu ĐÚNG phạm vi fused 0.657-0.716, 5 câu SAI/ngoài phạm vi
# fused 0.296-0.520 — có khoảng trống rõ ràng [0.520, 0.657], chọn 0.58 gần
# giữa khoảng trống đó (lệch nhẹ về phía an toàn — margin ~0.06 với BAD-max,
# ~0.08 với GOOD-min). Xem PROGRESS.md mục M11 để biết số liệu đầy đủ.
_CONFIDENCE_THRESHOLD = 0.58

# Disambiguation: margin (confidence top1 - top2) đo thật trên 5 câu — case
# "trang 28/29" (2 trang cùng nói về stakeholder, 1 trang divider lặp từ
# khoá của trang chi tiết) có margin=0.025; 1 case khác (AI-Fit Matrix, top1
# trang 20 vs runner-up trang 19 — cũng cùng cụm chủ đề Rule/LLM/Agent) có
# margin=0.001; các case rõ ràng còn lại có margin 0.030-0.065. Chọn 0.03:
# bắt được cả 2 case mơ hồ thật (0.025, 0.001), KHÔNG bắt nhầm case margin=
# 0.030 (so sánh strict "<", 0.030 không < 0.030). Xem PROGRESS.md mục M11.
_MARGIN_THRESHOLD = 0.03

# Range nhỏ (nối text trực tiếp) vs lớn (NÊN map-reduce thật — M5 chưa có,
# tạm dùng chung cơ chế nối trực tiếp, xem docstring đầu file).
_RANGE_SMALL_MAX_PAGES = 8

_NOT_FOUND_ANSWER = (
    "Tôi không tìm thấy thông tin này trong tài liệu, bạn có thể kiểm tra "
    "lại câu hỏi hoặc chỉ rõ trang không?"
)

# --- general_meta: câu hỏi về CHÍNH AGENT ----------------------------------
# Trả lời TĨNH (không gọi LLM) — tránh rủi ro model tự bịa năng lực không có
# thật (VD hứa làm được việc ngoài phạm vi MVP). Nội dung khớp đúng Mục tiêu
# sản phẩm trong CLAUDE.md, không thêm/bớt.
_GENERAL_META_ANSWER = (
    "Mình là trợ lý học tập cho tài liệu bạn đang mở. Mình giúp được:\n"
    "- Tìm nội dung theo trang: hỏi trực tiếp hoặc bôi đen đoạn cần hỏi\n"
    "- Tóm tắt: đoạn bôi đen, đến trang hiện tại, 1 khoảng trang, hoặc toàn "
    "bộ tài liệu\n"
    "- Tìm các trang khác liên quan tới trang bạn đang xem\n"
    "Cứ hỏi hoặc bôi đen trên slide để bắt đầu nhé!"
)

# --- full_document: map-reduce thật (M5) -----------------------------------
# CLAUDE.md M5 gốc: "Map: tóm tắt từng trang song song (async)... giới hạn
# concurrency (Semaphore, tối đa 5 call song song)". Codebase hiện tại toàn
# bộ SYNC (FastAPI route là `def`, LLMProvider là sync HTTP call blocking) —
# dùng ThreadPoolExecutor(max_workers=5) thay vì asyncio.Semaphore để đạt
# đúng giới hạn concurrency mà không phải refactor toàn bộ app sang async
# (rủi ro/quy mô vượt xa phạm vi bugfix này).
_MAP_REDUCE_MAX_CONCURRENCY = 5
_MAP_SUMMARY_MAX_TOKENS = 256

# Cache theo doc_id (CLAUDE.md ghi "Cache theo (doc_id, up_to_page)" — với
# full_document, up_to_page luôn = trang cuối cùng nên chỉ cần key doc_id).
# In-memory, mất khi restart server — đủ cho MVP, không cần cache bền theo
# đúng tinh thần "KHÔNG làm quá tay".
_full_document_cache: dict[str, list[KeywordChip]] = {}

# Câu hỏi TỔNG QUAN (mục đích/chủ đề tài liệu) — fast-path dùng trang 1 làm
# proxy "metadata" (slide đầu thường là tiêu đề/mở bài), KHÔNG chạy map-reduce
# đầy đủ, KHÔNG qua guardrail retrieval-match — đúng yêu cầu "không cần
# retrieval match chính xác 100% cho loại câu hỏi tổng quan kiểu này".
_OVERVIEW_MARKERS = [
    "mục đích", "muc dich", "nói về", "noi ve", "giới thiệu", "gioi thieu",
    "tổng quan", "tong quan", "chủ đề", "chu de", "nội dung chính", "noi dung chinh",
]


def _looks_like_overview_question(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _OVERVIEW_MARKERS)

_GROUNDING_SCHEMA: dict = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}},
    "required": ["grounded"],
}


def _grounding_check_prompt(answer_text: str, context: str) -> str:
    return (
        "Kiểm tra xem CÂU TRẢ LỜI dưới đây có hoàn toàn dựa trên NỘI DUNG TÀI "
        "LIỆU được cung cấp hay không — không có chi tiết nào bị suy diễn/bịa "
        "thêm mà không xuất hiện (trực tiếp hoặc gần nghĩa) trong nội dung.\n\n"
        f"Nội dung tài liệu:\n{context}\n\n"
        f"Câu trả lời cần kiểm tra:\n{answer_text}\n\n"
        'Trả JSON {"grounded": true} nếu câu trả lời chỉ dùng thông tin có '
        'trong tài liệu trên. Trả {"grounded": false} nếu có BẤT KỲ chi tiết '
        "nào không xuất hiện trong tài liệu."
    )


def _is_grounded(answer_text: str, context: str) -> bool:
    """Lỗi provider lúc verify → coi như KHÔNG chắc chắn grounded, nhưng vẫn
    cho qua (fail-open) thay vì chặn hết câu trả lời chỉ vì bước verify lỗi
    mạng/quota — guardrail chính vẫn là _CONFIDENCE_THRESHOLD ở bước trước."""
    try:
        provider = _get_provider()
        result = provider.classify(
            _grounding_check_prompt(answer_text, context), _GROUNDING_SCHEMA, model_tier="light"
        )
        return bool(result.get("grounded", True)) if isinstance(result, dict) else True
    except LLMProviderError:
        return True


def _parse_keywords(result: dict) -> list[KeywordChip]:
    raw = result.get("keywords") if isinstance(result, dict) else None
    if not raw:
        raise ValueError("Gemini không trả về keywords hợp lệ")
    return [KeywordChip(**item) for item in raw]


def _flatten_keywords(chips: list[KeywordChip]) -> str:
    """Bản text-only của keywords — dùng cho eval log và làm answer fallback
    (ChatResponse.answer vẫn là str bắt buộc, kể cả khi có keywords)."""
    out = chips[0].text
    for chip in chips[1:]:
        out += (" → " if chip.connector else " · ") + chip.text
    return out


def _keywords_response(mode: str, chips: list[KeywordChip], citation_page: int | None = None) -> ChatResponse:
    return ChatResponse(mode=mode, answer=_flatten_keywords(chips), keywords=chips, citation_page=citation_page)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    page_index = read_page_index(req.doc_id)
    if not page_index:
        raise HTTPException(status_code=404, detail="doc_id không tồn tại, hãy upload trước")

    response = _route(req, page_index)
    log_interaction(req, response)
    return response


def _route(req: ChatRequest, page_index: dict[str, str]) -> ChatResponse:
    if req.selected_text and req.selected_text.strip():
        return _summarize_selection(req.selected_text.strip())

    if req.up_to_page:
        return _summarize_progress(req.up_to_page, page_index)

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message rỗng")

    # M11: "từ trang X đến Y" bắt bằng regex TRƯỚC — deterministic, không
    # cần LLM, và phải chạy trước _PAGE_HINT_PATTERN (regex 1-số sẽ chỉ bắt
    # được số đầu của 1 range, làm sai ý người hỏi).
    range_match = _extract_range_from_text(message)
    if range_match:
        start, end = range_match
        return _handle_range(start, end, page_index)

    # Số trang gõ TƯỜNG MINH trong câu hỏi — ưu tiên tuyệt đối, không qua
    # classify_intent (đây chính là loại tín hiệu mà bug "page UI không
    # khớp retrieval" ở hệ thống cũ đáng lẽ phải tôn trọng tuyệt đối).
    text_page_hint = _extract_page_hint_from_text(message)
    if text_page_hint is not None:
        return _locate_extract(message, text_page_hint, page_index, req.doc_id)

    if _looks_ambiguous(message):
        return ChatResponse(
            mode="clarification",
            needs_clarification=True,
            answer=(
                "Câu hỏi của bạn hơi mơ hồ — bạn có thể bôi đen đoạn cụ thể trên "
                "slide, hoặc cho biết trang/số slide bạn đang hỏi không?"
            ),
        )

    # Không có số trang tường minh, không mơ hồ → classify 5 loại. Checkbox
    # "dùng trang đang xem" (req.page_hint) KHÔNG còn tự động ép single_page
    # như trước M11 — giờ dùng làm page_hint cho single_page HOẶC làm trang
    # neo cho related, tuỳ intent LLM phân loại (cho phép "còn phần nào liên
    # quan không?" hoạt động đúng khi checkbox đang bật).
    intent = _classify_query_intent(message)
    if intent == "general_meta":
        # Câu hỏi về CHÍNH AGENT — trả lời tĩnh, KHÔNG qua guardrail/retrieval
        # (không cần grounding từ tài liệu vì không hỏi về nội dung tài liệu).
        return ChatResponse(mode="general_meta", answer=_GENERAL_META_ANSWER)
    if intent == "full_document":
        return _handle_full_document(req.doc_id, message, page_index)
    if intent == "range":
        # Bugfix: KHÔNG còn mặc định (1, len(page_index)) như trước — đó
        # chính là nguồn gây bug "44 trang nhồi vào range nhỏ". "range" giờ
        # chỉ còn là case mơ hồ KHÔNG xác định được biên cụ thể (câu hỏi rõ
        # ràng số trang tường minh đã bị regex bắt sớm hơn ở trên; câu hỏi
        # muốn TOÀN BỘ tài liệu đã tách sang "full_document") — hỏi lại thay
        # vì đoán, đúng tinh thần "không đoán bừa".
        return _clarification_response(
            "Bạn muốn tóm tắt từ trang mấy đến trang mấy? Bạn có thể ghi rõ, "
            'VD "từ trang 5 đến trang 12".'
        )
    if intent == "related":
        return _handle_related(req.doc_id, req.page_hint, page_index)

    return _locate_extract(message, req.page_hint, page_index, req.doc_id)


def _summarize_selection(selected_text: str) -> ChatResponse:
    try:
        provider = _get_provider()
        result = provider.classify(_keywords_prompt_for_summary(selected_text), _KEYWORDS_SCHEMA)
        chips = _parse_keywords(result)
    except LLMProviderError as exc:
        return ChatResponse(
            mode="summarize_selection",
            answer=f"Không thể tóm tắt lúc này ({exc}). Vui lòng thử lại sau.",
        )
    except (ValueError, ValidationError):
        return ChatResponse(
            mode="summarize_selection",
            answer="Không rút được từ khoá từ đoạn này, vui lòng thử lại.",
        )
    return _keywords_response("summarize_selection", chips)


def _summarize_page_range(start: int, end: int, page_index: dict[str, str], mode: str) -> ChatResponse:
    """Nối text các trang [start, end] TRỰC TIẾP rồi gọi LLM 1 lần duy nhất.

    Dùng chung cho summarize_progress (start=1 cố định, nút "Tóm tắt đến
    trang N" ở FE) VÀ range query mới ở M11 (_handle_range). Chưa phải
    map-reduce từng trang + cache theo (doc_id, up_to_page) như CLAUDE.md mô
    tả cho bản MVP cuối cùng (việc đó thuộc M5, chưa làm) — đây là bản rút
    gọn dùng được cho CẢ range nhỏ lẫn lớn hiện tại. Khi M5 làm map-reduce
    thật, chỉ cần thay implementation của nhánh range LỚN trong
    _handle_range, không đụng tới range nhỏ/summarize_progress.
    """
    n_total = len(page_index)
    start = max(1, start)
    end = min(end, n_total)
    if start > end:
        return ChatResponse(mode=mode, answer="Phạm vi trang không hợp lệ trong tài liệu này.")

    pages_text = "\n\n".join(
        page_index[str(p)] for p in range(start, end + 1) if page_index.get(str(p))
    )
    if not pages_text.strip():
        return ChatResponse(mode=mode, answer="Chưa có nội dung để tóm tắt trong phạm vi này.")

    try:
        provider = _get_provider()
        result = provider.classify(_keywords_prompt_for_summary(pages_text), _KEYWORDS_SCHEMA)
        chips = _parse_keywords(result)
    except LLMProviderError as exc:
        return ChatResponse(mode=mode, answer=f"Không thể tóm tắt lúc này ({exc}). Vui lòng thử lại sau.")
    except (ValueError, ValidationError):
        return ChatResponse(mode=mode, answer="Không rút được từ khoá từ nội dung này, vui lòng thử lại.")
    return _keywords_response(mode, chips)


def _summarize_progress(up_to_page: int, page_index: dict[str, str]) -> ChatResponse:
    return _summarize_page_range(1, up_to_page, page_index, mode="summarize_progress")


def _handle_range(start: int, end: int, page_index: dict[str, str]) -> ChatResponse:
    """M11 Phần A #3. n_pages ≤ _RANGE_SMALL_MAX_PAGES (đã test/báo cáo = 8):
    range nhỏ, nối trực tiếp. > 8: range lớn, NÊN map-reduce thật (M5 chưa
    có) — tạm dùng chung _summarize_page_range, xem docstring hàm đó."""
    return _summarize_page_range(start, end, page_index, mode="range")


def _map_summarize_page(page_number: int, text: str, provider: LLMProvider) -> tuple[int, str]:
    """1 tác vụ Map — tóm tắt ngắn 1 trang thành văn xuôi (KHÔNG phải
    keyword-chip — đây là bước trung gian nội bộ, chỉ Reduce mới sinh câu
    trả lời cuối hiển thị cho user). Lỗi 1 trang → bỏ qua trang đó (chuỗi
    rỗng), KHÔNG làm hỏng cả map-reduce."""
    try:
        return page_number, provider.summarize(text, max_tokens=_MAP_SUMMARY_MAX_TOKENS)
    except LLMProviderError:
        return page_number, ""


def _handle_full_document(doc_id: str, message: str, page_index: dict[str, str]) -> ChatResponse:
    """full_document: (a) câu hỏi TỔNG QUAN (mục đích/chủ đề) → fast-path
    dùng trang 1 làm proxy metadata, không guardrail, không map-reduce đầy
    đủ. (b) yêu cầu tóm tắt toàn bộ thật sự → map-reduce thật (Map: tóm tắt
    song song từng trang, tối đa 5 call cùng lúc; Reduce: gộp theo đúng thứ
    tự trang thành keyword-chip cuối cùng), cache theo doc_id."""
    if not page_index:
        return ChatResponse(mode="full_document", answer="Chưa có nội dung để tóm tắt.")

    if _looks_like_overview_question(message):
        first_page = min((int(p) for p in page_index), default=None)
        if first_page is not None and page_index.get(str(first_page), "").strip():
            return _summarize_page_range(first_page, first_page, page_index, mode="full_document")

    if doc_id in _full_document_cache:
        return _keywords_response("full_document", _full_document_cache[doc_id])

    provider = _get_provider()
    pages = sorted(
        ((int(p), t) for p, t in page_index.items() if t.strip()), key=lambda item: item[0]
    )
    if not pages:
        return ChatResponse(mode="full_document", answer="Chưa có nội dung để tóm tắt.")

    # Map: tóm tắt song song từng trang, giới hạn tối đa
    # _MAP_REDUCE_MAX_CONCURRENCY call cùng lúc (xem lý do dùng
    # ThreadPoolExecutor thay asyncio.Semaphore ở docstring hằng số đầu file).
    page_summaries: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAP_REDUCE_MAX_CONCURRENCY) as executor:
        futures = [executor.submit(_map_summarize_page, p, t, provider) for p, t in pages]
        for future in concurrent.futures.as_completed(futures):
            p, summary = future.result()
            page_summaries[p] = summary

    # Reduce: gộp theo ĐÚNG thứ tự trang (không phải thứ tự hoàn thành song
    # song) thành 1 bản tóm tắt mạch lạc, dạng keyword-chip như mọi nơi khác.
    combined = "\n".join(
        f"Trang {p}: {s}" for p, s in sorted(page_summaries.items()) if s.strip()
    )
    if not combined.strip():
        return ChatResponse(
            mode="full_document",
            answer="Không thể tóm tắt lúc này (lỗi tóm tắt từng trang). Vui lòng thử lại sau.",
        )

    try:
        result = provider.classify(_keywords_prompt_for_summary(combined), _KEYWORDS_SCHEMA)
        chips = _parse_keywords(result)
    except LLMProviderError as exc:
        return ChatResponse(mode="full_document", answer=f"Không thể tóm tắt lúc này ({exc}). Vui lòng thử lại sau.")
    except (ValueError, ValidationError):
        return ChatResponse(mode="full_document", answer="Không rút được từ khoá, vui lòng thử lại.")

    _full_document_cache[doc_id] = chips
    return _keywords_response("full_document", chips)


def _handle_related(doc_id: str, anchor_page: int | None, page_index: dict[str, str]) -> ChatResponse:
    """M11 Phần A #4. Cần trang neo (anchor) — lấy từ req.page_hint (checkbox
    "dùng trang đang xem"). Không có anchor → hỏi lại thay vì đoán bừa."""
    if not anchor_page or str(anchor_page) not in page_index:
        return ChatResponse(
            mode="clarification",
            needs_clarification=True,
            answer=(
                "Bạn đang xem trang nào? Bật ô \"Dùng trang đang xem làm gợi ý\" "
                "hoặc cho tôi biết số trang để tôi tìm nội dung liên quan."
            ),
        )

    related = find_related_pages(doc_id, anchor_page, page_index)
    if not related:
        return ChatResponse(
            mode="related",
            answer=f"Không tìm thấy trang nào khác liên quan rõ ràng tới trang {anchor_page}.",
        )

    related_pages = [RelatedPage(page=p, similarity=round(s, 3)) for p, s in related]
    page_list = ", ".join(str(p) for p, _ in related)
    return ChatResponse(
        mode="related",
        answer=f"Tìm thấy {len(related)} trang liên quan tới trang {anchor_page}: trang {page_list}.",
        related_pages=related_pages,
        citation_page=anchor_page,
    )


def _looks_ambiguous(message: str) -> bool:
    lowered = message.lower()
    return len(message) < 40 and any(marker in lowered for marker in _AMBIGUOUS_MARKERS)


def _clarification_response(answer: str) -> ChatResponse:
    """Tái dùng ĐÚNG field/flow clarification đã có (needs_clarification=true)
    — không thêm state machine riêng cho disambiguation, đúng yêu cầu M11."""
    return ChatResponse(mode="clarification", needs_clarification=True, answer=answer)


def _locate_extract(
    message: str, page_hint: int | None, page_index: dict[str, str], doc_id: str
) -> ChatResponse:
    if page_hint and str(page_hint) in page_index:
        # Page hint tường minh (text hoặc checkbox) = confidence tuyệt đối,
        # KHÔNG qua guardrail/disambiguation — đúng nguyên tắc "page_hint ưu
        # tiên tuyệt đối" của CLAUDE.md, không để các cơ chế đó chặn nhầm 1
        # chỉ định tường minh của người dùng.
        page = page_hint
    else:
        # M3: hybrid BM25 + semantic (Gemini Embedding thật), weighted-fusion.
        result = hybrid_search(message, doc_id, page_index)
        # M11 Phần B: guardrail chống bịa — confidence thấp = KHÔNG đoán,
        # từ chối thẳng thay vì đi tiếp gọi LLM sinh câu trả lời có thể sai.
        if result.page is None or result.confidence < _CONFIDENCE_THRESHOLD:
            return ChatResponse(mode="locate_extract", answer=_NOT_FOUND_ANSWER)
        # M11: disambiguation — top1/top2 quá sát nhau (margin nhỏ) = mơ hồ
        # thật (2 trang cùng liên quan), hỏi lại thay vì âm thầm chọn top1
        # có thể sai (VD case trang 28 "divider" vs 29 "chi tiết").
        if result.runner_up_page is not None and result.margin < _MARGIN_THRESHOLD:
            return _clarification_response(
                f"Mình tìm thấy nội dung liên quan ở cả trang {result.page} và "
                f"trang {result.runner_up_page} — bạn muốn xem trang nào?"
            )
        page = result.page

    if page is None:
        return ChatResponse(mode="locate_extract", answer=_NOT_FOUND_ANSWER)

    page_text = page_index.get(str(page), "")
    if not page_text.strip():
        return ChatResponse(
            mode="locate_extract",
            citation_page=page,
            answer=f"Trang {page} không có nội dung văn bản (có thể là hình ảnh/slide trống).",
        )

    # Nội dung trả về phải đúng với text gốc lấy từ page_index (ground-truth
    # từ PDF, không suy diễn). classify() trả structured keyword-chip (M9) —
    # thay cho answer_stream cũ ("1 câu + bullet"). Bấm "Xem chi tiết hơn" ở
    # FE vẫn gọi /chat/detail (dùng summarize_stream, câu văn đầy đủ).
    try:
        provider = _get_provider()
        result = provider.classify(_keywords_prompt_for_question(message, page_text), _KEYWORDS_SCHEMA)
        chips = _parse_keywords(result)
    except LLMProviderError as exc:
        return ChatResponse(
            mode="locate_extract",
            citation_page=page,
            answer=f"Không thể trả lời lúc này ({exc}). Vui lòng thử lại sau.",
        )
    except (ValueError, ValidationError):
        return ChatResponse(
            mode="locate_extract",
            citation_page=page,
            answer="Không rút được từ khoá từ nội dung trang này, vui lòng thử lại.",
        )

    # M11 Phần B: grounding-check — LLM call thứ 2, tự kiểm tra câu trả lời
    # vừa sinh có thật sự dựa trên page_text hay không, trước khi trả về
    # user. Đây là lớp phòng thủ THỨ 2 (khác guardrail confidence ở trên —
    # confidence đo độ tin cậy của RETRIEVAL, cái này đo độ trung thực của
    # CÂU TRẢ LỜI đã sinh ra).
    answer_text = _flatten_keywords(chips)
    if not _is_grounded(answer_text, page_text):
        return ChatResponse(mode="locate_extract", citation_page=page, answer=_NOT_FOUND_ANSWER)

    return _keywords_response("locate_extract", chips, citation_page=page)


@router.post("/chat/detail", response_model=DetailResponse)
def chat_detail(req: DetailRequest) -> DetailResponse:
    """Nút "Xem chi tiết hơn" dưới câu trả lời ngắn của locate_extract —
    trả bản giải thích đầy đủ hơn (2-3 câu, dùng summarize_stream có sẵn),
    KHÔNG thay thế câu trả lời ngắn, chỉ mở rộng thêm bên dưới ở FE."""
    page_index = read_page_index(req.doc_id)
    if not page_index:
        raise HTTPException(status_code=404, detail="doc_id không tồn tại, hãy upload trước")

    page_text = page_index.get(str(req.page), "")
    if not page_text.strip():
        raise HTTPException(status_code=404, detail=f"Trang {req.page} không có nội dung")

    try:
        provider = _get_provider()
        detail = "".join(provider.summarize_stream(page_text))
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=f"Không thể lấy chi tiết lúc này ({exc})") from exc

    return DetailResponse(page=req.page, detail=detail)


def _keyword_search(message: str, page_index: dict[str, str]) -> int | None:
    words = [
        w.lower()
        for w in re.findall(r"\w+", message)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]
    if not words:
        return None
    best_page, best_score = None, 0
    for page_str, text in page_index.items():
        lowered = text.lower()
        score = sum(lowered.count(w) for w in words)
        if score > best_score:
            best_score, best_page = score, int(page_str)
    return best_page
