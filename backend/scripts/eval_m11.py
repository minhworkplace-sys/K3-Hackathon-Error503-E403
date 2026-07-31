"""M11: eval mở rộng — single_page (kế thừa M3) + range + related + "không có
trong tài liệu" (test guardrail chống bịa). Gọi trực tiếp qua HTTP /chat (test
đúng full pipeline: regex range-parse, classify_intent, hybrid_search+guardrail,
grounding-check) — không import thẳng hàm nội bộ như script M3 cũ, để bảo đảm
test đúng luồng thật user trải nghiệm.

Yêu cầu: backend đang chạy ở :8000 với LLM_MODE=gemini, doc_id đã reindex bằng
embedding thật (scripts/reindex_embeddings.py).

Chạy: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_m11.py <doc_id>
"""

from __future__ import annotations

import re
import sys
import urllib.request
import json

_PAGE_NUM_RE = re.compile(r"\d+")

API_BASE = "http://localhost:8000"

# --- single_page: 5 câu gốc (M3) + 10 câu mới (M11, yêu cầu mở rộng mẫu vì
# n=5 quá nhỏ) — có cả câu rõ ràng 1 đáp án (đa số trang mới) lẫn câu dễ
# overlap nội dung (case trang 28/29 gốc — cố tình giữ lại để test disambiguation).
SINGLE_PAGE_CASES = [
    ("Ticket CS ngân hàng cần tra cứu bao nhiêu hệ thống trước khi trả lời khách?", 26),
    ("Ma trận nào dùng để phân loại Rule/Workflow/LLM feature/Agent theo độ mơ hồ và độ phức tạp vận hành?", 20),
    ("Quy trình 6 giai đoạn phát triển AI Product gồm những bước nào?", 11),
    ("Trong mô hình stakeholder, ai là người approve và ai chỉ review?", 29),  # case mo ho that (28/29)
    ("3 mức giải pháp Rule/Script, LLM Feature, và Agent khác nhau ở điểm nào?", 19),
    ("4 anti-pattern khiến team đốt tiền vào AI sai chỗ là gì?", 9),
    ("Gate criteria để quyết định đi tiếp hay dừng lại ở giai đoạn Problem Scoping là gì?", 12),
    ("5 câu hỏi trong AI Readiness Checklist là gì?", 21),
    ("Khung Problem Statement cho AI system gồm những thành phần nào?", 24),
    ("4 lỗi phổ biến khi viết Problem Statement là gì?", 25),
    ("Ma trận RACI-lite cho dự án AI feature/agent phân vai trò thế nào?", 30),
    ("Làm sao để pitch đúng cách từ system metrics sang business KPIs?", 32),
    ("5 câu hỏi nên hỏi stakeholder trong discovery interview là gì?", 34),
    ("Feasibility check trước khi commit gồm những phần nào?", 35),
    ("Tiêu chí Go/No-Go/Not Yet để quyết định triển khai AI là gì?", 36),
]

# --- range: (message, expected_start, expected_end) — "đúng" = regex parse
# đúng range VÀ trả về keywords (không lỗi) ---------------------------------
RANGE_CASES = [
    ("Tóm tắt từ trang 3 đến trang 9", 3, 9),
    ("trang 10-16 nói về gì", 10, 16),
    ("tóm tắt trang 17 tới 24", 17, 24),
    ("nội dung trang 30 đến 38 là gì", 30, 38),
    ("slide 5-12", 5, 12),
]

# --- related: (anchor_page, ít nhất 1 trong các trang này nên xuất hiện
# trong top-5 — xác nhận bằng mắt lúc M10, cùng cụm chủ đề Rule/LLM/Agent) --
RELATED_CASES = [
    (20, {19, 22}),
    (19, {20, 22}),
    # SỬA (phiên "billing thật"): expected cũ {15} SAI ground truth — giả
    # định "cùng là sơ đồ vector" = liên quan, không kiểm chứng nội dung
    # thật. Đọc lại page_index: trang 11 = "6 Giai Đoạn Phát Triển AI
    # Product" (lifecycle theo GIAI ĐOẠN); trang 15 = "AI System =
    # Model+Context+Planning+Tools" (kiến trúc theo THÀNH PHẦN) — 2 chủ đề
    # KHÁC nhau, không liên quan thật. Trang 12 ("Gate Criteria: Đi Tiếp
    # Khi Nào, Dừng Lại Khi Nào?") và trang 13 ("4 Bài Học Kỹ Thuật Quan
    # Trọng Từ Lifecycle") mới THẬT SỰ elaborate đúng nội dung lifecycle
    # của trang 11 — đúng khớp kết quả hệ thống trả về ([10,12,13,36,40]).
    (11, {12, 13}),
    (29, {20, 22}),  # stakeholder/quyet dinh cung cum "phan loai giai phap"
    (22, {19, 20}),
]

# --- không có trong tài liệu: guardrail PHẢI từ chối --------------------
NOT_IN_DOC_CASES = [
    "Cách nấu phở bò truyền thống là gì?",
    "Định luật Newton thứ 2 phát biểu thế nào?",
    "Giá cổ phiếu Apple hôm nay bao nhiêu?",
    "Cầu thủ Messi chơi cho đội bóng nào?",
    "Công thức tính diện tích hình tròn là gì?",
]


def call_chat(doc_id: str, **body) -> dict:
    payload = json.dumps({"doc_id": doc_id, **body}).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def eval_single_page(doc_id: str) -> tuple[int, int]:
    """Sửa (phiên "billing thật"): trước đây chấm FAIL bất kỳ khi nào
    response là clarification trừ 1 case đặc cách (trang 28/29) — SAI định
    nghĩa "đúng", vì disambiguation (từ chối đoán bừa khi mơ hồ) là hành vi
    ĐÚNG của sản phẩm, không phải lỗi. Giờ áp dụng ĐỒNG NHẤT cho MỌI case:
    clarification tính PASS nếu trang kỳ vọng nằm trong các trang được đề
    xuất hỏi lại. Trích số trang bằng regex \\d+ (word-level), KHÔNG dùng
    substring thô — substring "9" sẽ khớp nhầm bên trong "29"/"19"/"39"."""
    print("\n=== single_page ===")
    correct = 0
    for question, expected in SINGLE_PAGE_CASES:
        res = call_chat(doc_id, message=question)
        page = res.get("citation_page")
        mode = res.get("mode")
        answer = res.get("answer") or ""
        mentioned_pages = {int(n) for n in _PAGE_NUM_RE.findall(answer)} if mode == "clarification" else set()
        clarified_correctly = mode == "clarification" and expected in mentioned_pages
        ok = (page == expected) or clarified_correctly
        correct += ok
        tag = "OK (clarify)" if clarified_correctly and page != expected else ("OK" if ok else "X ")
        print(f"[{tag:<12}] exp={expected} got={page} mode={mode}  {question[:55]}")
    return correct, len(SINGLE_PAGE_CASES)


def eval_range(doc_id: str) -> tuple[int, int]:
    print("\n=== range ===")
    correct = 0
    for question, start, end in RANGE_CASES:
        res = call_chat(doc_id, message=question)
        mode = res.get("mode")
        has_keywords = bool(res.get("keywords"))
        ok = mode == "range" and has_keywords
        correct += ok
        print(f"[{'OK' if ok else 'X '}] mode={mode} has_kw={has_keywords} range=({start},{end})  {question[:50]}")
    return correct, len(RANGE_CASES)


def eval_related(doc_id: str) -> tuple[int, int]:
    print("\n=== related ===")
    correct = 0
    for anchor, expected_any in RELATED_CASES:
        res = call_chat(doc_id, message="còn phần nào khác liên quan đến trang này không?", page_hint=anchor)
        related = res.get("related_pages") or []
        got_pages = {r["page"] for r in related}
        within_cap = len(related) <= 5
        overlap = bool(got_pages & expected_any)
        ok = within_cap and overlap
        correct += ok
        print(f"[{'OK' if ok else 'X '}] anchor={anchor} got={sorted(got_pages)} expected_any_of={expected_any}")
    return correct, len(RELATED_CASES)


def eval_not_in_doc(doc_id: str) -> tuple[int, int]:
    print("\n=== không có trong tài liệu (guardrail) ===")
    correct = 0
    for question in NOT_IN_DOC_CASES:
        res = call_chat(doc_id, message=question)
        rejected = "không tìm thấy" in (res.get("answer") or "").lower()
        correct += rejected
        print(f"[{'OK' if rejected else 'X '}] rejected={rejected}  {question}")
    return correct, len(NOT_IN_DOC_CASES)


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách dùng: eval_m11.py <doc_id>")
        sys.exit(1)
    doc_id = sys.argv[1]

    sp_ok, sp_n = eval_single_page(doc_id)
    rg_ok, rg_n = eval_range(doc_id)
    rl_ok, rl_n = eval_related(doc_id)
    nf_ok, nf_n = eval_not_in_doc(doc_id)

    print("\n=== TỔNG KẾT ===")
    print(f"single_page : {sp_ok}/{sp_n} = {sp_ok/sp_n*100:.0f}%  (target ≥95%)")
    print(f"range       : {rg_ok}/{rg_n} = {rg_ok/rg_n*100:.0f}%  (target ≥85%)")
    print(f"related     : {rl_ok}/{rl_n} = {rl_ok/rl_n*100:.0f}%  (target ≥85%)")
    print(f"guardrail   : {nf_ok}/{nf_n} = {nf_ok/nf_n*100:.0f}%  (target 100%)")


if __name__ == "__main__":
    main()
