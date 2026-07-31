"""M3 (tối giản): so sánh accuracy retrieval CŨ (_keyword_search — substring
đếm thô) vs MỚI (hybrid_search — BM25 + Gemini Embedding thật, weighted-fusion)
trên 5 câu hỏi single-page. KHÔNG có script/bộ câu hỏi cũ nào để "chạy lại" —
đã grep toàn bộ project, không tìm thấy — nên viết mới 5 câu, neo vào đúng nội
dung thật đã xác nhận bằng mắt (render PNG) trên file test 44 trang trong
`storage/uploads/`.

Bắt buộc: doc_id đã được reindex bằng embedding thật trước (xem
scripts/reindex_embeddings.py) và LLM_MODE=gemini lúc chạy script này.

Chạy: cd backend && LLM_MODE=gemini PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_m3_single_page.py <doc_id>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.indexer import read_page_index  # noqa: E402
from app.retrieval import hybrid_search  # noqa: E402
from app.routers.chat import _keyword_search  # noqa: E402

# 5 câu hỏi single-page, neo vào nội dung thật đã xác nhận bằng mắt (render
# PNG trực tiếp, xem PROGRESS.md mục M10) trên file 44 trang trong
# storage/uploads/0f70ce67a23e454c93d9bcf00edf5f45.pdf.
TEST_CASES = [
    ("Ticket CS ngân hàng cần tra cứu bao nhiêu hệ thống trước khi trả lời khách?", 26),
    ("Ma trận nào dùng để phân loại Rule/Workflow/LLM feature/Agent theo độ mơ hồ và độ phức tạp vận hành?", 20),
    ("Quy trình 6 giai đoạn phát triển AI Product gồm những bước nào?", 11),
    ("Trong mô hình stakeholder, ai là người approve và ai chỉ review?", 29),
    ("3 mức giải pháp Rule/Script, LLM Feature, và Agent khác nhau ở điểm nào?", 19),
]


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách dùng: eval_m3_single_page.py <doc_id>")
        sys.exit(1)
    doc_id = sys.argv[1]

    mode = os.environ.get("LLM_MODE", "mock").strip().lower()
    if mode != "gemini":
        print(f"CẢNH BÁO: LLM_MODE='{mode}', hybrid_search cần 'gemini' để có semantic thật.")

    page_index = read_page_index(doc_id)
    if not page_index:
        print(f"doc_id '{doc_id}' không có page_index — upload/reindex trước.")
        sys.exit(1)

    old_correct = 0
    new_correct = 0
    print(f"{'Câu hỏi':<70} {'Đúng':>5} {'CŨ':>6} {'MỚI':>6}")
    for question, expected_page in TEST_CASES:
        old_page = _keyword_search(question, page_index)
        result = hybrid_search(question, doc_id, page_index)
        new_page, confidence = result.page, result.confidence

        old_ok = old_page == expected_page
        new_ok = new_page == expected_page
        old_correct += old_ok
        new_correct += new_ok

        short_q = question[:66] + ("…" if len(question) > 66 else "")
        print(
            f"{short_q:<70} {expected_page:>5} "
            f"{str(old_page):>6}{'✓' if old_ok else '✗'} "
            f"{str(new_page):>6}{'✓' if new_ok else '✗'} (conf={confidence:.2f})"
        )

    n = len(TEST_CASES)
    print(f"\nAccuracy CŨ  (_keyword_search): {old_correct}/{n} = {old_correct/n*100:.0f}%")
    print(f"Accuracy MỚI (hybrid_search)  : {new_correct}/{n} = {new_correct/n*100:.0f}%")


if __name__ == "__main__":
    main()
