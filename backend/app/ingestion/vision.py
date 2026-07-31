"""M10: heuristic lọc trang có ảnh/biểu đồ đáng kể TRƯỚC khi gọi Gemini
Vision — mục tiêu tối ưu chi phí (không gọi Vision tràn lan cho mọi trang).

Khác với đề xuất ban đầu (chỉ dùng `page.get_images()` + `page.get_text()`
để so tỉ lệ ảnh/text): thử nghiệm thực tế trên PDF thật trong project cho
thấy heuristic thuần image-ratio BỎ SÓT gần như toàn bộ biểu đồ/sơ đồ thật
trong slide (quadrant chart, flow diagram...) — vì các slide đó vẽ bằng
VECTOR SHAPES (box + arrow qua `page.get_drawings()`), không phải ảnh nhúng
(embedded raster image). Test 4 ngưỡng trên file thật (44 trang):
  A. img_ratio > 0.05                                   → 1/44 trang
     (chỉ bắt được trang bìa có ảnh nền full-page, bỏ sót 5/5 biểu đồ vector
     đã xác nhận bằng mắt — VD trang có sơ đồ "6 giai đoạn", "AI-Fit Matrix")
  B. n_drawings > 15                                     → 7/44 trang
     (bắt được toàn bộ 5 biểu đồ vector đã xác nhận, có lẫn 1-2 false
     positive như trang chỉ có danh sách đánh số + icon tròn trang trí)
  C. n_sig_shapes>=4 AND spread_ratio>0.3                → 33/44 trang
     (quá lỏng — bắt cả trang thuần văn bản có khung viền, không đạt mục
     tiêu tối ưu chi phí)
  D. (img_ratio > 0.05) OR (n_drawings > 15)              → 8/44 trang
     (= B, cộng thêm trang có ảnh nền full-page mà B bỏ sót)

→ Chọn **D**: kết hợp cả 2 tín hiệu (ảnh nhúng thật SỰ lớn — ảnh chụp/
screenshot dán vào slide — và vector shape dày đặc — biểu đồ/sơ đồ vẽ tay
trong công cụ trình chiếu), giữ tỉ lệ trang được xử lý thấp (~18% trên file
test) để tối ưu chi phí, đồng thời không bỏ sót loại biểu đồ phổ biến nhất
trong slide bài giảng thực tế.
"""

from __future__ import annotations

import logging

import fitz

from app.config import VISION_MIN_DRAWINGS, VISION_MIN_IMAGE_AREA_RATIO, VISION_RENDER_DPI
from app.ingestion.pdf_parser import ParsedPage
from app.llm_provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


def _page_image_ratio(page: fitz.Page) -> float:
    page_area = page.rect.width * page.rect.height
    if not page_area:
        return 0.0
    area = 0.0
    for img in page.get_images(full=True):
        try:
            for bbox in page.get_image_rects(img[0]):
                area += bbox.width * bbox.height
        except Exception:  # noqa: BLE001 - PyMuPDF có thể raise nhiều loại lỗi tuỳ file lỗi
            continue
    return area / page_area


def should_process_vision(page: fitz.Page) -> bool:
    """True nếu trang có ảnh nhúng đáng kể (ảnh nền, screenshot...) HOẶC
    nhiều vector shape (box/arrow của flow-chart, quadrant chart...)."""
    if _page_image_ratio(page) > VISION_MIN_IMAGE_AREA_RATIO:
        return True
    return len(page.get_drawings()) > VISION_MIN_DRAWINGS


def describe_pages_with_images(
    file_path: str, pages: list[ParsedPage], provider: LLMProvider
) -> tuple[list[ParsedPage], int]:
    """Với mỗi trang qua được heuristic, render CẢ TRANG (không phải ảnh cắt
    riêng) thành PNG, gọi provider.describe_image(), nối mô tả vào CUỐI
    ParsedPage.text (không thay thế text gốc đã extract).

    Lỗi Vision ở 1 trang không được làm hỏng cả upload — bỏ qua trang đó,
    giữ nguyên text gốc, tiếp tục các trang còn lại.

    Trả về (pages_mới, số_trang_đã_gọi_vision).
    """
    doc = fitz.open(file_path)
    descriptions: dict[int, str] = {}
    try:
        for page in doc:
            if not should_process_vision(page):
                continue
            page_number = page.number + 1
            pix = page.get_pixmap(dpi=VISION_RENDER_DPI)
            image_bytes = pix.tobytes("png")
            try:
                description = provider.describe_image(image_bytes, "image/png", max_tokens=512)
            except LLMProviderError as exc:
                logger.warning("Vision lỗi ở trang %d, bỏ qua augmentation: %s", page_number, exc)
                continue
            descriptions[page_number] = description.strip()
    finally:
        doc.close()

    if not descriptions:
        return pages, 0

    new_pages = [
        ParsedPage(
            page_number=p.page_number,
            text=f"{p.text}\n\n[Mô tả hình ảnh/biểu đồ]: {descriptions[p.page_number]}",
        )
        if p.page_number in descriptions
        else p
        for p in pages
    ]
    return new_pages, len(descriptions)
