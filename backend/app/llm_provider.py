"""LLM abstraction layer.

Theo CLAUDE.md: KHÔNG import trực tiếp SDK của bất kỳ LLM provider nào
(google-genai, anthropic, openai...) rải rác trong code — mọi nơi khác
(routers, ...) chỉ được gọi qua LLMProvider. Chọn provider qua biến môi
trường LLM_MODE=mock|gemini (get_provider()), để đổi provider mà không
phải sửa logic ứng dụng.

M9: bỏ `answer`/`answer_stream` (format "1 câu + bullet" cũ) — locate_extract/
summarize_selection/summarize_progress giờ dùng chung `classify(prompt, schema)`
để lấy structured output dạng keyword-chip (xem chat.py, _KEYWORDS_SCHEMA).
`classify` vốn đã có sẵn trong interface gốc CLAUDE.md (dành cho classify_intent
ở M4) — dùng lại nguyên vẹn cho use case này vì bản chất đều là "ép Gemini trả
JSON đúng schema", không cần thêm method mới trong interface.

M10: thêm `describe_image` — method MỚI (ngoài 3 method gốc), vì mô tả ảnh
cần input dạng bytes+mime_type chứ không phải text như summarize/classify,
không gò được vào 2 method đó mà không làm interface rối. Dùng ở bước
ingest (app/ingestion/vision.py) để mô tả trang có biểu đồ/sơ đồ, chạy 1
lần lúc upload — không phải theo mỗi câu hỏi.

M11: `classify()` thêm tham số `model_tier` ("light" | "standard", mặc định
"standard") — KHÔNG lộ tên model cụ thể ra ngoài abstraction (đúng tinh
thần "LLM Provider bắt buộc qua abstraction layer" của CLAUDE.md, caller
chỉ nói "cần rẻ/nhanh" hay "cần chất lượng", không cần biết tên model
Gemini). "light" dùng cho classify_intent + grounding-check (tác vụ phân
loại/kiểm tra đơn giản, không cần model mạnh) — "standard" (mặc định,
không đổi behavior code cũ) cho keyword-chip generation (câu trả lời cuối
cùng user thấy trực tiếp, cần chất lượng). MockProvider bỏ qua model_tier
(không phân biệt, không tốn phí).
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

_GEMINI_FLASH_MODEL = "gemini-flash-latest"
# M11: model nhẹ hơn cho tác vụ phân loại/kiểm tra đơn giản (classify_intent,
# grounding-check) — đã probe thật, alias tồn tại và resolve ra
# gemini-3.5-flash-lite (cùng thế hệ với gemini-flash-latest -> 3.6-flash,
# chỉ nhẹ hơn). Không dùng cho câu trả lời cuối cùng user thấy trực tiếp.
_GEMINI_FLASH_LITE_MODEL = "gemini-flash-lite-latest"
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
# gemini-flash-latest tiêu tốn vài trăm token "thinking" nội bộ trước khi
# sinh answer visible (đo thực tế ~800 token ở chế độ thinking mặc định) —
# để thấp quá sẽ bị cắt cụt câu trả lời giữa chừng dù answer thật ngắn.
_DEFAULT_SUMMARIZE_MAX_TOKENS = 2048
# classify() (structured JSON, dùng cho keyword-chip) cũng tốn thinking budget
# tương tự — dùng chung mức ngân sách 2048 + thinking_level="low" như summarize,
# JSON output thực tế ngắn (3-6 chip) nên vẫn thừa dư địa an toàn.
_CLASSIFY_MAX_TOKENS = 2048
_CLASSIFY_THINKING_LEVEL = "low"

# M10: mô tả ảnh trang chỉ cần 2-3 câu ngắn — dùng chung mức thinking "low"
# như classify() để tiết kiệm chi phí (đây là bước ingest, chạy 1 lần/trang
# đủ điều kiện, nhưng vẫn nên rẻ vì có thể chạy nhiều trang trong 1 lần upload).
_VISION_MAX_TOKENS_DEFAULT = 512
_VISION_PROMPT = (
    "Đây là ảnh chụp 1 trang slide bài giảng. Nếu trang có biểu đồ/sơ đồ/hình "
    "minh hoạ mang thông tin, mô tả ngắn gọn (2-3 câu) nội dung mà hình đó "
    "truyền tải (VD: các bước trong quy trình, mối quan hệ giữa các thành "
    "phần, trục và vùng trong biểu đồ, số liệu). Nếu trang chỉ có văn bản "
    "hoặc hình chỉ mang tính trang trí (logo, icon nhỏ), trả lời đúng một "
    'câu: "Không có hình ảnh/biểu đồ đáng kể."'
)


class LLMProviderError(Exception):
    """Lỗi từ LLM provider (rate limit, timeout, thiếu key...). Router bắt
    exception này để trả lỗi thân thiện, không để crash trắng màn hình."""


class LLMProvider(ABC):
    @abstractmethod
    def summarize(self, text: str, max_tokens: int) -> str: ...

    @abstractmethod
    def summarize_stream(self, text: str) -> Iterator[str]: ...

    @abstractmethod
    def classify(self, prompt: str, schema: dict, model_tier: str = "standard") -> dict: ...

    @abstractmethod
    def describe_image(self, image_bytes: bytes, mime_type: str, max_tokens: int) -> str: ...


class MockProvider(LLMProvider):
    """Tái hiện đúng behavior prototype cũ: template response cố định,
    không gọi API thật. Dùng khi build UI/luồng hoặc không có API key."""

    def summarize(self, text: str, max_tokens: int) -> str:
        return "".join(self.summarize_stream(text))

    def summarize_stream(self, text: str) -> Iterator[str]:
        short = text[:80] + ("..." if len(text) > 80 else "")
        yield (
            f'[Demo] Tóm tắt đoạn bạn bôi đen: "{short}" — đây là câu trả lời '
            "giả lập minh hoạ chức năng, chưa dùng LLM thật (sẽ làm ở M4)."
        )

    def classify(self, prompt: str, schema: dict, model_tier: str = "standard") -> dict:
        # model_tier bị bỏ qua — mock không tốn phí, không cần phân biệt.
        # Dispatch theo shape của schema — mock đủ cho cả 3 use case đã có:
        # keywords (M9), intent (M11 classify_intent), grounded (M11 guardrail).
        properties = schema.get("properties", {})
        if "intent" in properties:
            return {"intent": "single_page"}  # mặc định an toàn cho test wiring
        if "grounded" in properties:
            return {"grounded": True}  # mặc định "hợp lệ" cho test wiring
        if "keywords" not in properties:
            raise NotImplementedError("MockProvider.classify: schema lạ, chưa hỗ trợ")

        tail = prompt.strip()[-400:]
        parts = [p.strip() for p in re.split(r"[.\n;]", tail) if len(p.strip()) > 3][-4:]
        keywords = [
            {"text": f"[Demo] {p[:40]}", "category": "none", "connector": False} for p in parts
        ] or [{"text": "[Demo] Không có nội dung để tóm tắt", "category": "none", "connector": False}]
        return {"keywords": keywords}

    def describe_image(self, image_bytes: bytes, mime_type: str, max_tokens: int) -> str:
        return f"[Demo] Mô tả hình ảnh trang ({len(image_bytes)} bytes, mime={mime_type}) — chưa gọi Vision thật."


class GeminiProvider(LLMProvider):
    """Provider thật, gọi Google Gemini qua SDK `google-genai`."""

    def __init__(self) -> None:
        from google import genai  # import cục bộ: SDK không rò rỉ ra module khác

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "Thiếu GOOGLE_API_KEY trong .env (lấy key tại aistudio.google.com)"
            )
        self._client = genai.Client(api_key=api_key)

    def summarize(self, text: str, max_tokens: int) -> str:
        return "".join(self._generate_stream(self._summarize_prompt(text), max_tokens=max_tokens))

    def summarize_stream(self, text: str) -> Iterator[str]:
        yield from self._generate_stream(
            self._summarize_prompt(text), max_tokens=_DEFAULT_SUMMARIZE_MAX_TOKENS
        )

    def classify(self, prompt: str, schema: dict, model_tier: str = "standard") -> dict:
        from google.genai import errors, types

        model = _GEMINI_FLASH_LITE_MODEL if model_tier == "light" else _GEMINI_FLASH_MODEL
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=_CLASSIFY_MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=_CLASSIFY_THINKING_LEVEL),
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                if not response.text:
                    raise LLMProviderError("Gemini trả về rỗng, không có dữ liệu JSON")
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError as exc:
                    raise LLMProviderError(f"Gemini trả JSON không hợp lệ: {exc}") from exc
            except errors.ClientError as exc:
                if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                    continue
                raise LLMProviderError(f"Gemini API lỗi ({exc.code}): {exc.message}") from exc
            except errors.APIError as exc:
                raise LLMProviderError(f"Gemini API lỗi: {exc}") from exc

    def describe_image(self, image_bytes: bytes, mime_type: str, max_tokens: int) -> str:
        from google.genai import errors, types

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=_CLASSIFY_THINKING_LEVEL),
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=_GEMINI_FLASH_MODEL,
                    contents=[image_part, _VISION_PROMPT],
                    config=config,
                )
                return response.text or ""
            except errors.ClientError as exc:
                if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                    continue
                raise LLMProviderError(f"Gemini API lỗi ({exc.code}): {exc.message}") from exc
            except errors.APIError as exc:
                raise LLMProviderError(f"Gemini API lỗi: {exc}") from exc

    @staticmethod
    def _summarize_prompt(text: str) -> str:
        return (
            "Tóm tắt đoạn văn bản sau bằng tiếng Việt, ngôn ngữ đơn giản dễ "
            "hiểu, trong đúng 2-3 câu:\n\n" + text
        )

    def _generate_stream(
        self, prompt: str, max_tokens: int, thinking_level: str | None = None
    ) -> Iterator[str]:
        from google.genai import errors, types

        thinking_config = (
            types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
        )
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens, thinking_config=thinking_config
        )

        for attempt in range(_MAX_RETRIES):
            try:
                stream = self._client.models.generate_content_stream(
                    model=_GEMINI_FLASH_MODEL,
                    contents=prompt,
                    config=config,
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                return
            except errors.ClientError as exc:
                if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                    continue
                raise LLMProviderError(f"Gemini API lỗi ({exc.code}): {exc.message}") from exc
            except errors.APIError as exc:
                raise LLMProviderError(f"Gemini API lỗi: {exc}") from exc


def get_provider() -> LLMProvider:
    mode = os.environ.get("LLM_MODE", "mock").strip().lower()
    if mode == "gemini":
        return GeminiProvider()
    return MockProvider()
