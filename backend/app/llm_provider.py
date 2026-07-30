"""LLM abstraction layer.

Theo CLAUDE.md: KHÔNG import trực tiếp SDK của bất kỳ LLM provider nào
(google-genai, anthropic, openai...) rải rác trong code — mọi nơi khác
(routers, ...) chỉ được gọi qua LLMProvider. Chọn provider qua biến môi
trường LLM_MODE=mock|gemini (get_provider()), để đổi provider mà không
phải sửa logic ứng dụng.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

_GEMINI_FLASH_MODEL = "gemini-flash-latest"
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
# gemini-flash-latest tiêu tốn vài trăm token "thinking" nội bộ trước khi
# sinh answer visible (đo thực tế ~800 token) — để thấp quá sẽ bị cắt cụt
# câu trả lời giữa chừng dù answer thật ngắn.
_DEFAULT_SUMMARIZE_MAX_TOKENS = 2048


class LLMProviderError(Exception):
    """Lỗi từ LLM provider (rate limit, timeout, thiếu key...). Router bắt
    exception này để trả lỗi thân thiện, không để crash trắng màn hình."""


class LLMProvider(ABC):
    @abstractmethod
    def summarize(self, text: str, max_tokens: int) -> str: ...

    @abstractmethod
    def summarize_stream(self, text: str) -> Iterator[str]: ...

    @abstractmethod
    def classify(self, prompt: str, schema: dict) -> dict: ...


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

    def classify(self, prompt: str, schema: dict) -> dict:
        raise NotImplementedError("MockProvider.classify sẽ implement ở M4 (intent router)")


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
        return "".join(self._generate_stream(text, max_tokens=max_tokens))

    def summarize_stream(self, text: str) -> Iterator[str]:
        yield from self._generate_stream(text, max_tokens=_DEFAULT_SUMMARIZE_MAX_TOKENS)

    def classify(self, prompt: str, schema: dict) -> dict:
        raise NotImplementedError("GeminiProvider.classify sẽ implement ở M4 (intent router)")

    def _generate_stream(self, text: str, max_tokens: int) -> Iterator[str]:
        from google.genai import errors, types

        prompt = (
            "Tóm tắt đoạn văn bản sau bằng tiếng Việt, ngôn ngữ đơn giản dễ "
            "hiểu, trong đúng 2-3 câu:\n\n" + text
        )
        config = types.GenerateContentConfig(max_output_tokens=max_tokens)

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
