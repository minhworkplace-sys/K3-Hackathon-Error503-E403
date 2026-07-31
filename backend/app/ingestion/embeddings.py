"""M3: embedding thật qua Gemini Embedding API (`gemini-embedding-001`),
thay cho offline hashing tạm dùng ở M1. Vẫn giữ `OfflineHashingEmbeddingFunction`
cho `LLM_MODE=mock` — cùng cơ chế mock|gemini như `llm_provider.get_provider()`,
để dev/test luồng không cần API key/quota thật.

Bất đối xứng document/query (theo đúng khuyến nghị của Gemini Embedding API):
- Lúc index (`GeminiEmbeddingFunction.__call__`, dùng trong `collection.add()`):
  `task_type="RETRIEVAL_DOCUMENT"`
- Lúc query (`embed_query()`, gọi tay ở bước search — KHÔNG để ChromaDB tự
  embed query bằng cùng function trên): `task_type="RETRIEVAL_QUERY"`
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any

import numpy as np
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction, Space

from app.llm_provider import LLMProviderError

_DIM = 256
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


def _embed_one_hash(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = digest % dim
        sign = 1.0 if (digest // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class OfflineHashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Dùng cho LLM_MODE=mock — không gọi API thật, không cần key/quota."""

    def __init__(self, dim: int = _DIM):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [_embed_one_hash(text, self.dim) for text in input]

    @staticmethod
    def name() -> str:
        return "offline_hashing"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OfflineHashingEmbeddingFunction":
        return OfflineHashingEmbeddingFunction(dim=config.get("dim", _DIM))

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    def validate_config_update(self, old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
        return

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        return


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "Thiếu GOOGLE_API_KEY trong .env (cần cho Gemini Embedding API)"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import errors, types

    client = _get_gemini_client()
    config = types.EmbedContentConfig(task_type=task_type)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=_GEMINI_EMBEDDING_MODEL, contents=texts, config=config
            )
            return [e.values for e in response.embeddings]
        except errors.ClientError as exc:
            if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            raise LLMProviderError(f"Gemini Embedding API lỗi ({exc.code}): {exc.message}") from exc
        except errors.APIError as exc:
            raise LLMProviderError(f"Gemini Embedding API lỗi: {exc}") from exc


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Dùng cho LLM_MODE=gemini — embedding thật, task_type=RETRIEVAL_DOCUMENT."""

    def __call__(self, input: Documents) -> Embeddings:
        return _embed_texts(list(input), task_type="RETRIEVAL_DOCUMENT")

    @staticmethod
    def name() -> str:
        return "gemini_embedding_001"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction()

    def get_config(self) -> dict[str, Any]:
        return {}

    def validate_config_update(self, old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
        return

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        return


def get_embedding_function() -> EmbeddingFunction:
    mode = os.environ.get("LLM_MODE", "mock").strip().lower()
    if mode == "gemini":
        return GeminiEmbeddingFunction()
    return OfflineHashingEmbeddingFunction()


def embed_query(text: str) -> list[float]:
    """Embed 1 câu hỏi lúc search — task_type=RETRIEVAL_QUERY khi dùng Gemini
    thật (bất đối xứng có chủ ý với lúc index), hoặc dùng lại hash function
    khi mock (giữ cùng không gian vector với dữ liệu đã index bằng mock)."""
    mode = os.environ.get("LLM_MODE", "mock").strip().lower()
    if mode != "gemini":
        return _embed_one_hash(text, _DIM).tolist()
    return _embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_similarity(text: str) -> list[float]:
    """M11: embed 1 trang để so sánh độ tương đồng văn bản-với-văn bản (tìm
    trang liên quan) — task_type=SEMANTIC_SIMILARITY, khác RETRIEVAL_QUERY/
    RETRIEVAL_DOCUMENT (vốn dành cho query-tìm-document bất đối xứng, không
    hợp lý khi so sánh 2 trang ngang hàng với nhau)."""
    mode = os.environ.get("LLM_MODE", "mock").strip().lower()
    if mode != "gemini":
        return _embed_one_hash(text, _DIM).tolist()
    return _embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]
