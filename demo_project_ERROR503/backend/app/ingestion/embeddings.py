"""Embedding function offline (feature hashing), không cần tải model qua
mạng. Dùng tạm cho M1 vì mục tiêu M1 là verify ingestion pipeline (page
number, chunk, dual storage) — chất lượng semantic search thực sự chỉ cần
chính xác từ M3. Có thể thay bằng model tốt hơn (vd. sentence-transformers)
sau này mà không phải đổi lại code index đã viết, miễn giữ interface
EmbeddingFunction của chromadb.
"""

import hashlib
import re
from typing import Any

import numpy as np
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction, Space

_DIM = 256
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _embed_one(text: str, dim: int) -> np.ndarray:
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
    def __init__(self, dim: int = _DIM):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [_embed_one(text, self.dim) for text in input]

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
