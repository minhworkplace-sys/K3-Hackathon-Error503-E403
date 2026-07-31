"""Ép LLM_MODE=mock cho toàn bộ test suite — tests không được phụ thuộc
network/API key/quota thật. Phải set TRƯỚC khi bất kỳ module `app.*` nào
được import (app.config gọi load_dotenv() với override=False mặc định,
nên nếu LLM_MODE đã có trong os.environ từ đây thì .env sẽ không ghi đè).

Phát hiện lúc làm M3: trước đây .env có LLM_MODE=gemini leak vào tiến
trình pytest qua load_dotenv() (không ai set LLM_MODE=mock tường minh),
nhưng không lộ ra vì code cũ chưa có chỗ nào gọi API thật trong test. M3
thêm get_embedding_function() đọc LLM_MODE thật thì lộ ngay — test cố gọi
Gemini Embedding API thật.
"""

import os

os.environ.setdefault("LLM_MODE", "mock")
# Collection riêng cho test — collection thật ("slides") đã bind embedding
# function thật (gemini_embedding_001), khác config với offline_hashing mà
# test dùng — dùng chung sẽ bị ChromaDB từ chối (embedding function conflict).
os.environ.setdefault("CHROMA_COLLECTION_NAME", "slides_test")
