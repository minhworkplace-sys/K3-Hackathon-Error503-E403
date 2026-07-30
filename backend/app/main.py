from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.chat import router as chat_router
from app.routers.docs import router as docs_router
from app.routers.upload import router as upload_router

app = FastAPI(title="AI Study Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(docs_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
