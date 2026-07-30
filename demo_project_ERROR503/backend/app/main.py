from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import SLIDES_DIR
from app.routers.chat import router as chat_router
from app.routers.docs import router as docs_router
from app.routers.upload import router as upload_router

app = FastAPI(title="AI Study Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/slides", StaticFiles(directory=str(SLIDES_DIR)), name="slides")

app.include_router(upload_router)
app.include_router(docs_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
