from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.recommend import router as recommend_router
from src.api.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title="Equipassa Smart Recommender",
        version="0.1.0",
        description="User-based CF mit Cosine Similarity"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(recommend_router)
    return app


app = create_app()
