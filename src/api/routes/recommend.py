from fastapi import APIRouter, Depends, HTTPException
import pandas as pd
from src.recommender import CosineRecommender, build_interaction_matrix
from src.api.settings import Settings
from src.api.models import RecommendationResponse, Recommendation

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.on_event("startup")
async def load_recommender(settings: Settings = Depends()):
    df = pd.read_csv(settings.data_path)
    R, user_ids, tool_ids = build_interaction_matrix(df)
    router.recommender = CosineRecommender(R, user_ids, tool_ids)


@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, k: int = 5):
    rec = getattr(router, "recommender", None)
    if rec is None:
        raise HTTPException(500, "Recommender not initialized")
    try:
        items = rec.recommend(user_id, top_k=k)
    except ValueError:
        raise HTTPException(404, f"User {user_id} not found")
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[Recommendation(tool_id=t, score=s) for t, s in items]
    )
