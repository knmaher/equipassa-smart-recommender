"""
Recommender-Modul: baut die Interaktionsmatrix und liefert Cosine-CF-Empfehlungen.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_interaction_matrix(
    df: pd.DataFrame
) -> Tuple[np.ndarray, List[str], List[str]]:
    required = {"user_id", "tool_id", "usage_count"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")
    R_df = df.pivot_table(
        index="user_id", columns="tool_id",
        values="usage_count", fill_value=0
    )
    return R_df.values, list(R_df.index), list(R_df.columns)


class CosineRecommender:
    def __init__(
        self,
        R: np.ndarray,
        user_ids: List[str],
        tool_ids: List[str]
    ):
        self.R = R
        self.user_ids = user_ids
        self.tool_ids = tool_ids
        self.sim = cosine_similarity(R)

    def recommend(
        self,
        user_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if user_id not in self.user_ids:
            raise ValueError(f"Unknown user_id {user_id}")
        u = self.user_ids.index(user_id)
        scores = self.sim[u].copy()
        scores[u] = 0
        scores = scores @ self.R
        used = self.R[u] > 0
        scores[used] = -1
        idx = np.argsort(-scores)[:top_k]
        return [(self.tool_ids[i], float(scores[i])) for i in idx]
