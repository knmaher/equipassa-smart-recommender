import pandas as pd
import numpy as np
import pytest
from src.recommender import build_interaction_matrix


def test_build_interaction_matrix_basic():
    df = pd.DataFrame([
        {"user_id": "U1", "tool_id": "T1", "usage_count": 2},
        {"user_id": "U2", "tool_id": "T2", "usage_count": 3},
    ])
    R, users, tools = build_interaction_matrix(df)
    assert isinstance(R, np.ndarray)
    assert users == ["U1","U2"]
    assert tools == ["T1","T2"]
    assert R.shape == (2,2)


def test_missing_columns_raises():
    df = pd.DataFrame([{"foo":1}])
    with pytest.raises(ValueError):
        build_interaction_matrix(df)
