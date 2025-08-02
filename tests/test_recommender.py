import pytest
import pandas as pd
import numpy as np
from src.recommender import build_interaction_matrix


def test_build_interaction_matrix_basic():
    # Beispiel-Daten
    df = pd.DataFrame({
        'user_id': ['U1', 'U1', 'U2', 'U3', 'U3'],
        'tool_id': ['T1', 'T3', 'T2', 'T1', 'T2'],
        'usage_count': [3, 1, 5, 2, 4]
    })
    
    R, user_ids, item_ids = build_interaction_matrix(df)
    # Erwartete IDs (sortiert nach Pivot-Index und Pivot-Spalten)
    assert user_ids == ['U1', 'U2', 'U3']
    assert item_ids == ['T1', 'T2', 'T3']

    # Erwartete Matrix (Zeilen: U1, U2, U3 | Spalten: T1, T2, T3)
    expected = np.array([
        [3, 0, 1],  # U1
        [0, 5, 0],  # U2
        [2, 4, 0]   # U3
    ])
    assert isinstance(R, np.ndarray)
    np.testing.assert_array_equal(R, expected)


def test_missing_columns_raises():
    df = pd.DataFrame({'foo': [1, 2, 3]})
    with pytest.raises(ValueError) as excinfo:
        build_interaction_matrix(df)
    assert 'missing required columns' in str(excinfo.value).lower()
