import pandas as pd
import numpy as np
from typing import Tuple, List


def build_interaction_matrix(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "tool_id",
    value_col: str = "usage_count"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Builds a user-item interaction matrix from a long-form DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with user-item interactions.
        user_col (str): Column name for user identifiers. Default: 'user_id'.
        item_col (str): Column name for item (tool) identifiers. Default: 'tool_id'.
        value_col (str): Column name for interaction values. Default: 'usage_count'.

    Returns:
        R (np.ndarray): 2D array of shape (n_users, n_items). Rows correspond to users, columns to items.
        user_ids (List[str]): Ordered list of user identifiers for each row.
        item_ids (List[str]): Ordered list of item identifiers for each column.

    Raises:
        ValueError: If the DataFrame is missing any of the required columns.
    """
    # Überprüfen, ob alle Spalten vorhanden sind
    required = {user_col, item_col, value_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Pivotieren der Daten
    interaction_df = df.pivot_table(
        index=user_col,
        columns=item_col,
        values=value_col,
        fill_value=0
    )

    # IDs und Matrix extrahieren
    user_ids = interaction_df.index.tolist()
    item_ids = interaction_df.columns.tolist()
    R = interaction_df.to_numpy()

    return R, user_ids, item_ids