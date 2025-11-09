"""Windows-friendly pandas helper utilities.

This module contains small helper functions to load and save CSVs robustly on Windows.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional


def read_csv_df(path: str, *, encoding: Optional[str] = None) -> pd.DataFrame:
    """Read CSV into DataFrame using a reasonable set of defaults.

    Args:
        path: Path to CSV file.
        encoding: Optional encoding override (default: let pandas infer).

    Returns:
        pd.DataFrame
    """
    kwargs = {"low_memory": False}
    if encoding:
        kwargs["encoding"] = encoding
    return pd.read_csv(path, **kwargs)


def save_df(df: pd.DataFrame, path: str, *, index: bool = False) -> None:
    """Save DataFrame to CSV.

    Args:
        df: DataFrame to save
        path: Destination path
        index: Whether to write row indices
    """
    df.to_csv(path, index=index)
