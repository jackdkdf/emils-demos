"""Utility functions for data preprocessing."""

import pandas as pd
from typing import Optional


def normalize_team_name(name: str) -> str:
    """Normalize team name by lowercasing and replacing spaces with hyphens.
    
    Args:
        name: Team name to normalize.
        
    Returns:
        Normalized team name.
    """
    if pd.isna(name):
        return name
    return str(name).lower().strip().replace(' ', '-')


def normalize_map_name(name: str) -> str:
    """Normalize map name by lowercasing and replacing spaces with hyphens.
    
    Args:
        name: Map name to normalize.
        
    Returns:
        Normalized map name.
    """
    if pd.isna(name):
        return name
    return str(name).lower().strip().replace(' ', '-')

