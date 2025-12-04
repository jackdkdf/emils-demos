"""Create final features CSV with one-hot encoded map IDs."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_final_features(
    match_features_file: Path,
    output_file: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Create final features CSV with one-hot encoded map IDs.
    
    Args:
        match_features_file: Path to match_features.csv.
        output_file: Path to save final_features.csv.
        verbose: Whether to print progress messages.
        
    Returns:
        DataFrame with final features.
    """
    if verbose:
        logger.info("Loading match features...")
    
    df = pd.read_csv(match_features_file)
    
    if verbose:
        logger.info(f"Loaded {len(df)} matches")
        logger.info(f"Original columns: {len(df.columns)}")
    
    # Remove team_a_id, team_b_id, and match_date first
    columns_to_remove = ['team_a_id', 'team_b_id', 'match_date']
    df_cleaned = df.drop(columns=columns_to_remove)
    
    if verbose:
        logger.info(f"Removed columns: {columns_to_remove}")
    
    # One-hot encode map_id using pd.get_dummies (matches notebook)
    if verbose:
        logger.info("\nOne-hot encoding map_id...")
    
    map_dummies = pd.get_dummies(df_cleaned['map_id'], prefix='map_id', dtype=int)
    
    # Remove original map_id column
    df_without_map = df_cleaned.drop(columns=['map_id'])
    
    # Combine: map columns first (left side), then other columns (matches notebook)
    df_final = pd.concat([map_dummies, df_without_map], axis=1)
    
    map_onehot_cols = list(map_dummies.columns)
    other_cols = list(df_without_map.columns)
    
    if verbose:
        logger.info(f"Final columns: {len(df_final.columns)}")
        logger.info(f"  - Map one-hot columns: {len(map_onehot_cols)}")
        logger.info(f"  - Other features: {len(other_cols)}")
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    
    if verbose:
        logger.info(f"\nSaved to: {output_file}")
    
    return df_final
