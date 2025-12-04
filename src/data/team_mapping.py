"""Create team name to ID mapping."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from .utils import normalize_team_name

logger = logging.getLogger(__name__)


def create_team_mapping(
    teams_peak_file: Path,
    output_file: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Create team name to ID mapping from teams_peak_36.csv.
    
    Args:
        teams_peak_file: Path to teams_peak_36.csv file.
        output_file: Path to save the mapping CSV.
        verbose: Whether to print progress messages.
        
    Returns:
        DataFrame with team_name and team_id columns.
    """
    # Load the teams_peak_36.csv file
    df = pd.read_csv(teams_peak_file)
    
    # Create mapping dataframe with normalized team names
    team_name_to_id = pd.DataFrame({
        'team_name': df['name'].apply(normalize_team_name),
        'team_id': df['team_id']
    })
    
    # Remove any duplicate team names (if any) - keep first occurrence
    team_name_to_id = team_name_to_id.drop_duplicates(subset='team_name', keep='first')
    
    # Sort by team_id for consistency
    team_name_to_id = team_name_to_id.sort_values('team_id').reset_index(drop=True)
    
    # Save to CSV
    team_name_to_id.to_csv(output_file, index=False)
    
    if verbose:
        logger.info(f"Created team name to ID mapping with {len(team_name_to_id)} teams")
        logger.info(f"Saved to: {output_file}")
        logger.info("\nFirst few rows:")
        logger.info(f"\n{team_name_to_id.head(10)}")
    
    return team_name_to_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    teams_peak_file = project_root / "data" / "raw" / "rankings" / "teams_peak_36.csv"
    output_file = project_root / "data" / "mappings" / "team_name_to_id.csv"
    create_team_mapping(teams_peak_file, output_file)
