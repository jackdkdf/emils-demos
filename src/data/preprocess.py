"""Main preprocessing pipeline orchestrator."""

import logging
from pathlib import Path
from typing import Optional

from .team_mapping import create_team_mapping
from .map_mapping import create_map_mapping
from .cumulative_stats import calculate_opponent_cumulative_stats, calculate_map_cumulative_stats
from .match_features import create_match_features

logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> None:
    """Run the complete preprocessing pipeline.
    
    Args:
        project_root: Root directory of the project. If None, uses current directory's parent.
        verbose: Whether to print progress messages.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    # Define paths
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    mappings_dir = data_dir / "mappings"
    team_results_dir = raw_dir / "team_results"
    player_results_dir = raw_dir / "player_results"
    rankings_dir = raw_dir / "rankings"
    
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        logger.info("PREPROCESSING PIPELINE")
    
    # Step 1: Create team name to ID mapping
    if verbose:
        logger.info("\n[Step 1/5] Creating team name to ID mapping...")
    create_team_mapping(
        teams_peak_file=rankings_dir / "teams_peak_36.csv",
        output_file=mappings_dir / "team_name_to_id.csv",
        verbose=verbose
    )
    
    # Step 2: Create map name to ID mapping
    if verbose:
        logger.info("\n[Step 2/5] Creating map name to ID mapping...")
    create_map_mapping(
        team_results_dir=team_results_dir,
        output_file=mappings_dir / "map_name_to_id.csv",
        num_files=10,
        verbose=verbose
    )
    
    # Step 3: Calculate cumulative opponent statistics
    if verbose:
        logger.info("\n[Step 3/5] Calculating cumulative opponent statistics...")
    calculate_opponent_cumulative_stats(
        team_results_dir=team_results_dir,
        output_file=preprocessed_dir / "team_opponent_cumulative_stats.csv",
        verbose=verbose
    )
    
    # Step 4: Calculate cumulative map statistics
    if verbose:
        logger.info("\n[Step 4/5] Calculating cumulative map statistics...")
    calculate_map_cumulative_stats(
        team_results_dir=team_results_dir,
        output_file=preprocessed_dir / "team_map_cumulative_stats.csv",
        verbose=verbose
    )
    
    # Step 5: Create match features
    if verbose:
        logger.info("\n[Step 5/5] Creating match features...")
    create_match_features(
        team_results_dir=team_results_dir,
        opponent_stats_file=preprocessed_dir / "team_opponent_cumulative_stats.csv",
        map_stats_file=preprocessed_dir / "team_map_cumulative_stats.csv",
        rankings_file=rankings_dir / "hltv_team_rankings_original.csv",
        team_name_to_id_file=mappings_dir / "team_name_to_id.csv",
        player_results_dir=player_results_dir if player_results_dir.exists() else None,
        output_file=preprocessed_dir / "match_features.csv",
        verbose=verbose
    )
    
    if verbose:
        logger.info("\nPREPROCESSING PIPELINE COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_preprocessing_pipeline()
