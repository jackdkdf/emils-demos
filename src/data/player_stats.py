"""Player statistics handling."""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional

from .utils import normalize_team_name, normalize_map_name

logger = logging.getLogger(__name__)


def load_player_stats(player_results_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Load all player stats files.
    
    Args:
        player_results_dir: Directory containing player weekly stats CSV files.
        verbose: Whether to print progress messages.
        
    Returns:
        Combined DataFrame with all player stats.
    """
    player_stats_files = sorted(player_results_dir.glob("*_weekly_stats.csv"))
    
    all_player_stats = []
    for file_path in player_stats_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                df['team_name'] = df['team_name'].apply(normalize_team_name)
                df['map_name'] = df['map_name'].apply(normalize_map_name)
                df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
                df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
                df = df.dropna(subset=['start_date', 'end_date'])
                all_player_stats.append(df)
        except Exception as e:
            if verbose:
                logger.warning(f"  Error loading {file_path.name}: {e}")
    
    if all_player_stats:
        player_stats_df = pd.concat(all_player_stats, ignore_index=True)
        player_stats_df = player_stats_df.sort_values(['team_name', 'map_name', 'start_date']).reset_index(drop=True)
        
        if verbose:
            logger.info(f"Loaded {len(player_stats_df)} player stat rows")
            logger.info(f"Unique teams in player stats: {player_stats_df['team_name'].nunique()}")
        
        return player_stats_df
    else:
        if verbose:
            logger.warning("No player stats files found")
        return pd.DataFrame()


def get_player_stats(
    team_name: str,
    map_name: str,
    match_date: pd.Timestamp,
    player_stats_df: pd.DataFrame,
    max_players: int = 5
) -> List[float]:
    """Get player stats for a team on a map from the week BEFORE the match date.
    
    Args:
        team_name: Normalized team name.
        map_name: Normalized map name.
        match_date: Date of the match.
        player_stats_df: DataFrame with player statistics.
        max_players: Maximum number of players to include.
        
    Returns:
        List of features: [player1_overall, player1_utility, player1_opening, ...]
        Returns zeros if no stats available.
    """
    if len(player_stats_df) == 0:
        return [0.0] * (max_players * 3)
    
    team_map_stats = player_stats_df[
        (player_stats_df['team_name'] == team_name) &
        (player_stats_df['map_name'] == map_name)
    ]
    
    if len(team_map_stats) == 0:
        return [0.0] * (max_players * 3)
    
    # Get stats from weeks BEFORE the match date (end_date < match_date)
    before_match = team_map_stats[team_map_stats['end_date'] < match_date]
    
    if len(before_match) == 0:
        return [0.0] * (max_players * 3)
    
    # Get the most recent week (latest end_date)
    latest_end_date = before_match['end_date'].max()
    week_stats = before_match[before_match['end_date'] == latest_end_date]
    
    if len(week_stats) == 0:
        return [0.0] * (max_players * 3)
    
    # Sort by overall_rating descending to get top players
    week_stats_sorted = week_stats.sort_values('overall_rating', ascending=False).head(max_players)
    
    # Extract features: [player1_overall, player1_utility, player1_opening, player2_overall, ...]
    features = []
    for _, player_row in week_stats_sorted.iterrows():
        features.append(float(player_row['overall_rating']) if pd.notna(player_row['overall_rating']) else 0.0)
        features.append(float(player_row['utility_success']) if pd.notna(player_row['utility_success']) else 0.0)
        features.append(float(player_row['opening_rating']) if pd.notna(player_row['opening_rating']) else 0.0)
    
    # Pad with zeros if we have fewer than max_players
    while len(features) < max_players * 3:
        features.extend([0.0, 0.0, 0.0])
    
    return features[:max_players * 3]
