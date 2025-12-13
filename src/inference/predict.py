"""Inference module for predicting match outcomes."""

import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

from ..data.match_features import (
    normalize_team_name,
    normalize_map_name,
    get_player_stats,
    calculate_streak,
)
from .historical_data import fetch_historical_data_for_match

logger = logging.getLogger(__name__)


def create_features_from_historical_data(
    team_a: str,
    team_b: str,
    map_name: str,
    match_date: str,
    historical_data: Dict,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """Create features from fetched historical data.
    
    Args:
        team_a: Name of team A
        team_b: Name of team B
        map_name: Map name
        match_date: Match date
        historical_data: Dictionary with fetched historical data
        verbose: Whether to print progress
        
    Returns:
        DataFrame with features
    """
    team_a_norm = normalize_team_name(team_a)
    team_b_norm = normalize_team_name(team_b)
    map_name_norm = normalize_map_name(map_name)
    match_date_dt = pd.to_datetime(match_date)
    
    team_results = historical_data['team_results']
    head_to_head = historical_data['head_to_head']
    team_a_id = int(historical_data['team_a_id'])
    team_b_id = int(historical_data['team_b_id'])
    map_id = int(historical_data['map_id'])
    
    # Convert to DataFrame - empty is OK, we'll use default values
    if not team_results:
        if verbose:
            logger.debug(f"No historical matches found for {team_a} vs {team_b} on {map_name}, using default features")
        # Create empty DataFrame with correct structure
        df = pd.DataFrame(columns=["team_name", "team_id", "map_name", "map_id", "match_date", 
                                   "opponent_name", "opponent_id", "score_us", "score_them", "result"])
    else:
        df = pd.DataFrame(team_results)
        df['match_date'] = pd.to_datetime(df['match_date'])
        df = df[df['match_date'] < match_date_dt].sort_values('match_date')
    
    # Compute cumulative stats
    # Team A vs Team B
    h2h_df = pd.DataFrame(head_to_head)
    if len(h2h_df) > 0:
        h2h_df['match_date'] = pd.to_datetime(h2h_df['match_date'])
        h2h_df = h2h_df[h2h_df['match_date'] < match_date_dt].sort_values('match_date')
        team_a_wins_vs_b = len(h2h_df[h2h_df['result'] == 'W'])
        team_a_losses_vs_b = len(h2h_df[h2h_df['result'] == 'L'])
        team_a_total_vs_b = team_a_wins_vs_b + team_a_losses_vs_b
        team_a_winrate_vs_b = (team_a_wins_vs_b / team_a_total_vs_b * 100) if team_a_total_vs_b > 0 else 0.0
    else:
        team_a_wins_vs_b = 0
        team_a_losses_vs_b = 0
        team_a_total_vs_b = 0
        team_a_winrate_vs_b = 0.0
    
    # Team A map stats
    team_a_map_matches = df[(df['team_id'] == str(team_a_id)) & (df['map_name'] == map_name_norm)]
    if len(team_a_map_matches) > 0:
        team_a_map_wins = len(team_a_map_matches[team_a_map_matches['result'] == 'W'])
        team_a_map_losses = len(team_a_map_matches[team_a_map_matches['result'] == 'L'])
        team_a_map_total = team_a_map_wins + team_a_map_losses
        team_a_map_winrate = (team_a_map_wins / team_a_map_total * 100) if team_a_map_total > 0 else 0.0
    else:
        team_a_map_wins = 0
        team_a_map_losses = 0
        team_a_map_total = 0
        team_a_map_winrate = 0.0
    
    # Team B map stats
    team_b_map_matches = df[(df['team_id'] == str(team_b_id)) & (df['map_name'] == map_name_norm)]
    if len(team_b_map_matches) > 0:
        team_b_map_wins = len(team_b_map_matches[team_b_map_matches['result'] == 'W'])
        team_b_map_losses = len(team_b_map_matches[team_b_map_matches['result'] == 'L'])
        team_b_map_total = team_b_map_wins + team_b_map_losses
        team_b_map_winrate = (team_b_map_wins / team_b_map_total * 100) if team_b_map_total > 0 else 0.0
    else:
        team_b_map_wins = 0
        team_b_map_losses = 0
        team_b_map_total = 0
        team_b_map_winrate = 0.0
    
    # Calculate streaks
    team_a_matches = df[df['team_id'] == str(team_a_id)].sort_values('match_date')
    team_b_matches = df[df['team_id'] == str(team_b_id)].sort_values('match_date')
    
    team_match_history = {}
    for _, row in team_a_matches.iterrows():
        if team_a_id not in team_match_history:
            team_match_history[team_a_id] = []
        won = 1 if row['result'] == 'W' else 0
        team_match_history[team_a_id].append((row['match_date'], won))
    
    for _, row in team_b_matches.iterrows():
        if team_b_id not in team_match_history:
            team_match_history[team_b_id] = []
        won = 1 if row['result'] == 'W' else 0
        team_match_history[team_b_id].append((row['match_date'], won))
    
    team_a_win_streak, team_a_loss_streak = calculate_streak(team_a_id, team_match_history)
    team_b_win_streak, team_b_loss_streak = calculate_streak(team_b_id, team_match_history)
    
    # Rankings (set to 0 if not available)
    team_a_global_ranking = 0
    team_b_global_ranking = 0
    
    # Player stats (set to zeros if not available)
    team_a_player_features = [0.0] * 15
    team_b_player_features = [0.0] * 15
    
    # Create feature row
    feature_row = {
        'team_a_id': team_a_id,
        'team_b_id': team_b_id,
        'map_id': map_id,
        'match_date': match_date_dt,
        'team_a_wins_vs_b': team_a_wins_vs_b,
        'team_a_losses_vs_b': team_a_losses_vs_b,
        'team_a_total_vs_b': team_a_total_vs_b,
        'team_a_winrate_vs_b': round(team_a_winrate_vs_b, 2),
        'team_a_map_wins': team_a_map_wins,
        'team_a_map_losses': team_a_map_losses,
        'team_a_map_total': team_a_map_total,
        'team_a_map_winrate': round(team_a_map_winrate, 2),
        'team_b_map_wins': team_b_map_wins,
        'team_b_map_losses': team_b_map_losses,
        'team_b_map_total': team_b_map_total,
        'team_b_map_winrate': round(team_b_map_winrate, 2),
        'team_a_global_ranking': team_a_global_ranking,
        'team_b_global_ranking': team_b_global_ranking,
        'team_a_win_streak': team_a_win_streak,
        'team_a_loss_streak': team_a_loss_streak,
        'team_b_win_streak': team_b_win_streak,
        'team_b_loss_streak': team_b_loss_streak,
    }
    
    # Add player features
    for i in range(5):
        feature_row[f'team_a_player_{i+1}_overall_rating'] = team_a_player_features[i*3]
        feature_row[f'team_a_player_{i+1}_utility_success'] = team_a_player_features[i*3+1]
        feature_row[f'team_a_player_{i+1}_opening_rating'] = team_a_player_features[i*3+2]
        feature_row[f'team_b_player_{i+1}_overall_rating'] = team_b_player_features[i*3]
        feature_row[f'team_b_player_{i+1}_utility_success'] = team_b_player_features[i*3+1]
        feature_row[f'team_b_player_{i+1}_opening_rating'] = team_b_player_features[i*3+2]
    
    return pd.DataFrame([feature_row])


def create_match_features_for_inference(
    team_a: str,
    team_b: str,
    map_name: str,
    match_date: str,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """Create features for a single match for inference.
    
    This function computes features for a specific match (identified by team_a, team_b, map, date)
    by querying historical data up to (but not including) the match date. The match details come
    from the URL, but features are computed from historical statistics.
    
    Note: Assumes preprocessing has already been completed. Required files:
    - data/preprocessed/team_opponent_cumulative_stats.csv (historical team vs team stats)
    - data/preprocessed/team_map_cumulative_stats.csv (historical team map stats)
    - data/mappings/team_name_to_id.csv (team name mappings)
    - data/raw/rankings/hltv_team_rankings.csv (historical rankings, optional)
    - data/raw/team_results/*.csv (historical match results for streak calculation)
    - data/raw/player_results/*.csv (historical player stats, optional)
    
    Args:
        team_a: Name of team A (from URL)
        team_b: Name of team B (from URL)
        map_name: Name of the map (from URL)
        match_date: Match date in YYYY-MM-DD format (from URL)
        project_root: Root directory of the project
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with features for the match, or None if data is insufficient
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    # Normalize names
    team_a_norm = normalize_team_name(team_a)
    team_b_norm = normalize_team_name(team_b)
    map_name_norm = normalize_map_name(map_name)
    match_date_dt = pd.to_datetime(match_date)
    
    # Load required data files
    data_dir = project_root / "data"
    preprocessed_dir = data_dir / "preprocessed"
    mappings_dir = data_dir / "mappings"
    raw_dir = data_dir / "raw"
    team_results_dir = raw_dir / "team_results"
    player_results_dir = raw_dir / "player_results"
    
    if verbose:
        logger.info("Loading preprocessed data files...")
    
    # Load preprocessed files (assume preprocessing is already done)
    opponent_stats_file = preprocessed_dir / "team_opponent_cumulative_stats.csv"
    map_stats_file = preprocessed_dir / "team_map_cumulative_stats.csv"
    rankings_file = raw_dir / "rankings" / "hltv_team_rankings.csv"
    team_name_to_id_file = mappings_dir / "team_name_to_id.csv"
    
    # Check that required files exist
    required_files = {
        'opponent_stats': opponent_stats_file,
        'map_stats': map_stats_file,
        'team_mapping': team_name_to_id_file,
    }
    
    missing_files = [name for name, path in required_files.items() if not path.exists()]
    if missing_files:
        logger.error(f"Required preprocessed files not found: {', '.join(missing_files)}")
        logger.error("Please run preprocessing first: python -m src.main preprocess")
        return None
    
    opponent_stats = pd.read_csv(opponent_stats_file)
    map_stats = pd.read_csv(map_stats_file)
    team_name_to_id_df = pd.read_csv(team_name_to_id_file)
    
    # Load rankings (optional - will use 0 if not found)
    if rankings_file.exists():
        rankings_df = pd.read_csv(rankings_file)
    else:
        if verbose:
            logger.warning(f"Rankings file not found: {rankings_file}. Rankings will be set to 0.")
        rankings_df = pd.DataFrame(columns=['name', 'rank', 'points', 'date', 'team_id', 'team_slug'])
    
    # Convert dates
    opponent_stats['match_date'] = pd.to_datetime(opponent_stats['match_date'])
    map_stats['match_date'] = pd.to_datetime(map_stats['match_date'])
    
    # Normalize names
    opponent_stats['team_name'] = opponent_stats['team_name'].apply(normalize_team_name)
    opponent_stats['opponent_name'] = opponent_stats['opponent_name'].apply(normalize_team_name)
    map_stats['team_name'] = map_stats['team_name'].apply(normalize_team_name)
    map_stats['map_name'] = map_stats['map_name'].apply(normalize_map_name)
    
    # Merge rankings with team IDs (if rankings exist)
    if len(rankings_df) > 0:
        rankings_df['date'] = pd.to_datetime(rankings_df['date'])
        rankings_df['name_normalized'] = rankings_df['name'].apply(normalize_team_name)
        rankings_df = rankings_df.merge(
            team_name_to_id_df,
            left_on='name_normalized',
            right_on='team_name',
            how='left'
        )
        rankings_df = rankings_df.dropna(subset=['team_id'])
        if 'rank' in rankings_df.columns:
            rankings_df['rank'] = rankings_df['rank'].astype(str).str.replace('#', '').astype(int)
    else:
        # Create empty rankings with required columns
        rankings_df = pd.DataFrame(columns=['team_id', 'date', 'points', 'rank'])
        rankings_df['date'] = pd.to_datetime([])
    
    # Get team IDs
    team_a_id_row = team_name_to_id_df[team_name_to_id_df['team_name'] == team_a_norm]
    team_b_id_row = team_name_to_id_df[team_name_to_id_df['team_name'] == team_b_norm]
    
    if len(team_a_id_row) == 0:
        logger.error(f"Could not find team ID for '{team_a}'. Available teams may not include this team.")
        logger.error("Please ensure the team name matches the normalized name in the team mapping.")
        return None
    
    if len(team_b_id_row) == 0:
        logger.error(f"Could not find team ID for '{team_b}'. Available teams may not include this team.")
        logger.error("Please ensure the team name matches the normalized name in the team mapping.")
        return None
    
    team_a_id = int(team_a_id_row.iloc[0]['team_id'])
    team_b_id = int(team_b_id_row.iloc[0]['team_id'])
    
    # Get map ID
    map_mapping_file = mappings_dir / "map_name_to_id.csv"
    if map_mapping_file.exists():
        map_mapping_df = pd.read_csv(map_mapping_file)
        map_mapping_df['map_name'] = map_mapping_df['map_name'].apply(normalize_map_name)
        map_id_row = map_mapping_df[map_mapping_df['map_name'] == map_name_norm]
        if len(map_id_row) > 0:
            map_id = int(map_id_row.iloc[0]['map_id'])
        else:
            logger.warning(f"Could not find map ID for {map_name}, using 0")
            map_id = 0
    else:
        logger.warning("Map mapping file not found, using map_id=0")
        map_id = 0
    
    # Load team match history for streak calculation
    team_match_history = {}
    team_results_files = sorted(team_results_dir.glob("*.csv"))
    for file_path in team_results_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                df['team_name'] = df['team_name'].apply(normalize_team_name)
                df['opponent_name'] = df['opponent_name'].apply(normalize_team_name)
                df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
                df = df.dropna(subset=['match_date'])
                df = df[df['match_date'] < match_date_dt]  # Only matches before this one
                
                for _, row in df.iterrows():
                    team_id = int(row['team_id'])
                    opponent_id = int(row['opponent_id'])
                    result = row['result'].upper().strip()
                    won = 1 if result == 'W' else 0
                    
                    if team_id not in team_match_history:
                        team_match_history[team_id] = []
                    team_match_history[team_id].append((row['match_date'], won))
                    
                    if opponent_id not in team_match_history:
                        team_match_history[opponent_id] = []
                    team_match_history[opponent_id].append((row['match_date'], 1 - won))
        except Exception as e:
            if verbose:
                logger.warning(f"Error loading {file_path.name}: {e}")
    
    # Sort match history by date
    for team_id in team_match_history:
        team_match_history[team_id].sort(key=lambda x: x[0])
    
    # Load player stats
    player_stats_df = pd.DataFrame()
    player_stats_files = sorted(player_results_dir.glob("*_weekly_stats.csv")) if player_results_dir.exists() else []
    for file_path in player_stats_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                df['team_name'] = df['team_name'].apply(normalize_team_name)
                df['map_name'] = df['map_name'].apply(normalize_map_name)
                df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
                df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
                df = df.dropna(subset=['start_date', 'end_date'])
                player_stats_df = pd.concat([player_stats_df, df], ignore_index=True)
        except Exception as e:
            if verbose:
                logger.warning(f"Error loading {file_path.name}: {e}")
    
    if verbose:
        logger.info("Computing features for match...")
    
    # Get team A's stats against team B BEFORE this match
    team_a_vs_b = opponent_stats[
        (opponent_stats['team_name'] == team_a_norm) & 
        (opponent_stats['opponent_name'] == team_b_norm) &
        (opponent_stats['match_date'] < match_date_dt)
    ]
    
    if len(team_a_vs_b) > 0:
        latest_stats = team_a_vs_b.iloc[-1]
        team_a_wins_vs_b = latest_stats['cumulative_wins']
        team_a_losses_vs_b = latest_stats['cumulative_losses']
        team_a_total_vs_b = team_a_wins_vs_b + team_a_losses_vs_b
        team_a_winrate_vs_b = (team_a_wins_vs_b / team_a_total_vs_b * 100) if team_a_total_vs_b > 0 else 0.0
    else:
        team_a_wins_vs_b = 0
        team_a_losses_vs_b = 0
        team_a_total_vs_b = 0
        team_a_winrate_vs_b = 0.0
    
    # Get team A's stats on this map BEFORE this match
    team_a_map = map_stats[
        (map_stats['team_name'] == team_a_norm) &
        (map_stats['map_name'] == map_name_norm) &
        (map_stats['match_date'] < match_date_dt)
    ]
    
    if len(team_a_map) > 0:
        latest_map_stats_a = team_a_map.iloc[-1]
        team_a_map_wins = latest_map_stats_a['cumulative_wins']
        team_a_map_losses = latest_map_stats_a['cumulative_losses']
        team_a_map_total = team_a_map_wins + team_a_map_losses
        team_a_map_winrate = (team_a_map_wins / team_a_map_total * 100) if team_a_map_total > 0 else 0.0
    else:
        team_a_map_wins = 0
        team_a_map_losses = 0
        team_a_map_total = 0
        team_a_map_winrate = 0.0
    
    # Get team B's stats on this map BEFORE this match
    team_b_map = map_stats[
        (map_stats['team_name'] == team_b_norm) &
        (map_stats['map_name'] == map_name_norm) &
        (map_stats['match_date'] < match_date_dt)
    ]
    
    if len(team_b_map) > 0:
        latest_map_stats_b = team_b_map.iloc[-1]
        team_b_map_wins = latest_map_stats_b['cumulative_wins']
        team_b_map_losses = latest_map_stats_b['cumulative_losses']
        team_b_map_total = team_b_map_wins + team_b_map_losses
        team_b_map_winrate = (team_b_map_wins / team_b_map_total * 100) if team_b_map_total > 0 else 0.0
    else:
        team_b_map_wins = 0
        team_b_map_losses = 0
        team_b_map_total = 0
        team_b_map_winrate = 0.0
    
    # Get team rankings
    team_a_rankings = rankings_df[
        (rankings_df['team_id'] == team_a_id) &
        (rankings_df['date'] <= match_date_dt)
    ]
    
    if len(team_a_rankings) > 0:
        latest_ranking_a = team_a_rankings.iloc[-1]
        team_a_global_ranking = latest_ranking_a['points']
    else:
        team_a_global_ranking = 0
    
    team_b_rankings = rankings_df[
        (rankings_df['team_id'] == team_b_id) &
        (rankings_df['date'] <= match_date_dt)
    ]
    
    if len(team_b_rankings) > 0:
        latest_ranking_b = team_b_rankings.iloc[-1]
        team_b_global_ranking = latest_ranking_b['points']
    else:
        team_b_global_ranking = 0
    
    # Calculate streaks
    team_a_win_streak, team_a_loss_streak = calculate_streak(team_a_id, team_match_history)
    team_b_win_streak, team_b_loss_streak = calculate_streak(team_b_id, team_match_history)
    
    # Get player stats
    team_a_player_features = get_player_stats(team_a_norm, map_name_norm, match_date_dt, player_stats_df, max_players=5)
    team_b_player_features = get_player_stats(team_b_norm, map_name_norm, match_date_dt, player_stats_df, max_players=5)
    
    # Create feature row
    feature_row = {
        'team_a_id': team_a_id,
        'team_b_id': team_b_id,
        'map_id': map_id,
        'match_date': match_date_dt,
        'team_a_wins_vs_b': team_a_wins_vs_b,
        'team_a_losses_vs_b': team_a_losses_vs_b,
        'team_a_total_vs_b': team_a_total_vs_b,
        'team_a_winrate_vs_b': round(team_a_winrate_vs_b, 2),
        'team_a_map_wins': team_a_map_wins,
        'team_a_map_losses': team_a_map_losses,
        'team_a_map_total': team_a_map_total,
        'team_a_map_winrate': round(team_a_map_winrate, 2),
        'team_b_map_wins': team_b_map_wins,
        'team_b_map_losses': team_b_map_losses,
        'team_b_map_total': team_b_map_total,
        'team_b_map_winrate': round(team_b_map_winrate, 2),
        'team_a_global_ranking': team_a_global_ranking,
        'team_b_global_ranking': team_b_global_ranking,
        'team_a_win_streak': team_a_win_streak,
        'team_a_loss_streak': team_a_loss_streak,
        'team_b_win_streak': team_b_win_streak,
        'team_b_loss_streak': team_b_loss_streak,
    }
    
    # Add player features
    for i in range(5):
        feature_row[f'team_a_player_{i+1}_overall_rating'] = team_a_player_features[i*3]
        feature_row[f'team_a_player_{i+1}_utility_success'] = team_a_player_features[i*3+1]
        feature_row[f'team_a_player_{i+1}_opening_rating'] = team_a_player_features[i*3+2]
        feature_row[f'team_b_player_{i+1}_overall_rating'] = team_b_player_features[i*3]
        feature_row[f'team_b_player_{i+1}_utility_success'] = team_b_player_features[i*3+1]
        feature_row[f'team_b_player_{i+1}_opening_rating'] = team_b_player_features[i*3+2]
    
    return pd.DataFrame([feature_row])


def prepare_features_for_model(features_df: pd.DataFrame, project_root: Optional[Path] = None) -> pd.DataFrame:
    """Prepare features for model inference (one-hot encode map_id, remove unnecessary columns).
    
    Args:
        features_df: DataFrame with match features
        project_root: Root directory of the project
        
    Returns:
        DataFrame ready for model inference
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    df = features_df.copy()
    
    # Remove team_a_id, team_b_id, match_date
    columns_to_remove = ['team_a_id', 'team_b_id', 'match_date']
    df_cleaned = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    
    # One-hot encode map_id
    map_dummies = pd.get_dummies(df_cleaned['map_id'], prefix='map_id', dtype=int)
    df_without_map = df_cleaned.drop(columns=['map_id'])
    df_final = pd.concat([map_dummies, df_without_map], axis=1)
    
    return df_final


def predict_match(
    match_url: Optional[str] = None,
    fetched_data_file: Optional[Path] = None,
    team_a: Optional[str] = None,
    team_b: Optional[str] = None,
    map_name: Optional[str] = None,
    match_date: Optional[str] = None,
    model_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Optional[Dict]:
    """Predict match outcome.
    
    The match information (teams, map, date) is fetched from the HLTV URL.
    Features are then computed using this match info combined with historical
    data (win rates, streaks, rankings, etc.) up to the match date.
    
    Args:
        match_url: HLTV match URL (will scrape teams, map, and date from this URL)
        team_a: Name of team A (required if match_url not provided)
        team_b: Name of team B (required if match_url not provided)
        map_name: Map name (required if match_url not provided)
        match_date: Match date in YYYY-MM-DD format (required if match_url not provided)
        model_path: Path to saved model file (default: models/xgboost_model.pkl)
        project_root: Root directory of the project
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary with prediction results:
        - team_a: Name of team A (from URL)
        - team_b: Name of team B (from URL)
        - map: Map name (from URL)
        - match_date: Match date (from URL)
        - team_a_win_probability: Probability that team A wins
        - team_b_win_probability: Probability that team B wins
        - predicted_winner: Predicted winner (team_a or team_b)
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load model first
    if model_path is None:
        model_path = project_root / "models" / "xgboost_model.pkl"
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train a model first using: python -m src.main train")
        logger.error("This will save the model to models/xgboost_model.pkl")
        return None
    
    if verbose:
        logger.info(f"Loading model from {model_path}...")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load calibration data if available
    calibration_path = model_path.parent / "calibration_data.pkl"
    calibration_data = None
    if calibration_path.exists():
        with open(calibration_path, "rb") as f:
            calibration_data = pickle.load(f)
    
    # Load fetched data if provided
    if fetched_data_file:
        if not fetched_data_file.exists():
            logger.error(f"Fetched data file not found: {fetched_data_file}")
            return None
        
        import json
        with open(fetched_data_file, 'r') as f:
            fetched_data = json.load(f)
        
        team_a = fetched_data['team_a']
        team_b = fetched_data['team_b']
        map_name = fetched_data['map']
        match_date = fetched_data['match_date']
        historical_data_dict = fetched_data.get('historical_data', {})
        
        if verbose:
            logger.info(f"Loaded match data: {team_a} vs {team_b} on {map_name or 'All Maps'} ({match_date})")
        
        # If multiple maps, predict for all
        if not map_name and len(historical_data_dict) > 1:
            predictions_by_map = {}
            
            for map_name_iter, hist_data in historical_data_dict.items():
                if verbose:
                    logger.info(f"Computing prediction for {map_name_iter}...")
                
                # Convert JSON data back to format expected by create_features_from_historical_data
                historical_data = {
                    "team_results": hist_data.get("team_results", []),
                    "head_to_head": hist_data.get("head_to_head", []),
                    "team_a_id": hist_data.get("team_a_id"),
                    "team_b_id": hist_data.get("team_b_id"),
                    "team_a_slug": hist_data.get("team_a_slug"),
                    "team_b_slug": hist_data.get("team_b_slug"),
                    "map_id": hist_data.get("map_id"),
                }
                
                features_df = create_features_from_historical_data(
                    team_a=team_a,
                    team_b=team_b,
                    map_name=map_name_iter,
                    match_date=match_date,
                    historical_data=historical_data,
                    verbose=False
                )
                
                if features_df is None:
                    continue
                
                X = prepare_features_for_model(features_df, project_root=project_root)
                
                # Align columns
                training_data_file = project_root / "data" / "preprocessed" / "final_features.csv"
                if training_data_file.exists():
                    training_df = pd.read_csv(training_data_file, nrows=1)
                    training_cols = [col for col in training_df.columns if col != 'team_a_won']
                    X_aligned = pd.DataFrame(0, index=X.index, columns=training_cols)
                    for col in X.columns:
                        if col in training_cols:
                            X_aligned[col] = X[col].values
                    X = X_aligned
                
                proba = model.predict_proba(X)[0]
                team_a_win_prob = proba[1]
                team_b_win_prob = proba[0]
                predicted_winner = team_a if team_a_win_prob > 0.5 else team_b
                
                # Get calibration accuracy
                # Use the probability of the predicted winner for calibration lookup
                from ..eval.calibration import get_calibration_accuracy
                calibration_accuracy = None
                if calibration_data:
                    prob_for_calibration = team_a_win_prob if predicted_winner == team_a else team_b_win_prob
                    calibration_accuracy = get_calibration_accuracy(
                        calibration_data, prob_for_calibration
                    )
                
                predictions_by_map[map_name_iter] = {
                    'team_a_win_probability': float(team_a_win_prob),
                    'team_b_win_probability': float(team_b_win_prob),
                    'predicted_winner': predicted_winner,
                    'calibration_accuracy': float(calibration_accuracy) if calibration_accuracy is not None else None,
                }
            
            return {
                'team_a': team_a,
                'team_b': team_b,
                'map': None,
                'match_date': match_date,
                'predictions_by_map': predictions_by_map,
            }
        
        # Single map prediction
        elif map_name and map_name in historical_data_dict:
            hist_data = historical_data_dict[map_name]
            historical_data = {
                "team_results": hist_data.get("team_results", []),
                "head_to_head": hist_data.get("head_to_head", []),
                "team_a_id": hist_data.get("team_a_id"),
                "team_b_id": hist_data.get("team_b_id"),
                "team_a_slug": hist_data.get("team_a_slug"),
                "team_b_slug": hist_data.get("team_b_slug"),
                "map_id": hist_data.get("map_id"),
            }
            
            features_df = create_features_from_historical_data(
                team_a=team_a,
                team_b=team_b,
                map_name=map_name,
                match_date=match_date,
                historical_data=historical_data,
                verbose=verbose
            )
            
            if features_df is None:
                logger.error(f"Failed to create features from fetched data")
                return None
            # features_df is set, continue to prediction part below (skip elif/else blocks)
        else:
            if map_name:
                logger.error(f"Historical data not found for map: {map_name}")
            else:
                logger.error("No historical data found in fetched file")
            return None
    
    # Scrape match info and fetch historical data from URL (only if fetched_data_file was not used)
    if not fetched_data_file and match_url:
        if verbose:
            logger.info(f"Fetching match information and historical data from URL: {match_url}")
        from .match_scraper import scrape_match_info
        match_info = scrape_match_info(match_url)
        if not match_info:
            logger.error("Failed to scrape match information from URL")
            return None
        team_a = match_info['team_a']
        team_b = match_info['team_b']
        map_name = match_info['map']
        match_date = match_info['match_date']
        
        if verbose:
            logger.info(f"Match info: {team_a} vs {team_b} on {map_name or 'TBD'} ({match_date})")
        
        # If map not decided, predict for all maps
        if not map_name:
            if verbose:
                logger.info("Map not yet decided. Computing predictions for all maps...")
            
            all_maps = ["Mirage", "Inferno", "Dust2", "Nuke", "Overpass", "Vertigo", "Ancient", "Anubis", "Train"]
            predictions_by_map = {}
            
            for idx, map_name_iter in enumerate(all_maps):
                if verbose:
                    logger.info(f"\nComputing prediction for {map_name_iter} ({idx+1}/{len(all_maps)})...")
                
                # Add delay between map requests to prevent rate limiting
                if idx > 0:
                    import time
                    import random
                    delay = random.uniform(2.0, 4.0)
                    if verbose:
                        logger.debug(f"Waiting {delay:.1f}s before next request...")
                    time.sleep(delay)
                
                # Fetch historical data for this map
                historical_data = fetch_historical_data_for_match(
                    match_url=match_url,
                    team_a_name=team_a,
                    team_b_name=team_b,
                    map_name=map_name_iter,
                    match_date=match_date,
                    verbose=False  # Less verbose for multiple maps
                )
                
                if not historical_data:
                    if verbose:
                        logger.warning(f"  Could not fetch historical data for {map_name_iter} (failed to get team IDs or fetch data)")
                    continue
                
                # Check if we actually got any data (even if empty, historical_data exists)
                # Empty data is OK - we can still predict with default values
                if verbose and not historical_data.get("data_found", True):
                    logger.debug(f"  No historical matches found for {map_name_iter}, using default features")
                
                # Create features from fetched historical data
                features_df = create_features_from_historical_data(
                    team_a=team_a,
                    team_b=team_b,
                    map_name=map_name_iter,
                    match_date=match_date,
                    historical_data=historical_data,
                    verbose=False
                )
                
                if features_df is None:
                    if verbose:
                        logger.warning(f"  Could not create features for {map_name_iter}")
                    continue
                
                # Prepare features for model
                X = prepare_features_for_model(features_df, project_root=project_root)
                
                # Ensure all columns match training data
                training_data_file = project_root / "data" / "preprocessed" / "final_features.csv"
                if training_data_file.exists():
                    training_df = pd.read_csv(training_data_file, nrows=1)
                    training_cols = [col for col in training_df.columns if col != 'team_a_won']
                    
                    X_aligned = pd.DataFrame(0, index=X.index, columns=training_cols)
                    for col in X.columns:
                        if col in training_cols:
                            X_aligned[col] = X[col].values
                    X = X_aligned
                
                # Predict
                proba = model.predict_proba(X)[0]
                team_a_win_prob = proba[1]
                team_b_win_prob = proba[0]
                predicted_winner = team_a if team_a_win_prob > 0.5 else team_b
                
                predictions_by_map[map_name_iter] = {
                    'team_a_win_probability': float(team_a_win_prob),
                    'team_b_win_probability': float(team_b_win_prob),
                    'predicted_winner': predicted_winner,
                }
            
            if not predictions_by_map:
                logger.error("Failed to generate predictions for any map")
                return None
            
            # Return aggregated results
            return {
                'team_a': team_a,
                'team_b': team_b,
                'map': None,
                'match_date': match_date,
                'predictions_by_map': predictions_by_map,
            }
        
        # Map is specified - proceed with single prediction
        # Fetch historical data from HLTV
        historical_data = fetch_historical_data_for_match(
            match_url=match_url,
            team_a_name=team_a,
            team_b_name=team_b,
            map_name=map_name,
            match_date=match_date,
            verbose=verbose
        )
        
        if not historical_data:
            logger.error("Failed to fetch historical data from HLTV")
            return None
        
        # Create features from fetched historical data
        features_df = create_features_from_historical_data(
            team_a=team_a,
            team_b=team_b,
            map_name=map_name,
            match_date=match_date,
            historical_data=historical_data,
            verbose=verbose
        )
        
        if features_df is None:
            logger.error("Failed to create features from historical data")
            return None
    
    # Manual specification - use preprocessed data (only if neither fetched_data_file nor match_url was used)
    if not fetched_data_file and not match_url:
        if not all([team_a, team_b, map_name, match_date]):
            logger.error("Either fetched_data_file, match_url, or all of (team_a, team_b, map_name, match_date) must be provided")
            return None
        
        if verbose:
            logger.info("Using preprocessed historical data...")
        features_df = create_match_features_for_inference(
            team_a=team_a,
            team_b=team_b,
            map_name=map_name,
            match_date=match_date,
            project_root=project_root,
            verbose=verbose
        )
        
        if features_df is None:
            logger.error("Failed to create features for match")
            return None
    
    # Prepare features for model
    X = prepare_features_for_model(features_df, project_root=project_root)
    
    # Ensure all columns match training data
    training_data_file = project_root / "data" / "preprocessed" / "final_features.csv"
    if training_data_file.exists():
        training_df = pd.read_csv(training_data_file, nrows=1)
        training_cols = [col for col in training_df.columns if col != 'team_a_won']
        
        X_aligned = pd.DataFrame(0, index=X.index, columns=training_cols)
        for col in X.columns:
            if col in training_cols:
                X_aligned[col] = X[col].values
        X = X_aligned
    
    # Predict
    proba = model.predict_proba(X)[0]
    team_a_win_prob = proba[1]
    team_b_win_prob = proba[0]
    predicted_winner = team_a if team_a_win_prob > 0.5 else team_b
    
    # Get calibration accuracy
    # Use the probability of the predicted winner, not team_a
    from ..eval.calibration import get_calibration_accuracy
    calibration_accuracy = None
    if calibration_data:
        # Use the probability of the predicted winner for calibration lookup
        # If team_a wins, use team_a_win_prob; if team_b wins, use team_b_win_prob
        prob_for_calibration = team_a_win_prob if predicted_winner == team_a else team_b_win_prob
        calibration_accuracy = get_calibration_accuracy(
            calibration_data, prob_for_calibration
        )
    
    return {
        'team_a': team_a,
        'team_b': team_b,
        'map': map_name,
        'match_date': match_date,
        'team_a_win_probability': float(team_a_win_prob),
        'team_b_win_probability': float(team_b_win_prob),
        'predicted_winner': predicted_winner,
        'calibration_accuracy': float(calibration_accuracy) if calibration_accuracy is not None else None,
    }

