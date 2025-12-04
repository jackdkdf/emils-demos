"""Create match features from cumulative statistics and player data."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_team_name(name):
    """Normalize team name by lowercasing and replacing spaces with hyphens."""
    if pd.isna(name):
        return name
    return str(name).lower().strip().replace(' ', '-')


def normalize_map_name(name):
    """Normalize map name by lowercasing and replacing spaces with hyphens."""
    if pd.isna(name):
        return name
    return str(name).lower().strip().replace(' ', '-')


def get_player_stats(team_name, map_name, match_date, player_stats_df, max_players=5):
    """
    Get player stats for a team on a map from the week BEFORE the match date.
    Returns up to max_players players with their overall_rating, utility_success, and opening_rating.
    """
    if len(player_stats_df) == 0:
        return [0.0] * (max_players * 3)  # Return zeros for all features
    
    # Filter to this team and map
    team_map_stats = player_stats_df[
        (player_stats_df['team_name'] == team_name) &
        (player_stats_df['map_name'] == map_name)
    ]
    
    if len(team_map_stats) == 0:
        return [0.0] * (max_players * 3)
    
    # Get stats from weeks BEFORE the match date (end_date < match_date)
    # We want the most recent week before the match
    before_match = team_map_stats[team_map_stats['end_date'] < match_date]
    
    if len(before_match) == 0:
        return [0.0] * (max_players * 3)
    
    # Get the most recent week (latest end_date)
    latest_week = before_match.loc[before_match['end_date'].idxmax()]
    latest_end_date = latest_week['end_date']
    
    # Get all players from that week
    week_stats = before_match[before_match['end_date'] == latest_end_date]
    
    if len(week_stats) == 0:
        return [0.0] * (max_players * 3)
    
    # Sort by overall_rating descending to get top players, then take up to max_players
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
    
    return features[:max_players * 3]  # Ensure exactly max_players * 3 features


def calculate_streak(team_id, team_match_history):
    """Calculate current win streak and loss streak for a team based on their match history."""
    if team_id not in team_match_history or len(team_match_history[team_id]) == 0:
        return 0, 0  # No history, so no streak
    
    # Get the most recent matches (they're already in chronological order)
    team_matches = team_match_history[team_id]
    
    # Calculate streak from most recent matches backwards
    win_streak = 0
    loss_streak = 0
    
    # Start from the most recent match and work backwards
    for date, won in reversed(team_matches):
        if won == 1:
            if loss_streak > 0:
                break  # Streak broken by a win
            win_streak += 1
        else:
            if win_streak > 0:
                break  # Streak broken by a loss
            loss_streak += 1
    
    return win_streak, loss_streak


def create_match_features(
    team_results_dir: Path,
    opponent_stats_file: Path,
    map_stats_file: Path,
    rankings_file: Path,
    team_name_to_id_file: Path,
    player_results_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Create match features dataset - EXACTLY matching notebook Cell 4.
    
    This function replicates the notebook preprocessing cell exactly to ensure
    identical results.
    """
    if verbose:
        logger.info("Loading cumulative statistics...")
    
    # Load cumulative statistics
    opponent_stats = pd.read_csv(opponent_stats_file)
    map_stats = pd.read_csv(map_stats_file)
    
    # Convert dates to datetime
    opponent_stats['match_date'] = pd.to_datetime(opponent_stats['match_date'])
    map_stats['match_date'] = pd.to_datetime(map_stats['match_date'])
    
    # Normalize names in the stats DataFrames
    opponent_stats['team_name'] = opponent_stats['team_name'].apply(normalize_team_name)
    opponent_stats['opponent_name'] = opponent_stats['opponent_name'].apply(normalize_team_name)
    map_stats['team_name'] = map_stats['team_name'].apply(normalize_team_name)
    map_stats['map_name'] = map_stats['map_name'].apply(normalize_map_name)
    
    if verbose:
        logger.info(f"Loaded {len(opponent_stats)} opponent statistics rows")
        logger.info(f"Loaded {len(map_stats)} map statistics rows")
    
    if verbose:
        logger.info("\nLoading team rankings...")
    
    # Load rankings and team name to ID mapping
    rankings_df = pd.read_csv(rankings_file)
    team_name_to_id_df = pd.read_csv(team_name_to_id_file)
    
    # Extract rank number from "#1", "#2", etc.
    rankings_df['rank'] = rankings_df['rank'].str.replace('#', '').astype(int)
    
    # Normalize team names in rankings
    rankings_df['name_normalized'] = rankings_df['name'].apply(normalize_team_name)
    
    # Convert date to datetime
    rankings_df['date'] = pd.to_datetime(rankings_df['date'])
    
    # Merge rankings with team IDs
    rankings_df = rankings_df.merge(
        team_name_to_id_df,
        left_on='name_normalized',
        right_on='team_name',
        how='left'
    )
    
    # Filter out rows where we couldn't find a team ID (teams not in our dataset)
    rankings_df = rankings_df.dropna(subset=['team_id'])
    
    if verbose:
        logger.info(f"Loaded {len(rankings_df)} ranking entries")
        logger.info(f"Unique teams in rankings: {rankings_df['team_id'].nunique()}")
        logger.info(f"Date range in rankings: {rankings_df['date'].min()} to {rankings_df['date'].max()}")
    
    # Sort rankings by date and team_id for efficient lookup
    rankings_df = rankings_df.sort_values(['date', 'team_id']).reset_index(drop=True)
    
    if verbose:
        logger.info("\nLoading all team match data...")
    
    # Load all team result files and combine
    all_matches = []
    
    team_results_files = sorted(team_results_dir.glob("*.csv"))
    for file_path in team_results_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Normalize team and opponent names
                df['team_name'] = df['team_name'].apply(normalize_team_name)
                df['opponent_name'] = df['opponent_name'].apply(normalize_team_name)
                df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
                df = df.dropna(subset=['match_date'])
                all_matches.append(df)
        except Exception as e:
            if verbose:
                logger.warning(f"  Error loading {file_path.name}: {e}")
    
    # Combine all matches
    matches_df = pd.concat(all_matches, ignore_index=True)
    matches_df = matches_df.sort_values('match_date').reset_index(drop=True)
    
    if verbose:
        logger.info(f"Loaded {len(matches_df)} total matches")
        logger.info(f"Date range: {matches_df['match_date'].min()} to {matches_df['match_date'].max()}")
    
    # Ensure matches are sorted by date for streak calculation
    matches_df = matches_df.sort_values('match_date').reset_index(drop=True)
    
    if verbose:
        logger.info("\nLoading player statistics...")
    
    # Load all player stats files
    if player_results_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        player_results_dir = project_root / "data" / "raw" / "player_results"
    
    player_stats_files = sorted(player_results_dir.glob("*_weekly_stats.csv")) if player_results_dir.exists() else []
    
    all_player_stats = []
    for file_path in player_stats_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Normalize team and map names
                df['team_name'] = df['team_name'].apply(normalize_team_name)
                df['map_name'] = df['map_name'].apply(normalize_map_name)
                df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
                df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
                df = df.dropna(subset=['start_date', 'end_date'])
                all_player_stats.append(df)
        except Exception as e:
            if verbose:
                logger.warning(f"  Error loading {file_path.name}: {e}")
    
    # Combine all player stats
    if all_player_stats:
        player_stats_df = pd.concat(all_player_stats, ignore_index=True)
        player_stats_df = player_stats_df.sort_values(['team_name', 'map_name', 'start_date']).reset_index(drop=True)
        if verbose:
            logger.info(f"Loaded {len(player_stats_df)} player stat rows")
            logger.info(f"Unique teams in player stats: {player_stats_df['team_name'].nunique()}")
            logger.info(f"Date range: {player_stats_df['start_date'].min()} to {player_stats_df['end_date'].max()}")
    else:
        player_stats_df = pd.DataFrame()
        if verbose:
            logger.warning("No player stats files found")
    
    if verbose:
        logger.info("\nCalculating win/loss streaks...")
    
    # Track each team's recent match results chronologically
    # We'll build this as we iterate through matches
    team_match_history = {}  # {team_id: [(date, won), ...]}
    
    if verbose:
        logger.info("\nCreating feature dataset...")
    
    # Create feature rows
    feature_rows = []
    
    for idx, match in matches_df.iterrows():
        if verbose and idx % 1000 == 0:
            logger.info(f"  Processing match {idx+1}/{len(matches_df)}")
        
        team_a = match['team_name']
        team_b = match['opponent_name']
        team_a_id = match['team_id']
        team_b_id = match['opponent_id']
        map_name = normalize_map_name(match['map_name']) if pd.notna(match['map_name']) else match['map_name']
        match_date = match['match_date']
        
        # Get team A's stats against team B BEFORE this match
        team_a_vs_b = opponent_stats[
            (opponent_stats['team_name'] == team_a) & 
            (opponent_stats['opponent_name'] == team_b) &
            (opponent_stats['match_date'] < match_date)
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
            (map_stats['team_name'] == team_a) &
            (map_stats['map_name'] == map_name) &
            (map_stats['match_date'] < match_date)
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
            (map_stats['team_name'] == team_b) &
            (map_stats['map_name'] == map_name) &
            (map_stats['match_date'] < match_date)
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
        
        # Get team A's global ranking at or before this match date
        team_a_rankings = rankings_df[
            (rankings_df['team_id'] == team_a_id) &
            (rankings_df['date'] <= match_date)
        ]
        
        if len(team_a_rankings) > 0:
            # Get the most recent ranking
            latest_ranking_a = team_a_rankings.iloc[-1]
            team_a_global_ranking = latest_ranking_a['points']
        else:
            # No ranking found - use a low number (e.g., 0) to indicate unranked
            team_a_global_ranking = 0
        
        # Get team B's global ranking at or before this match date
        team_b_rankings = rankings_df[
            (rankings_df['team_id'] == team_b_id) &
            (rankings_df['date'] <= match_date)
        ]
        
        if len(team_b_rankings) > 0:
            # Get the most recent ranking
            latest_ranking_b = team_b_rankings.iloc[-1]
            team_b_global_ranking = latest_ranking_b['points']
        else:
            # No ranking found - use a low number (e.g., 0) to indicate unranked
            team_b_global_ranking = 0
        
        # Calculate win/loss streaks BEFORE this match (using current history)
        team_a_win_streak, team_a_loss_streak = calculate_streak(team_a_id, team_match_history)
        team_b_win_streak, team_b_loss_streak = calculate_streak(team_b_id, team_match_history)
        
        # Get player stats for both teams BEFORE this match (from week before match date)
        # IMPORTANT: Uses end_date < match_date to prevent data leakage
        team_a_player_features = get_player_stats(team_a, map_name, match_date, player_stats_df, max_players=5)
        team_b_player_features = get_player_stats(team_b, map_name, match_date, player_stats_df, max_players=5)
        
        # Determine if team A won
        result = match['result'].upper().strip()
        team_a_won = 1 if result == 'W' else 0
        team_b_won = 1 - team_a_won
        
        # Update match history AFTER calculating streaks (for next matches)
        # This ensures that when we process the next match, this match will be in the history
        # but the current match's result is NOT included in its own streak calculation
        if team_a_id not in team_match_history:
            team_match_history[team_a_id] = []
        if team_b_id not in team_match_history:
            team_match_history[team_b_id] = []
        
        # Add this match to history AFTER calculating streaks to prevent data leakage
        team_match_history[team_a_id].append((match_date, team_a_won))
        team_match_history[team_b_id].append((match_date, team_b_won))
        
        # Create feature row - EXACTLY matching notebook
        feature_row = {
            'team_a_id': team_a_id,
            'team_b_id': team_b_id,
            'map_id': match['map_id'],  # Direct access like notebook
            'match_date': match_date,
            # Team A vs Team B stats (up to this match)
            'team_a_wins_vs_b': team_a_wins_vs_b,
            'team_a_losses_vs_b': team_a_losses_vs_b,
            'team_a_total_vs_b': team_a_total_vs_b,
            'team_a_winrate_vs_b': round(team_a_winrate_vs_b, 2),
            # Team A map stats (up to this match)
            'team_a_map_wins': team_a_map_wins,
            'team_a_map_losses': team_a_map_losses,
            'team_a_map_total': team_a_map_total,
            'team_a_map_winrate': round(team_a_map_winrate, 2),
            # Team B map stats (up to this match)
            'team_b_map_wins': team_b_map_wins,
            'team_b_map_losses': team_b_map_losses,
            'team_b_map_total': team_b_map_total,
            'team_b_map_winrate': round(team_b_map_winrate, 2),
            # Win/Loss streaks (current streaks before this match)
            'team_a_win_streak': team_a_win_streak,
            'team_a_loss_streak': team_a_loss_streak,
            'team_b_win_streak': team_b_win_streak,
            'team_b_loss_streak': team_b_loss_streak,
            # Player stats (from week before match date to prevent data leakage)
            # Team A players (5 players × 3 stats = 15 features)
            'team_a_player1_overall_rating': team_a_player_features[0],
            'team_a_player1_utility_success': team_a_player_features[1],
            'team_a_player1_opening_rating': team_a_player_features[2],
            'team_a_player2_overall_rating': team_a_player_features[3],
            'team_a_player2_utility_success': team_a_player_features[4],
            'team_a_player2_opening_rating': team_a_player_features[5],
            'team_a_player3_overall_rating': team_a_player_features[6],
            'team_a_player3_utility_success': team_a_player_features[7],
            'team_a_player3_opening_rating': team_a_player_features[8],
            'team_a_player4_overall_rating': team_a_player_features[9],
            'team_a_player4_utility_success': team_a_player_features[10],
            'team_a_player4_opening_rating': team_a_player_features[11],
            'team_a_player5_overall_rating': team_a_player_features[12],
            'team_a_player5_utility_success': team_a_player_features[13],
            'team_a_player5_opening_rating': team_a_player_features[14],
            # Team B players (5 players × 3 stats = 15 features)
            'team_b_player1_overall_rating': team_b_player_features[0],
            'team_b_player1_utility_success': team_b_player_features[1],
            'team_b_player1_opening_rating': team_b_player_features[2],
            'team_b_player2_overall_rating': team_b_player_features[3],
            'team_b_player2_utility_success': team_b_player_features[4],
            'team_b_player2_opening_rating': team_b_player_features[5],
            'team_b_player3_overall_rating': team_b_player_features[6],
            'team_b_player3_utility_success': team_b_player_features[7],
            'team_b_player3_opening_rating': team_b_player_features[8],
            'team_b_player4_overall_rating': team_b_player_features[9],
            'team_b_player4_utility_success': team_b_player_features[10],
            'team_b_player4_opening_rating': team_b_player_features[11],
            'team_b_player5_overall_rating': team_b_player_features[12],
            'team_b_player5_utility_success': team_b_player_features[13],
            'team_b_player5_opening_rating': team_b_player_features[14],
            # Global rankings
            'team_a_global_ranking': team_a_global_ranking,
            'team_b_global_ranking': team_b_global_ranking,
            # Target variable
            'team_a_won': team_a_won
        }
        
        feature_rows.append(feature_row)
    
    # Create feature DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    if verbose:
        logger.info(f"\nCreated feature dataset with {len(features_df)} matches")
        logger.info(f"\nFeature columns: {list(features_df.columns)}")
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_file, index=False)
        if verbose:
            logger.info(f"\nSaved to: {output_file}")
    
    return features_df
