"""Calculate cumulative statistics for teams."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from .utils import normalize_team_name, normalize_map_name

logger = logging.getLogger(__name__)


def calculate_opponent_cumulative_stats(
    team_results_dir: Path,
    output_file: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Calculate cumulative wins/losses for each team against each opponent.
    
    Args:
        team_results_dir: Directory containing team result CSV files.
        output_file: Path to save the cumulative stats CSV.
        verbose: Whether to print progress messages.
        
    Returns:
        DataFrame with cumulative opponent statistics.
    """
    team_results_files = sorted(team_results_dir.glob("*.csv"))
    
    if verbose:
        logger.info(f"Processing {len(team_results_files)} team files to calculate cumulative wins/losses against opponents...\n")
    
    all_team_opponent_stats = []
    
    for file_path in team_results_files:
        try:
            df = pd.read_csv(file_path)
            
            if not all(col in df.columns for col in ['team_name', 'opponent_name', 'match_date', 'result']):
                if verbose:
                    logger.warning(f"  Skipping {file_path.name}: Missing required columns")
                continue
            
            df['opponent_name'] = df['opponent_name'].apply(normalize_team_name)
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            df = df.dropna(subset=['match_date'])
            df = df.sort_values('match_date', ascending=True).reset_index(drop=True)
            
            team_name = normalize_team_name(df['team_name'].iloc[0] if len(df) > 0 else file_path.stem)
            df['team_name'] = team_name
            
            for opponent_name, opponent_data in df.groupby('opponent_name'):
                opponent_data = opponent_data.sort_values('match_date', ascending=True).reset_index(drop=True)
                opponent_data['is_win'] = (opponent_data['result'].str.upper().str.strip() == 'W').astype(int)
                opponent_data['is_loss'] = (opponent_data['result'].str.upper().str.strip() == 'L').astype(int)
                opponent_data['cumulative_wins'] = opponent_data['is_win'].cumsum()
                opponent_data['cumulative_losses'] = opponent_data['is_loss'].cumsum()
                opponent_data['match_number'] = range(1, len(opponent_data) + 1)
                
                result_df = opponent_data[[
                    'team_name', 'opponent_name', 'opponent_id', 'match_date', 'result',
                    'cumulative_wins', 'cumulative_losses', 'match_number'
                ]].copy()
                
                all_team_opponent_stats.append(result_df)
            
            if verbose:
                num_opponents = df['opponent_name'].nunique()
                logger.info(f"  {file_path.name}: Processed {num_opponents} opponents, {len(df)} total matches")
        
        except Exception as e:
            if verbose:
                logger.error(f"  Error processing {file_path.name}: {e}")
    
    if all_team_opponent_stats:
        cumulative_stats_df = pd.concat(all_team_opponent_stats, ignore_index=True)
        cumulative_stats_df = cumulative_stats_df.sort_values(
            ['team_name', 'opponent_name', 'match_date'], 
            ascending=[True, True, True]
        ).reset_index(drop=True)
        
        if verbose:
            logger.info(f"\nCreated cumulative opponent statistics DataFrame")
            logger.info(f"Total rows: {len(cumulative_stats_df)}")
            logger.info(f"Unique teams: {cumulative_stats_df['team_name'].nunique()}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cumulative_stats_df.to_csv(output_file, index=False)
        
        if verbose:
            logger.info(f"Saved to: {output_file}")
        
        return cumulative_stats_df
    else:
        return pd.DataFrame()


def calculate_map_cumulative_stats(
    team_results_dir: Path,
    output_file: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Calculate cumulative wins/losses for each team on each map.
    
    Args:
        team_results_dir: Directory containing team result CSV files.
        output_file: Path to save the cumulative stats CSV.
        verbose: Whether to print progress messages.
        
    Returns:
        DataFrame with cumulative map statistics.
    """
    team_results_files = sorted(team_results_dir.glob("*.csv"))
    
    if verbose:
        logger.info(f"Processing {len(team_results_files)} team files to calculate cumulative wins/losses on maps...\n")
    
    all_team_map_stats = []
    
    for file_path in team_results_files:
        try:
            df = pd.read_csv(file_path)
            
            if not all(col in df.columns for col in ['team_name', 'map_name', 'match_date', 'result']):
                if verbose:
                    logger.warning(f"  Skipping {file_path.name}: Missing required columns")
                continue
            
            df['map_name'] = df['map_name'].apply(normalize_map_name)
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            df = df.dropna(subset=['match_date'])
            df = df.sort_values('match_date', ascending=True).reset_index(drop=True)
            
            team_name = normalize_team_name(df['team_name'].iloc[0] if len(df) > 0 else file_path.stem)
            df['team_name'] = team_name
            
            for map_name, map_data in df.groupby('map_name'):
                map_data = map_data.sort_values('match_date', ascending=True).reset_index(drop=True)
                map_data['is_win'] = (map_data['result'].str.upper().str.strip() == 'W').astype(int)
                map_data['is_loss'] = (map_data['result'].str.upper().str.strip() == 'L').astype(int)
                map_data['cumulative_wins'] = map_data['is_win'].cumsum()
                map_data['cumulative_losses'] = map_data['is_loss'].cumsum()
                map_data['match_number'] = range(1, len(map_data) + 1)
                
                result_df = map_data[[
                    'team_name', 'map_name', 'map_id', 'match_date', 'result',
                    'cumulative_wins', 'cumulative_losses', 'match_number'
                ]].copy()
                
                all_team_map_stats.append(result_df)
            
            if verbose:
                num_maps = df['map_name'].nunique()
                logger.info(f"  {file_path.name}: Processed {num_maps} maps, {len(df)} total matches")
        
        except Exception as e:
            if verbose:
                logger.error(f"  Error processing {file_path.name}: {e}")
    
    if all_team_map_stats:
        cumulative_stats_df = pd.concat(all_team_map_stats, ignore_index=True)
        cumulative_stats_df = cumulative_stats_df.sort_values(
            ['team_name', 'map_name', 'match_date'], 
            ascending=[True, True, True]
        ).reset_index(drop=True)
        
        if verbose:
            logger.info(f"\nCreated cumulative map statistics DataFrame")
            logger.info(f"Total rows: {len(cumulative_stats_df)}")
            logger.info(f"Unique teams: {cumulative_stats_df['team_name'].nunique()}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cumulative_stats_df.to_csv(output_file, index=False)
        
        if verbose:
            logger.info(f"Saved to: {output_file}")
        
        return cumulative_stats_df
    else:
        return pd.DataFrame()
