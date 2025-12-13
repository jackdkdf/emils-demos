"""Fetch match data from HLTV URL and save for later prediction."""

import logging
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from .match_scraper import scrape_match_info
from .historical_data import fetch_historical_data_for_match

logger = logging.getLogger(__name__)


def fetch_match_data(
    match_url: str,
    output_file: Optional[Path] = None,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Optional[Dict]:
    """Fetch match data from HLTV URL and save to temporary file.
    
    Args:
        match_url: HLTV match URL
        output_file: Path to save fetched data (default: data/temp/{match_id}.json)
        project_root: Root directory of the project
        verbose: Whether to print progress
        
    Returns:
        Dictionary with fetched match data (includes '_file_path' key), or None if failed
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    if verbose:
        logger.info(f"Fetching match data from URL: {match_url}")
    
    # Scrape match info
    match_info = scrape_match_info(match_url)
    if not match_info:
        logger.error("Failed to scrape match information from URL")
        return None
    
    team_a = match_info['team_a']
    team_b = match_info['team_b']
    map_name = match_info['map']
    match_date = match_info['match_date']
    match_id = match_info['match_id']
    
    if verbose:
        logger.info(f"Match info: {team_a} vs {team_b} on {map_name or 'TBD'} ({match_date})")
    
    # Determine output file path - use temp directory in project
    if output_file is None:
        temp_dir = project_root / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        if match_id:
            output_file = temp_dir / f"match_{match_id}.json"
        else:
            # Use timestamp if no match ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = temp_dir / f"match_{timestamp}.json"
    
    # Fetch historical data (for all maps if map not decided)
    fetched_data = {
        "match_url": match_url,
        "match_id": match_id,
        "team_a": team_a,
        "team_b": team_b,
        "map": map_name,
        "match_date": match_date,
        "fetched_at": datetime.now().isoformat(),
        "historical_data": {}
    }
    
    if map_name:
        # Map is decided - fetch data for this map only
        if verbose:
            logger.info(f"Fetching historical data for {map_name}...")
        
        historical_data = fetch_historical_data_for_match(
            match_url=match_url,
            team_a_name=team_a,
            team_b_name=team_b,
            map_name=map_name,
            match_date=match_date,
            verbose=verbose
        )
        
        if historical_data:
            fetched_data["historical_data"][map_name] = {
                "team_results": historical_data.get("team_results", []),
                "head_to_head": historical_data.get("head_to_head", []),
                "team_a_id": historical_data.get("team_a_id"),
                "team_b_id": historical_data.get("team_b_id"),
                "team_a_slug": historical_data.get("team_a_slug"),
                "team_b_slug": historical_data.get("team_b_slug"),
                "map_id": historical_data.get("map_id"),
            }
    else:
        # Map not decided - fetch data for all maps
        if verbose:
            logger.info("Map not decided. Fetching historical data for all maps...")
        
        all_maps = ["Mirage", "Inferno", "Dust2", "Nuke", "Overpass", "Vertigo", "Ancient", "Anubis", "Train"]
        
        for idx, map_name_iter in enumerate(all_maps):
            if verbose:
                logger.info(f"Fetching data for {map_name_iter} ({idx+1}/{len(all_maps)})...")
            
            historical_data = fetch_historical_data_for_match(
                match_url=match_url,
                team_a_name=team_a,
                team_b_name=team_b,
                map_name=map_name_iter,
                match_date=match_date,
                verbose=False  # Less verbose for multiple maps
            )
            
            if historical_data:
                fetched_data["historical_data"][map_name_iter] = {
                    "team_results": historical_data.get("team_results", []),
                    "head_to_head": historical_data.get("head_to_head", []),
                    "team_a_id": historical_data.get("team_a_id"),
                    "team_b_id": historical_data.get("team_b_id"),
                    "team_a_slug": historical_data.get("team_a_slug"),
                    "team_b_slug": historical_data.get("team_b_slug"),
                    "map_id": historical_data.get("map_id"),
                }
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(fetched_data, f, indent=2, default=str)
    
    if verbose:
        logger.info(f"Saved fetched data to temporary file: {output_file}")
        logger.info(f"Found historical data for {len(fetched_data['historical_data'])} map(s)")
        logger.info(f"Note: This file is stored in a temporary directory and may be cleaned up by the system")
    
    # Add file path to returned data for convenience
    fetched_data['_file_path'] = str(output_file)
    
    return fetched_data

