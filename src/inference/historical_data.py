"""Fetch historical data from HLTV for teams in a match."""

import logging
import re
import time
import random
import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Map IDs to names (from map_stats_utils.py)
MAP_ID_TO_NAME = {
    "32": "Mirage",
    "33": "Inferno",
    "31": "Dust2",
    "34": "Nuke",
    "40": "Overpass",
    "46": "Vertigo",
    "47": "Ancient",
    "48": "Anubis",
    "35": "Train",
}

NAME_TO_MAP_ID = {v: k for k, v in MAP_ID_TO_NAME.items()}


def extract_team_ids_from_match_page(match_url: str) -> Optional[Dict[str, str]]:
    """Extract team IDs and slugs from a match page.
    
    Args:
        match_url: HLTV match URL
        
    Returns:
        Dictionary with team_a_id, team_a_slug, team_b_id, team_b_slug, or None if failed
    """
    try:
        scraper = cloudscraper.create_scraper()
        # Small delay before fetching match page
        time.sleep(random.uniform(0.5, 1.5))
        response = scraper.get(match_url)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch match page: {e}")
        return None
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    team_info = {}
    seen_ids = set()
    
    # Try multiple strategies to find team links
    # Strategy 1: Look for team links in various locations
    team_links = soup.find_all("a", href=re.compile(r"/team/(\d+)/"))
    
    # Strategy 2: Look for team names in team name elements
    team_name_elements = soup.find_all(["div", "span"], class_=re.compile(r".*team.*name.*", re.I))
    
    # Strategy 3: Look in the match header area
    match_header = soup.find("div", class_=re.compile(r".*match.*header.*", re.I))
    if match_header:
        team_links.extend(match_header.find_all("a", href=re.compile(r"/team/(\d+)/")))
    
    # Strategy 4: Look for team logos/links
    team_logos = soup.find_all("img", alt=re.compile(r".*logo.*", re.I))
    for logo in team_logos:
        parent_link = logo.find_parent("a", href=re.compile(r"/team/(\d+)/"))
        if parent_link:
            team_links.append(parent_link)
    
    # Extract unique team IDs
    for link in team_links:
        href = link.get("href", "")
        match = re.search(r"/team/(\d+)/([\w-]+)", href)
        if match:
            team_id = match.group(1)
            team_slug = match.group(2)
            
            if team_id not in seen_ids and len(team_info) < 4:
                seen_ids.add(team_id)
                if "team_a_id" not in team_info:
                    team_info["team_a_id"] = team_id
                    team_info["team_a_slug"] = team_slug
                elif "team_b_id" not in team_info:
                    team_info["team_b_id"] = team_id
                    team_info["team_b_slug"] = team_slug
                    break
    
    # If we still don't have both teams, try extracting from the URL slug
    if len(team_info) < 4:
        url_match = re.search(r'/([\w-]+)-vs-([\w-]+)-', match_url)
        if url_match:
            team_a_slug = url_match.group(1)
            team_b_slug = url_match.group(2)
            
            # Try to find team IDs by searching for these slugs in team links
            all_team_links = soup.find_all("a", href=re.compile(r"/team/(\d+)/"))
            for link in all_team_links:
                href = link.get("href", "")
                match = re.search(r"/team/(\d+)/([\w-]+)", href)
                if match:
                    team_id = match.group(1)
                    slug = match.group(2)
                    
                    if slug == team_a_slug and "team_a_id" not in team_info:
                        team_info["team_a_id"] = team_id
                        team_info["team_a_slug"] = slug
                    elif slug == team_b_slug and "team_b_id" not in team_info:
                        team_info["team_b_id"] = team_id
                        team_info["team_b_slug"] = slug
                    
                    if len(team_info) >= 4:
                        break
    
    if len(team_info) < 4:
        logger.warning(f"Could not extract both team IDs from match page. Found: {list(team_info.keys())}")
        logger.debug(f"Found {len(team_links)} team links on the page")
        # Try one more time with a different approach - look for all links and extract IDs
        if len(team_links) > 0:
            logger.debug("Trying alternative extraction method...")
            for link in team_links[:10]:  # Check first 10 links
                href = link.get("href", "")
                match = re.search(r"/team/(\d+)/([\w-]+)", href)
                if match:
                    team_id = match.group(1)
                    team_slug = match.group(2)
                    if team_id not in seen_ids:
                        seen_ids.add(team_id)
                        if "team_a_id" not in team_info:
                            team_info["team_a_id"] = team_id
                            team_info["team_a_slug"] = team_slug
                        elif "team_b_id" not in team_info:
                            team_info["team_b_id"] = team_id
                            team_info["team_b_slug"] = team_slug
                            break
        
        if len(team_info) < 4:
            return None
    
    return team_info


def search_team_ids_by_name(team_a_name: str, team_b_name: str, match_url: str) -> Optional[Dict]:
    """Search for team IDs by searching HLTV team pages.
    
    This is a fallback method when direct extraction fails.
    """
    scraper = cloudscraper.create_scraper()
    team_info = {}
    
    # Try to extract team slugs from URL
    url_match = re.search(r'/([\w-]+)-vs-([\w-]+)-', match_url)
    if url_match:
        team_a_slug = url_match.group(1)
        team_b_slug = url_match.group(2)
        
        # Try to get team IDs by visiting team pages (they redirect to /team/{id}/{slug})
        for slug, team_key in [(team_a_slug, "team_a"), (team_b_slug, "team_b")]:
            try:
                team_url = f"https://www.hltv.org/team/{slug}"
                # Add delay between team page requests
                time.sleep(random.uniform(1.0, 2.0))
                response = scraper.get(team_url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    # Extract team ID from final URL (after redirect)
                    final_url = response.url
                    id_match = re.search(r'/team/(\d+)/', final_url)
                    if id_match:
                        team_id = id_match.group(1)
                        team_info[f"{team_key}_id"] = team_id
                        team_info[f"{team_key}_slug"] = slug
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Found {team_key}: ID={team_id}, slug={slug} via team page")
            except Exception as e:
                logger.debug(f"Failed to fetch team page for {slug}: {e}")
                continue
    
    if len(team_info) >= 4:
        return team_info
    
    return None


def scrape_team_map_stats(
    team_id: str,
    team_slug: str,
    map_id: str,
    match_date: str,
    scraper: Optional[cloudscraper.CloudScraper] = None
) -> List[Dict]:
    """Scrape historical map statistics for a team up to a given date.
    
    Args:
        team_id: Team ID
        team_slug: Team slug
        map_id: Map ID
        match_date: Match date (YYYY-MM-DD) - only get stats before this date
        scraper: Optional cloudscraper instance
        
    Returns:
        List of match dictionaries
    """
    if scraper is None:
        scraper = cloudscraper.create_scraper()
    
    # Calculate date range (go back 1 year from match date)
    match_date_dt = datetime.strptime(match_date, "%Y-%m-%d")
    start_date = (match_date_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = (match_date_dt - timedelta(days=1)).strftime("%Y-%m-%d")  # Up to day before match
    
    url = (
        f"https://www.hltv.org/stats/teams/map/{map_id}/{team_id}/{team_slug}"
        f"?startDate={start_date}&endDate={end_date}"
    )
    
    try:
        # Add random delay to prevent rate limiting (1-3 seconds)
        time.sleep(random.uniform(1.0, 3.0))
        response = scraper.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.debug(f"Failed to fetch map stats page for team {team_id} on map {map_id}: {e}")
        return []
    
    soup = BeautifulSoup(response.content, "html.parser")
    matches = []
    
    # Strategy 1: Look for standard "Matches" headline with table
    headline_span = soup.find("span", class_="standard-headline", string=re.compile(r"\s*Matches\s*"))
    stats_table = None
    
    if headline_span:
        stats_table = headline_span.parent.find_next_sibling("table", class_="stats-table")
    
    # Strategy 2: If no headline, look for any stats-table
    if not stats_table:
        all_tables = soup.find_all("table", class_="stats-table")
        for table in all_tables:
            # Check if it looks like a matches table (has date/opponent columns)
            header_text = table.get_text().lower()
            if any(keyword in header_text for keyword in ["date", "opponent", "event", "result"]):
                stats_table = table
                break
    
    # Strategy 3: Look for table with match data by checking for score patterns
    if not stats_table:
        all_tables = soup.find_all("table")
        for table in all_tables:
            table_text = table.get_text()
            # Check if it contains score patterns like "16-14" or "2-1"
            if re.search(r'\d+\s*-\s*\d+', table_text):
                # Check if it has multiple rows (likely matches)
                rows = table.find_all("tr")
                if len(rows) > 1:  # More than just header
                    stats_table = table
                    break
    
    if not stats_table:
        logger.debug(f"Could not find matches table for team {team_id} on map {map_id} - page may have no matches or different structure")
        return []
    
    # Extract match rows
    match_rows = stats_table.find("tbody")
    if match_rows:
        match_rows = match_rows.find_all("tr")
    else:
        # Rows might be directly in table (skip header row)
        match_rows = stats_table.find_all("tr")[1:]  # Skip first row (header)
    
    for row in match_rows:
        try:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            
            # Parse date
            date_text = cells[0].get_text(strip=True)
            try:
                dt = datetime.strptime(date_text, "%d/%m/%y")
                match_date_str = dt.strftime("%Y-%m-%d")
            except:
                continue
            
            # Parse opponent
            opponent_name = "N/A"
            opponent_id = "N/A"
            opponent_link = cells[1].find("a")
            if opponent_link:
                opponent_name = opponent_link.find("span").get_text(strip=True)
                opponent_id_match = re.search(r"/stats/teams/(\d+)/", opponent_link["href"])
                if opponent_id_match:
                    opponent_id = opponent_id_match.group(1)
            
            # Parse result
            score_text = cells[3].get_text(strip=True)
            score_match = re.search(r"(\d+)\s*-\s*(\d+)", score_text)
            score_us = score_match.group(1) if score_match else "N/A"
            score_them = score_match.group(2) if score_match else "N/A"
            
            cell_classes = cells[3].get("class", [])
            result = "L" if "match-lost" in cell_classes else "W"
            
            matches.append({
                "team_name": team_slug.replace("-", " ").title(),
                "team_id": team_id,
                "map_name": MAP_ID_TO_NAME.get(map_id, "Unknown"),
                "map_id": map_id,
                "match_date": match_date_str,
                "opponent_name": opponent_name,
                "opponent_id": opponent_id,
                "score_us": score_us,
                "score_them": score_them,
                "result": result,
            })
        except Exception as e:
            logger.debug(f"Error parsing match row for team {team_id} on map {map_id}: {e}")
            continue
    
    # Log if we found table but no matches (vs no table at all)
    if stats_table and len(matches) == 0:
        logger.debug(f"Found matches table for team {team_id} on map {map_id} but no valid matches parsed")
    
    return matches


def fetch_historical_data_for_match(
    match_url: str,
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    match_date: str,
    verbose: bool = True
) -> Optional[Dict]:
    """Fetch all historical data needed for prediction from HLTV.
    
    Args:
        match_url: HLTV match URL
        team_a_name: Name of team A
        team_b_name: Name of team B
        map_name: Map name
        match_date: Match date (YYYY-MM-DD)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with:
        - team_results: List of historical match results
        - team_a_id: Team A ID
        - team_b_id: Team B ID
        - map_id: Map ID
    """
    if verbose:
        logger.info("Fetching team IDs from match page...")
    
    # Extract team IDs from match page
    team_info = extract_team_ids_from_match_page(match_url)
    
    # If extraction failed, try to search for teams by name
    if not team_info:
        if verbose:
            logger.warning("Could not extract team IDs directly, trying to search by team names...")
        team_info = search_team_ids_by_name(team_a_name, team_b_name, match_url)
    
    if not team_info:
        logger.error("Could not extract team IDs from match page")
        logger.error("You may need to manually specify team IDs or use preprocessed data")
        return None
    
    team_a_id = team_info["team_a_id"]
    team_a_slug = team_info["team_a_slug"]
    team_b_id = team_info["team_b_id"]
    team_b_slug = team_info["team_b_slug"]
    
    # Get map ID
    map_id = NAME_TO_MAP_ID.get(map_name)
    if not map_id:
        logger.warning(f"Unknown map: {map_name}, using map_id=0")
        map_id = "0"
    
    if verbose:
        logger.info(f"Team A: {team_a_name} (ID: {team_a_id})")
        logger.info(f"Team B: {team_b_name} (ID: {team_b_id})")
        logger.info(f"Map: {map_name} (ID: {map_id})")
        logger.info("Fetching historical match data from HLTV...")
    
    # Scrape historical data for both teams on this map
    scraper = cloudscraper.create_scraper()
    
    team_a_matches = scrape_team_map_stats(team_a_id, team_a_slug, map_id, match_date, scraper)
    # Small delay between team requests
    time.sleep(random.uniform(1.0, 2.0))
    team_b_matches = scrape_team_map_stats(team_b_id, team_b_slug, map_id, match_date, scraper)
    
    if verbose:
        logger.info(f"Found {len(team_a_matches)} historical matches for {team_a_name} on {map_name}")
        logger.info(f"Found {len(team_b_matches)} historical matches for {team_b_name} on {map_name}")
    
    # Also get matches between the two teams (need to check opponent in team_a_matches)
    head_to_head = [
        m for m in team_a_matches 
        if m.get("opponent_id") == team_b_id
    ]
    
    if verbose:
        logger.info(f"Found {len(head_to_head)} head-to-head matches on {map_name}")
    
    # Return data even if empty - empty list means no historical data, but we can still predict
    # The prediction will use default/zero values for features
    return {
        "team_results": team_a_matches + team_b_matches,
        "head_to_head": head_to_head,
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "team_a_slug": team_a_slug,
        "team_b_slug": team_b_slug,
        "map_id": map_id,
        "data_found": len(team_a_matches) > 0 or len(team_b_matches) > 0,  # Track if we actually found data
    }

