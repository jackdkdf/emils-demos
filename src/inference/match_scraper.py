"""Scraper for extracting match information from HLTV match pages."""

import logging
import re
import time
import random
import cloudscraper
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def scrape_match_info(match_url: str) -> Optional[Dict[str, str]]:
    """Scrape match information from an HLTV match page.
    
    Args:
        match_url: URL to the HLTV match page (e.g., https://www.hltv.org/matches/2388125/spirit-vs-falcons-...)
        
    Returns:
        Dictionary with match information:
        - team_a: First team name
        - team_b: Second team name  
        - map: Map name
        - match_date: Match date in YYYY-MM-DD format
        - match_id: Match ID from URL
    """
    try:
        scraper = cloudscraper.create_scraper()
        # Small delay to prevent rate limiting
        time.sleep(random.uniform(0.5, 1.5))
        response = scraper.get(match_url)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch match page: {e}")
        return None
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract match ID from URL
    match_id_match = re.search(r'/matches/(\d+)/', match_url)
    match_id = match_id_match.group(1) if match_id_match else None
    
    # Extract team names from the page
    # Look for team names in various possible locations
    team_a = None
    team_b = None
    
    # Try to find team names in the match header
    team_elements = soup.find_all("div", class_=re.compile(r".*team.*", re.I))
    if not team_elements:
        # Try alternative selectors
        team_elements = soup.find_all("a", href=re.compile(r"/team/\d+/"))
    
    team_names = []
    for elem in team_elements:
        # Look for team name in text or span
        name_elem = elem.find("span") or elem
        name = name_elem.get_text(strip=True)
        if name and len(name) > 2:  # Filter out very short strings
            # Check if it's a team link
            href = elem.get("href", "")
            if "/team/" in href:
                # Get the actual team name from the link text, not parent elements
                # This avoids picking up event names or other text
                link_text = name_elem.get_text(strip=True)
                if link_text:
                    team_names.append(link_text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_teams = []
    for name in team_names:
        # Filter out very long names (likely event names, not team names)
        if len(name) > 50:
            continue
        name_lower = name.lower()
        if name_lower not in seen:
            seen.add(name_lower)
            unique_teams.append(name)
    
    if len(unique_teams) >= 2:
        team_a = unique_teams[0]
        team_b = unique_teams[1]
    else:
        # Fallback: try to extract from URL slug
        url_slug_match = re.search(r'/([\w-]+)-vs-([\w-]+)-', match_url)
        if url_slug_match:
            team_a_slug = url_slug_match.group(1)
            team_b_slug = url_slug_match.group(2)
            # Convert slug to readable name (basic conversion)
            team_a = team_a_slug.replace('-', ' ').title()
            team_b = team_b_slug.replace('-', ' ').title()
    
    # Extract map name - only from actual map pick/ban section, not from random text
    map_name = None
    known_maps = ["Mirage", "Inferno", "Dust2", "Nuke", "Overpass", "Vertigo", "Ancient", "Anubis", "Train"]
    
    # Look for map in the map pick/ban section (more reliable)
    # HLTV typically shows maps in a specific section for match pages
    map_section = soup.find("div", class_=re.compile(r".*map.*pick.*", re.I))
    if not map_section:
        map_section = soup.find("div", class_=re.compile(r".*map.*ban.*", re.I))
    if not map_section:
        # Look for map in match details box
        map_section = soup.find("div", class_=re.compile(r".*match.*info.*", re.I))
    
    if map_section:
        map_text = map_section.get_text()
        for known_map in known_maps:
            if known_map.lower() in map_text.lower():
                # Check if it's actually selected/played (not just mentioned)
                # Look for context like "picked", "selected", or in a results section
                if any(indicator in map_text.lower() for indicator in ["picked", "selected", "played", "result"]):
                    map_name = known_map
                    break
    
    # If still not found, check if match has been played (has results)
    # If match hasn't been played yet, map might not be decided
    if not map_name:
        # Check for match results section - if it exists, map should be there
        results_section = soup.find("div", class_=re.compile(r".*result.*", re.I))
        if results_section:
            results_text = results_section.get_text()
            for known_map in known_maps:
                if known_map.lower() in results_text.lower():
                    map_name = known_map
                    break
    
    # Extract match date - look for date in match info section
    match_date = None
    
    # Look for date in the match info/time section (more reliable)
    date_section = soup.find("div", class_=re.compile(r".*date.*", re.I))
    if not date_section:
        date_section = soup.find("div", class_=re.compile(r".*time.*", re.I))
    if not date_section:
        date_section = soup.find("span", class_=re.compile(r".*date.*", re.I))
    
    if date_section:
        date_text = date_section.get_text(strip=True)
        # Try to parse date from this section
        date_patterns = [
            (r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", ["%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y"]),
            (r"(\d{4}-\d{2}-\d{2})", ["%Y-%m-%d"]),
            (r"(\d{1,2}\s+\w+\s+\d{4})", None),  # e.g., "18 Nov 2025"
        ]
        
        for pattern, formats in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                date_str = match.group(1)
                if formats:
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            match_date = dt.strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
                else:
                    # Try parsing with dateutil or common formats
                    try:
                        from dateutil import parser
                        dt = parser.parse(date_str)
                        match_date = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                if match_date:
                    break
    
    # If still not found, look for date strings in the page
    if not match_date:
        date_elements = soup.find_all(string=re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"))
        if date_elements:
            date_text = date_elements[0].strip()
            try:
                for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y"]:
                    try:
                        dt = datetime.strptime(date_text, fmt)
                        match_date = dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
    
    # If still no date, use today's date as fallback
    if not match_date:
        match_date = datetime.now().strftime("%Y-%m-%d")
        logger.warning(f"Could not extract match date from page, using today's date: {match_date}")
    
    if not team_a or not team_b:
        logger.error(f"Could not extract team names from match page")
        return None
    
    if not map_name:
        logger.warning(f"Could not extract map name - match may not have a map selected yet")
        logger.info("Note: If the map hasn't been decided, you can specify it manually with --map")
    
    result = {
        "team_a": team_a,
        "team_b": team_b,
        "map": map_name,
        "match_date": match_date,
        "match_id": match_id,
    }
    
    logger.info(f"Extracted match info: {team_a} vs {team_b} on {map_name} ({match_date})")
    
    return result

