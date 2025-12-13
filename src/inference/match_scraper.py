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
    
    # Strategy 1: Look for team links with specific structure
    team_links = soup.find_all("a", href=re.compile(r"/team/\d+/"))
    team_names = []
    
    for link in team_links:
        href = link.get("href", "")
        # Extract team slug from URL (e.g., /team/11283/falcons -> "falcons")
        slug_match = re.search(r"/team/\d+/([\w-]+)", href)
        if slug_match:
            team_slug = slug_match.group(1)
            
            # Try to find the team name in the link's direct children
            # Look for span with class containing "name" or just the first text node
            team_name = None
            
            # Check for span with name class
            name_span = link.find("span", class_=re.compile(r".*name.*", re.I))
            if name_span:
                team_name = name_span.get_text(strip=True)
            else:
                # Get direct text content, avoiding nested elements
                # Get all text nodes directly under the link
                text_parts = []
                for child in link.children:
                    if isinstance(child, str):
                        text_parts.append(child.strip())
                    elif child.name == "span":
                        span_text = child.get_text(strip=True)
                        if span_text:
                            text_parts.append(span_text)
                
                if text_parts:
                    # Take the first non-empty text part
                    team_name = text_parts[0] if text_parts[0] else (text_parts[1] if len(text_parts) > 1 else None)
            
            # If we found a name, clean it up
            if team_name and len(team_name) > 2:
                # Remove event-related text that might be concatenated
                # Common patterns: "TeamName EventName" or "TeamName - EventName"
                event_keywords = ["major", "championship", "cup", "league", "tournament", "starladder", "iem", "blast", "budapest", "cologne", "katowice"]
                name_lower = team_name.lower()
                
                # Check if name contains event keywords - if so, try to extract just the team name
                for keyword in event_keywords:
                    if keyword in name_lower:
                        # Try to split on common separators or take the first part
                        # Common patterns: "Falcons Starladder" -> "Falcons"
                        parts = re.split(r'\s+-\s+|\s+', team_name)
                        if len(parts) > 1:
                            # Take the first part that doesn't contain event keywords
                            for part in parts:
                                part_lower = part.lower()
                                if not any(ek in part_lower for ek in event_keywords) and len(part) > 2:
                                    team_name = part
                                    break
                        break
                
                # Final validation: reasonable length and no event keywords
                if len(team_name) < 50 and not any(keyword in team_name.lower() for keyword in event_keywords):
                    team_names.append((team_name, team_slug))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_teams = []
    for name, slug in team_names:
        name_lower = name.lower()
        if name_lower not in seen:
            seen.add(name_lower)
            unique_teams.append((name, slug))
    
    if len(unique_teams) >= 2:
        team_a = unique_teams[0][0]
        team_b = unique_teams[1][0]
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

