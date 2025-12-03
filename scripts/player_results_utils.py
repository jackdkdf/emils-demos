# player_results_utils.py

import datetime
from bs4 import BeautifulSoup
import re
import time

# Helper dictionary for map names
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


def get_weekly_intervals(start_str, end_str):
    """Generates a list of (start_date, end_date) strings, each 7 days apart."""
    try:
        start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        print("Error: Date format must be YYYY-MM-DD")
        return []

    intervals = []
    current = start_date
    while current < end_date:
        next_week = current + datetime.timedelta(days=7)
        if next_week > end_date:
            break

        s_str = current.strftime("%Y-%m-%d")
        e_str = next_week.strftime("%Y-%m-%d")
        intervals.append((s_str, e_str))
        current = next_week

    return intervals


def format_map_for_url(map_name):
    """Converts 'Ancient' -> 'de_ancient' for HLTV URL parameters."""
    clean = map_name.lower().strip()
    if not clean.startswith("de_"):
        return f"de_{clean}"
    return clean


def scrape_player_stats_dict(scraper, url, possible_keywords):
    """
    Fetches a URL and returns a DICTIONARY of player stats.
    Format: { "PlayerName": float_value }
    """
    try:
        resp = scraper.get(url)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")

        if "Just a moment..." in soup.get_text():
            print(f"      [!] Cloudflare Blocked: {url}")
            return None

        # Matches class="stats-table" even if other classes like "player-ratings-table" exist
        table = soup.find("table", class_="stats-table")
        if not table:
            return None

        # 1. Find Header Index
        thead = table.find("thead")
        if not thead:
            return None

        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        target_idx = -1

        for i, text in enumerate(headers):
            # Checks if any keyword (e.g., "Succ") is inside the header text (e.g., "Success")
            if any(keyword in text for keyword in possible_keywords):
                target_idx = i
                break

        if target_idx == -1:
            return None

        # 2. Extract Data for each player
        player_data = {}
        rows = table.find("tbody").find_all("tr")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) <= target_idx:
                continue

            # Find player name (usually in the first column class="playerCol" or "playerColSmall")
            # We look for the cell containing the player link/flag
            player_col = row.find("td", class_=re.compile("playerCol"))
            if not player_col:
                continue

            player_name = player_col.get_text(strip=True)

            val_str = cols[target_idx].get_text(strip=True)
            val_str = val_str.replace("%", "")

            if val_str in ["-", ""]:
                player_data[player_name] = 0.0
                continue

            try:
                player_data[player_name] = float(val_str)
            except ValueError:
                player_data[player_name] = 0.0

        if not player_data:
            return None

        return player_data

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
