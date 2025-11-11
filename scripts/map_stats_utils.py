# map_stats_utils.py

import csv
import os
import re
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import requests.exceptions

# A helper dictionary for map names. You'll need to fill this in by
# looking at the URLs. e.g., /map/47/... -> Nuke
# In map_stats_utils.py

# A helper dictionary for map names.
MAP_ID_TO_NAME = {
    "32": "Mirage",
    "33": "Inferno",
    "31": "Dust2",
    "34": "Nuke",  #
    "40": "Overpass",
    "46": "Vertigo",
    "47": "Ancient",  #
    "48": "Anubis",
    "35": "Train",
    # Add more as you find them
}


def parse_url_for_info(url):
    """
    Parses the URL to get Team ID, Team Name, Map ID, and Map Name.
    URL format: /stats/teams/map/{MAP_ID}/{TEAM_ID}/{TEAM_NAME}
    """
    match = re.search(r"/map/(\d+)/(\d+)/([\w-]+)", url)
    if not match:
        print(f"Error: Could not parse URL {url}")
        return None

    map_id, team_id, team_name = match.groups()
    map_name = MAP_ID_TO_NAME.get(map_id, f"UnknownMap({map_id})")

    return {
        "map_id": map_id,
        "team_id": team_id,
        "team_name": team_name,
        "map_name": map_name,
    }


def load_existing_map_rows(csv_file):
    """
    Reads the map stats CSV and returns a set of (team, opponent, map, date) tuples
    to prevent duplicate entries.
    """
    existing_rows = set()
    if not os.path.isfile(csv_file):
        return existing_rows  # File doesn't exist

    try:
        with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
            if os.path.getsize(csv_file) == 0:
                return existing_rows

            reader = csv.DictReader(file)
            for row in reader:
                if (
                    "team_name" not in row
                    or "opponent_name" not in row
                    or "map_name" not in row
                    or "match_date" not in row
                ):
                    print(
                        "Warning: CSV file is missing required columns. Skipping duplicate check."
                    )
                    return set()

                existing_rows.add(
                    (
                        row["team_name"],
                        row["opponent_name"],
                        row["map_name"],
                        row["match_date"],
                    )
                )

    except IOError as e:
        print(f"Warning: Could not read {csv_file} for duplicate checking: {e}")
    except csv.Error as e:
        print(f"Warning: CSV error in {csv_file}. File might be corrupt: {e}")

    return existing_rows


def scrape_map_stats_page(scraper, url, base_info):
    """
    Fetches a single map stats page and scrapes all match data from the table.
    Returns a list of match_data dictionaries.

    NOTE: This function is now corrected based on the debug_page.html provided.
    """
    try:
        response = scraper.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"  HTTP error occurred: {http_err}. Status Code: {response.status_code}")
        return []
    except Exception as e:
        print(f"  An unexpected error occurred: {e}. Skipping.")
        return []

    print("  Successfully fetched the page. Parsing HTML...")
    soup = BeautifulSoup(response.content, "html.parser")

    all_matches_data = []

    # 1. Find the "Matches" headline
    #    We use re.compile to ignore leading/trailing whitespace
    headline_span = soup.find(
        "span", class_="standard-headline", string=re.compile(r"\s*Matches\s*")
    )

    if not headline_span:
        print("  Error: Could not find the 'Matches' headline span.")
        print("  This means the page structure may have changed. Scraping failed.")
        return []

    # 2. Find the table. It's the next sibling of the headline's parent div.
    stats_table = headline_span.parent.find_next_sibling("table", class_="stats-table")

    if not stats_table:
        print(
            "  Error: Found 'Matches' headline, but no 'stats-table' table immediately after it."
        )
        print("  This means the page structure may have changed. Scraping failed.")
        return []

    # 3. Find all match rows in the table body
    match_rows = stats_table.find("tbody").find_all("tr")
    print(f"  Found {len(match_rows)} match row(s) in the table.")

    for row in match_rows:
        try:
            cells = row.find_all("td")
            if len(cells) < 4:  # Expecting Date, Opponent, Event, Result
                continue

            # --- Parse each cell (Selectors are now corrected) ---

            # Cell 0: Date (e.g., "09/11/25")
            match_date = "N/A"
            try:
                date_text = cells[0].get_text(strip=True)
                dt = datetime.strptime(date_text, "%d/%m/%y")
                match_date = dt.strftime("%Y-%m-%d")  # Formats to 2025-11-09
            except Exception as e:
                print(f"    Warning: Could not parse date '{date_text}': {e}")

            # Cell 1: Opponent
            opponent_name, opponent_id = "N/A", "N/A"
            opponent_link = cells[1].find("a")
            if opponent_link:
                opponent_name = opponent_link.find("span").get_text(strip=True)
                opponent_id_match = re.search(
                    r"/stats/teams/(\d+)/", opponent_link["href"]
                )
                if opponent_id_match:
                    opponent_id = opponent_id_match.group(1)

            # Cell 2: Event
            event_name, event_id = "N/A", "N/A"
            event_link = cells[2].find("a")
            if event_link:
                event_name = event_link.find("span").get_text(strip=True)
                # Event ID is in a query parameter, e.g., &event=8041
                event_id_match = re.search(r"&event=(\d+)", event_link["href"])
                if event_id_match:
                    event_id = event_id_match.group(1)

            # Cell 3: Result (Score and W/L)
            score_us, score_them, result = "N/A", "N/A", "N/A"
            score_text = cells[3].get_text(strip=True)
            score_match = re.search(r"(\d+)\s*-\s*(\d+)", score_text)
            if score_match:
                score_us = score_match.group(1)
                score_them = score_match.group(2)

            # Get W/L from the cell's class
            cell_classes = cells[3].get("class", [])
            if "match-lost" in cell_classes:
                result = "L"
            elif "match-won" in cell_classes:  # Guessed class name
                result = "W"
            elif "match-draw" in cell_classes:  # Guessed class name
                result = "T"

            # --- Assemble the data ---
            match_data = {
                "team_name": base_info["team_name"],
                "team_id": base_info["team_id"],
                "map_name": base_info["map_name"],
                "map_id": base_info["map_id"],
                "match_date": match_date,
                "opponent_name": opponent_name,
                "opponent_id": opponent_id,
                "score_us": score_us,
                "score_them": score_them,
                "result": result,
                "event_name": event_name,
                "event_id": event_id,
            }
            all_matches_data.append(match_data)

        except Exception as e:
            print(f"  Error parsing a match row: {e}. Skipping this row.")

    return all_matches_data
