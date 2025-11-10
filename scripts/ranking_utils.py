import csv
import os
import re
import time
import sys
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests.exceptions


def generate_ranking_urls(start_date, end_date):
    """
    Generates a list of HLTV ranking URLs for every Monday
    between start_date and end_date (inclusive).

    HLTV URLs are formatted as: .../YYYY/month_name/day
    """
    if end_date > start_date:
        print("Error: END_DATE must be older than START_DATE.")
        return []

    print(
        f"Generating URLs from {end_date.strftime('%Y-%m-%d')} to {start_date.strftime('%Y-%m-%d')}"
    )

    urls = []
    current_date = start_date

    # Loop backwards by weeks
    while current_date >= end_date:
        # Format the URL
        year = current_date.strftime("%Y")
        month_name = current_date.strftime("%B").lower()  # e.g., 'november'
        day = str(current_date.day)  # e.g., '3'

        url = f"https://www.hltv.org/ranking/teams/{year}/{month_name}/{day}"
        urls.append(url)

        # Move to the previous Monday
        current_date -= timedelta(weeks=1)

    print(f"Generated {len(urls)} weekly URLs.")
    return urls


def load_existing_rows(csv_file):
    """
    Reads the main rankings CSV and returns a set of (name, date) tuples
    to prevent duplicate entries.
    """
    existing_rows = set()
    if not os.path.isfile(csv_file):
        return existing_rows  # File doesn't exist, so no existing rows

    try:
        with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
            # Check if file is empty
            if os.path.getsize(csv_file) == 0:
                return existing_rows

            reader = csv.DictReader(file)
            for row in reader:
                # Handle case where file exists but header is corrupt
                if "name" not in row or "date" not in row:
                    print(
                        "Warning: CSV file is missing 'name' or 'date' columns. Skipping duplicate check."
                    )
                    return set()  # Return empty set to be safe

                existing_rows.add((row["name"], row["date"]))

    except IOError as e:
        print(f"Warning: Could not read {csv_file} for duplicate checking: {e}")
    except csv.Error as e:
        print(f"Warning: CSV error in {csv_file}. File might be corrupt: {e}")

    return existing_rows


def parse_ranking_date(soup, url):
    """
    Tries to find the ranking date from the page.
    Falls back to parsing the URL if not found.
    """
    try:
        # 1. Try to find it on the page
        date_element = soup.find("div", class_="ranking-week")
        if date_element and date_element.find("span"):
            date_text = date_element.find("span").get_text(strip=True)
            # Format: "Week of November 3rd 2025"
            # We need to parse this into "2025-11-03"

            # Remove "Week of "
            date_text = date_text.replace("Week of ", "")

            # Clean ordinals (th, st, rd, nd)
            date_text_cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_text)

            # Parse the cleaned string
            # Format: "November 3 2025"
            dt = datetime.strptime(date_text_cleaned, "%B %d %Y")
            return dt.strftime("%Y-%m-%d")

    except Exception:
        # If parsing fails, fall through to URL parsing
        pass

    # 2. Fallback: Parse from URL
    # URL: .../YYYY/month_name/day
    try:
        match = re.search(r"/ranking/teams/(\d{4})/(\w+)/(\d+)", url)
        if match:
            year, month_name, day = match.groups()
            date_str = f"{year} {month_name.capitalize()} {day}"
            dt = datetime.strptime(date_str, "%Y %B %d")
            return dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"  Warning: Could not parse date from URL {url}: {e}")

    # 3. Last fallback: today's date
    print("  Could not find or parse date element, using current date as fallback.")
    return datetime.now().strftime("%Y-%m-%d")


def scrape_ranking_page(scraper, url):
    """
    Fetches a single ranking page and scrapes all team data.
    Returns (all_teams_data, ranking_date).
    """
    try:
        response = scraper.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"  HTTP error occurred: {http_err}. Status Code: {response.status_code}")
        if response.status_code == 404:
            print(
                "  Page not found (404). This date might not have a ranking. Skipping."
            )
        else:
            print(f"  HLTV might be blocking the request for {url}.")
        return [], None
    except Exception as e:
        print(f"  An unexpected error occurred: {e}. Skipping.")
        return [], None

    print("  Successfully fetched the page. Parsing HTML...")
    soup = BeautifulSoup(response.content, "html.parser")

    # Get the date for this ranking
    ranking_date = parse_ranking_date(soup, url)

    all_teams_data = []

    # Find all ranked team elements
    ranked_teams = soup.find_all("div", class_="ranked-team")

    for team_div in ranked_teams:
        rank, name, points, team_id, team_slug = "N/A", "N/A", "0", "N/A", "N/A"

        try:
            # 1. Get Rank
            rank_element = team_div.find("span", class_="position")
            if rank_element:
                rank = rank_element.get_text(strip=True)  # e.g., "#1"

            # 2. Get Name
            name_element = team_div.find("span", class_="name")
            if name_element:
                name = name_element.get_text(strip=True)

            # 3. Get Points
            points_element = team_div.find("span", class_="points")
            if points_element:
                points_text = points_element.get_text(strip=True)
                # Use regex to find the first number in the string (e.g., "950")
                match = re.search(r"(\d+)", points_text)
                if match:
                    points = match.group(1)

            # 4. Get Team ID and Slug
            #    This comes from the <a> tag href
            link_element = team_div.find(
                "a", class_="moreLink", href=re.compile(r"/team/")
            )
            if link_element:
                href = link_element["href"]
                # Format: /team/9565/vitality
                match = re.search(r"/team/(\d+)/([\w-]+)", href)
                if match:
                    team_id = match.group(1)
                    team_slug = match.group(2)

            # Only add if we found a valid team
            if name != "N/A" and rank != "N/A":
                team_data = {
                    "rank": rank,
                    "name": name,
                    "points": points,
                    "date": ranking_date,
                    "team_id": team_id,
                    "team_slug": team_slug,
                }
                all_teams_data.append(team_data)

        except Exception as e:
            print(f"  Error parsing a team div: {e}. Skipping this team.")

    return all_teams_data, ranking_date


def create_top_x_teams_csv(input_csv, output_csv, top_x):
    """
    Reads the main rankings CSV, filters for all unique teams
    that were ever in the Top X, and saves them to a new CSV.
    """
    if not os.path.isfile(input_csv):
        print(f"Error: Main rankings file not found: {input_csv}")
        print("Please run the scraper with -gen or -urls first.")
        sys.exit(1)

    print(f"Reading {input_csv} to find all unique teams that hit the Top {top_x}...")

    peak_teams = {}  # Use dict to store unique teams by ID

    try:
        with open(input_csv, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    rank_num = int(re.sub(r"[^0-9]", "", row["rank"]))
                    team_id = row["team_id"]

                    # Check if they are in the top X and we haven't seen them
                    if rank_num <= top_x and team_id not in peak_teams:
                        if team_id == "N/A":
                            print(
                                f"Warning: Skipping team '{row['name']}' with N/A team_id."
                            )
                            continue

                        peak_teams[team_id] = {
                            "team_id": team_id,
                            "name": row["name"],
                            "team_slug": row["team_slug"],
                        }
                except (ValueError, KeyError, TypeError):
                    # Skip rows with bad data (e.g., missing rank, empty file)
                    continue

    except IOError as e:
        print(f"Error reading {input_csv}: {e}")
        sys.exit(1)

    if not peak_teams:
        print("No teams found matching the criteria. Is the rankings file empty?")
        return

    print(f"Found {len(peak_teams)} unique teams.")

    # Now, write these unique teams to the new CSV
    fieldnames = ["team_id", "name", "team_slug"]
    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for team_data in peak_teams.values():
                writer.writerow(team_data)

    except IOError as e:
        print(f"Error: Could not write to file {output_csv}: {e}")
        sys.exit(1)

    print(f"Successfully created {output_csv} with {len(peak_teams)} teams.")
