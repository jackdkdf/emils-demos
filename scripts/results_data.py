import os
import re
import sys
import csv
import time
import random
from datetime import datetime
from bs4 import BeautifulSoup
from patchright.sync_api import sync_playwright, TimeoutError


# --- Config ---
TEAM_HTML_FOLDER = "team_html_pages"
TEAM_RESULTS_FOLDER = "team_results"
COOKIES_FILE = "hltv_cookies.json"
TOP_X_TEAMS = 36
PEAK_TEAMS_CSV = "peak_teams.csv"
START_DATE = "2024-11-04"
END_DATE = "2025-11-03"


# --- Helper: Load list of top teams ---
def get_peak_teams(csv_path, top_x):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    teams = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(teams) >= top_x:
                break
            teams.append(
                {
                    "team_id": row["team_id"],
                    "name": row["name"],
                    "team_slug": row["team_slug"],
                    "csv_file": os.path.join(
                        TEAM_RESULTS_FOLDER, f"{row['team_slug']}_results.csv"
                    ),
                }
            )
    return teams


# --- Helper: Parse local HTML file for results ---
def parse_results_html(file_path):
    print(f"Processing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    results_containers = soup.find_all("div", class_="results-all")

    print(
        f"[Debug] Found {len(results_containers)} <div class='results-all'> containers"
    )

    all_matches = []
    for idx, container in enumerate(results_containers, start=1):
        print(f"[Debug] Parsing .results-all container #{idx}")
        sublists = container.find_all("div", class_="results-sublist")
        print(
            f"[Debug] Found {len(sublists)} .results-sublist elements in container #{idx}"
        )

        for sub_idx, sublist in enumerate(sublists, start=1):
            # --- Extract date from the sublist header ---
            date_div = sublist.find("div", class_="standard-headline")
            if not date_div:
                continue
            date_text = date_div.get_text(strip=True)
            date_text = re.sub(r"Results for ", "", date_text)
            date_text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_text)

            try:
                match_date = datetime.strptime(date_text, "%B %d %Y").strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                print(f"[Debug] Could not parse date: {date_text}")
                continue

            result_divs = sublist.find_all("div", class_="result-con")
            print(f"[Debug] Sublist #{sub_idx} date: {match_date}")
            print(
                f"[Debug] Found {len(result_divs)} .result-con entries in sublist #{sub_idx}"
            )

            for res in result_divs:
                a_tag = res.find("a", href=True)
                if not a_tag:
                    continue
                match_url = a_tag["href"]
                match_id_match = re.search(r"/matches/(\d+)/", match_url)
                match_id = match_id_match.group(1) if match_id_match else "N/A"

                result = res.find("div", class_="result")
                if not result:
                    continue

                # Extract teams
                team_cells = result.find_all("td", class_="team-cell")
                if len(team_cells) != 2:
                    continue

                team1_name = (
                    team_cells[0].find("div", class_="team").get_text(strip=True)
                    if team_cells[0].find("div", class_="team")
                    else "N/A"
                )
                team2_name = (
                    team_cells[1].find("div", class_="team").get_text(strip=True)
                    if team_cells[1].find("div", class_="team")
                    else "N/A"
                )

                # Extract score
                score_td = result.find("td", class_="result-score")
                score_text = score_td.get_text(strip=True) if score_td else "N/A"

                # Extract event
                event_td = result.find("td", class_="event")
                event_name = (
                    event_td.find("span", class_="event-name").get_text(strip=True)
                    if event_td and event_td.find("span", class_="event-name")
                    else "N/A"
                )

                all_matches.append(
                    {
                        "match_id": match_id,
                        "date": match_date,
                        "team1": team1_name,
                        "team2": team2_name,
                        "score": score_text,
                        "event": event_name,
                    }
                )

                print(
                    f"[Parsed] {match_date}: {team1_name} vs {team2_name} "
                    f"({score_text}) - {event_name} [ID: {match_id}]"
                )

    print(
        f"[Debug] Parsed total of {len(all_matches)} matches across all .results-all containers."
    )
    return all_matches


# --- Helper: Save matches to CSV (with header correction + deduplication) ---
def save_matches_to_csv(team_slug, all_matches):
    os.makedirs(TEAM_RESULTS_FOLDER, exist_ok=True)
    output_file = os.path.join(TEAM_RESULTS_FOLDER, f"{team_slug}_results.csv")

    fieldnames = ["match_id", "date", "team1", "team2", "score", "event"]

    recreate = not os.path.exists(output_file) or any(
        fn not in open(output_file).readline() for fn in fieldnames
    )

    if recreate:
        print(f"[Info] Recreating {output_file} with proper headers.")
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Load existing match IDs to deduplicate
    existing_ids = set()
    try:
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["match_id"])
    except Exception:
        pass

    new_matches = [m for m in all_matches if m["match_id"] not in existing_ids]
    print(f"Found {len(new_matches)} new matches to add.")

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_matches)

    print(f"✅ Successfully appended {len(new_matches)} matches to {output_file}")


# --- Main script ---
def main():
    print(f"Loading top {TOP_X_TEAMS} teams from {PEAK_TEAMS_CSV}...")
    teams = get_peak_teams(PEAK_TEAMS_CSV, TOP_X_TEAMS)
    print(f"Found {len(teams)} teams to process.")

    os.makedirs(TEAM_HTML_FOLDER, exist_ok=True)
    os.makedirs(TEAM_RESULTS_FOLDER, exist_ok=True)

    # Launch browser manually for CAPTCHA step
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("\n--- ACTION REQUIRED ---")
        print("1. The browser will open HLTV below.")
        print("2. Solve the 'Verify you are human' check manually.")
        print(
            "3. When you are past Cloudflare and see HLTV homepage, press ENTER here.\n"
        )

        page.goto("https://www.hltv.org", wait_until="domcontentloaded", timeout=60000)
        input("Press ENTER when you have completed the human verification...")

        # Save session cookies
        context.storage_state(path=COOKIES_FILE)
        print(f"✅ Saved cookies to {COOKIES_FILE}")
        browser.close()

    # Now process each team
    for i, team in enumerate(teams, start=1):
        team_slug = team["team_slug"]
        team_html = os.path.join(TEAM_HTML_FOLDER, f"{team_slug}.html")
        print("\n" + "=" * 60)
        print(f"Processing team {i}/{len(teams)}: {team['name']} ({team_slug})")

        if not os.path.exists(team_html):
            print("--- ACTION REQUIRED ---")
            print("1. Open this URL in your browser:")
            print(
                f"   https://www.hltv.org/results?team={team['team_id']}"
                f"&startDate={START_DATE}&endDate={END_DATE}"
            )
            print("\n2. Scroll to the bottom until ALL matches are loaded.")
            print(
                "3. Save as 'Webpage, Complete' or 'HTML Only' inside team_html_pages."
            )
            print(f"   Example filename: {TEAM_HTML_FOLDER}/{team_slug}.html\n")

            file_name = input(
                "Type the name of the file you saved (or 'skip' to pass): "
            ).strip()
            if file_name.lower() == "skip":
                continue
            team_html = os.path.join(TEAM_HTML_FOLDER, file_name)

        matches = parse_results_html(team_html)
        print(f"Found {len(matches)} total matches in HTML.")

        if matches:
            save_matches_to_csv(team_slug, matches)

        if i < len(teams):
            delay = random.randint(5, 10)
            print(f"Sleeping {delay}s before next team...")
            time.sleep(delay)

    print("\n✅ All teams processed successfully.")


if __name__ == "__main__":
    main()
