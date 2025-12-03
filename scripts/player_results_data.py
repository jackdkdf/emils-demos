# player_results_data.py

import cloudscraper
import csv
import sys
import time
import os
import random

# --- Import from our new utils file ---
try:
    from player_results_util import (
        get_weekly_intervals,
        format_map_for_url,
        scrape_column_average,
        MAP_ID_TO_NAME,
    )
except ImportError:
    print("Error: Could not import from 'player_results_utils.py'.")
    sys.exit(1)

# --- Configuration ---
OUTPUT_FOLDER = "player_results"
TEAMS_INPUT_FILE = "teams_peak_36.csv"
START_DATE_STR = "2024-11-04"
END_DATE_STR = "2025-12-01"
# ---------------------


def run_weekly_scraper():
    # 1. Setup Output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. Load Teams
    if not os.path.isfile(TEAMS_INPUT_FILE):
        print(f"Error: File {TEAMS_INPUT_FILE} not found.")
        sys.exit(1)

    try:
        with open(TEAMS_INPUT_FILE, "r", encoding="utf-8") as f:
            teams = list(csv.DictReader(f))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # 3. Generate Weeks
    weeks = get_weekly_intervals(START_DATE_STR, END_DATE_STR)
    if not weeks:
        print("No valid weeks generated from date range.")
        sys.exit(1)

    print(f"Loaded {len(teams)} teams.")
    print(f"Timeframe split into {len(weeks)} weeks.")

    # 4. Init Scraper
    scraper = cloudscraper.create_scraper()

    # 5. Loop Teams
    for team in teams:
        team_name = team.get("team_slug")
        team_id = team.get("team_id")

        if not team_name or not team_id:
            continue

        print(f"\n=== Starting Team: {team_name} ===")

        # Prepare CSV for this team
        output_csv = os.path.join(OUTPUT_FOLDER, f"{team_name}_weekly_stats.csv")
        file_exists = os.path.isfile(output_csv)

        fieldnames = [
            "team_name",
            "map_name",
            "start_date",
            "end_date",
            "overall_rating",
            "utility_success",
            "opening_rating",
        ]

        # Load existing rows to skip duplicates
        existing_entries = set()
        if file_exists:
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Key: Map + Start Date
                    existing_entries.add((row["map_name"], row["start_date"]))

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            # 6. Loop Maps
            for map_id, map_display_name in MAP_ID_TO_NAME.items():
                map_url_param = format_map_for_url(map_display_name)
                print(f"\n  [Map: {map_display_name}]")

                # 7. Loop Weeks
                for start_d, end_d in weeks:

                    # Check if we already have this data
                    if (map_display_name, start_d) in existing_entries:
                        continue

                    print(f"    > Processing week {start_d}...", end="", flush=True)

                    # --- Build URLs ---
                    base_url = f"https://www.hltv.org/stats/teams/players"
                    params = (
                        f"?startDate={start_d}&endDate={end_d}&maps={map_url_param}"
                    )

                    # 1. Overall Rating URL
                    url_overall = f"{base_url}/{team_id}/{team_name}{params}"

                    # 2. Utility URL (Flash stats)
                    url_utility = f"{base_url}/flashes/{team_id}/{team_name}{params}"

                    # 3. Opening URL
                    url_opening = (
                        f"{base_url}/openingkills/{team_id}/{team_name}{params}"
                    )

                    # --- Scrape (3 Steps) ---

                    # Step 1: Overall
                    # Looks for Rating 2.0 or 1.0
                    overall_val = scrape_column_average(
                        scraper, url_overall, ["Rating 2.0", "Rating 1.0"]
                    )

                    # If overall returns None, the team likely didn't play this map this week.
                    # We can skip the other two requests to save time.
                    if overall_val is None:
                        print(" [No Data]")
                        time.sleep(random.uniform(0.5, 1.0))
                        continue

                    time.sleep(random.uniform(1, 2))  # Polite pause

                    # Step 2: Utility
                    # Looks for "Succ." (Success %) or "Rating"
                    util_val = scrape_column_average(
                        scraper, url_utility, ["Succ.", "Rating"]
                    )

                    time.sleep(random.uniform(1, 2))  # Polite pause

                    # Step 3: Opening Kills
                    # Looks for "Rating"
                    open_val = scrape_column_average(scraper, url_opening, ["Rating"])

                    # --- Save ---
                    row_data = {
                        "team_name": team_name,
                        "map_name": map_display_name,
                        "start_date": start_d,
                        "end_date": end_d,
                        "overall_rating": overall_val,
                        "utility_success": util_val,
                        "opening_rating": open_val,
                    }

                    writer.writerow(row_data)
                    f.flush()  # Save immediately
                    print(" [Saved]")

                    # --- Rate Limit ---
                    # We just hit 3 pages, so sleep a bit longer
                    time.sleep(random.uniform(2, 5))


if __name__ == "__main__":
    run_weekly_scraper()
