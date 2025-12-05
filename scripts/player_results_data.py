# player_results_data.py

import cloudscraper
import csv
import sys
import time
import os
import random

try:
    from player_results_utils import (
        get_weekly_intervals,
        format_map_for_url,
        scrape_player_stats_dict,
        MAP_ID_TO_NAME,
    )
except ImportError:
    print("Error: Could not import from 'player_results_utils.py'.")
    sys.exit(1)

# --- Configuration ---
OUTPUT_FOLDER = "data/player_results"
TEAMS_INPUT_FILE = "data/teams_peak_36.csv"
START_DATE_STR = "2024-11-04"
END_DATE_STR = "2025-12-01"
# ---------------------


def run_weekly_scraper():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.isfile(TEAMS_INPUT_FILE):
        print(f"Error: File {TEAMS_INPUT_FILE} not found.")
        sys.exit(1)

    try:
        with open(TEAMS_INPUT_FILE, "r", encoding="utf-8") as f:
            teams = list(csv.DictReader(f))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    weeks = get_weekly_intervals(START_DATE_STR, END_DATE_STR)
    if not weeks:
        print("No valid weeks generated.")
        sys.exit(1)

    scraper = cloudscraper.create_scraper()
    total_maps = len(MAP_ID_TO_NAME)

    for team in teams:
        team_name = team.get("team_slug")
        team_id = team.get("team_id")

        if not team_name or not team_id:
            continue

        output_csv = os.path.join(OUTPUT_FOLDER, f"{team_name}_weekly_stats.csv")
        file_exists = os.path.isfile(output_csv)

        fieldnames = [
            "team_name",
            "map_name",
            "start_date",
            "end_date",
            "player_name",
            "overall_rating",
            "utility_success",
            "opening_rating",
        ]

        existing_entries = set()
        if file_exists:
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_entries.add((row["map_name"], row["start_date"]))

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            map_counter = 0
            for map_id, map_display_name in MAP_ID_TO_NAME.items():
                map_counter += 1
                map_url_param = format_map_for_url(map_display_name)

                print("\n" + "=" * 50)
                print(
                    f"Processing Map {map_counter} of {total_maps} for {team_name}: {map_display_name}"
                )
                print("=" * 50)

                rows_added_this_map = 0
                week_counter = 0

                for start_d, end_d in weeks:
                    week_counter += 1

                    if (map_display_name, start_d) in existing_entries:
                        # Optional: Uncomment to see skipped weeks
                        # print(f"Week {week_counter} ({start_d}): Skipped (Already in CSV)")
                        continue

                    # --- Build URLs ---
                    base_url = f"https://www.hltv.org/stats/teams/players"
                    params = (
                        f"?startDate={start_d}&endDate={end_d}&maps={map_url_param}"
                    )

                    url_overall = f"{base_url}/{team_id}/{team_name}{params}"
                    url_utility = f"{base_url}/flashes/{team_id}/{team_name}{params}"
                    url_opening = (
                        f"{base_url}/openingkills/{team_id}/{team_name}{params}"
                    )

                    print(f"Week {week_counter} ({start_d}):")
                    print(f"Overall URL:       {url_overall}")

                    # 1. Scrape Overall
                    players_overall = scrape_player_stats_dict(
                        scraper, url_overall, ["Rating"]
                    )

                    if not players_overall:
                        print("Result:            [No Data]")
                        print("-" * 20)

                        # Random sleep even if no data (helps avoid detection)
                        delay = random.uniform(2.0, 4.0)
                        # print(f"Sleeping {delay:.1f}s...")
                        time.sleep(delay)
                        continue

                    # 2. Scrape Utility
                    print(f"Flash URL:         {url_utility}")
                    # Tiny sleep between requests for the SAME week
                    time.sleep(random.uniform(1.0, 2.0))

                    # --- CHANGE HERE: Removed "Rating", Added "Success" ---
                    # The HTML shows the header is "Success", so we look for "Succ" or "Success"
                    players_utility = scrape_player_stats_dict(
                        scraper, url_utility, ["Succ", "Success"]
                    )

                    # 3. Scrape Opening
                    print(f"Opening Kills URL: {url_opening}")
                    # Tiny sleep between requests for the SAME week
                    time.sleep(random.uniform(1.0, 2.0))
                    players_opening = scrape_player_stats_dict(
                        scraper, url_opening, ["Rating"]
                    )

                    # 4. Merge & Save
                    count_saved = 0
                    for player, rating in players_overall.items():
                        util_score = 0.0
                        if players_utility and player in players_utility:
                            util_score = players_utility[player]

                        open_score = 0.0
                        if players_opening and player in players_opening:
                            open_score = players_opening[player]

                        row_data = {
                            "team_name": team_name,
                            "map_name": map_display_name,
                            "start_date": start_d,
                            "end_date": end_d,
                            "player_name": player,
                            "overall_rating": rating,
                            "utility_success": util_score,
                            "opening_rating": open_score,
                        }
                        writer.writerow(row_data)
                        count_saved += 1

                    f.flush()
                    rows_added_this_map += count_saved
                    print(f"Result:            Saved {count_saved} players.")
                    print("-" * 20)

                    # --- UPDATE: The main weekly delay you asked for ---
                    delay = random.uniform(5.0, 8.0)
                    print(f"Sleeping {delay:.2f}s...")
                    time.sleep(delay)

                print(
                    f"Finished Map {map_display_name}. Total rows added: {rows_added_this_map}"
                )

                if map_counter < total_maps:
                    delay = random.randint(10, 20)
                    print(f"Sleeping {delay}s before next map...")
                    time.sleep(delay)


if __name__ == "__main__":
    run_weekly_scraper()
