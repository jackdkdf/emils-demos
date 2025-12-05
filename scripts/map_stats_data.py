# map_stats_scraper.py

import cloudscraper
import csv
import sys
import time
import os
import random

# --- Import helper functions from our new utils file ---
try:
    from map_stats_utils import (
        scrape_map_stats_page,
        load_existing_map_rows,
        parse_url_for_info,
        MAP_ID_TO_NAME,  # <--- IMPORT THE MAP DICTIONARY
    )
except ImportError:
    print("Error: Could not import from 'map_stats_utils.py'.")
    print("Please make sure 'map_stats_utils.py' and MAP_ID_TO_NAME are set.")
    sys.exit(1)

# --- Main Configuration ---

# 1. Output FOLDER for team CSVs
OUTPUT_FOLDER = "data/team_results"

# 2. Date range for URL generation
START_DATE_STR = "2024-11-04"
END_DATE_STR = "2025-12-01"

# 3. Input CSV file with team list
TEAMS_INPUT_FILE = "data/teams_peak_36.csv"

# --- End Configuration ---


def generate_all_team_map_urls(teams_file, start_date, end_date):
    """
    Reads the teams_file and generates map stat URLs for every team and every map.
    """
    print(f"Generating URLs from {teams_file}...")
    urls_to_scrape = []

    if not os.path.isfile(teams_file):
        print(f"Error: Teams input file not found: {teams_file}")
        sys.exit(1)

    try:
        with open(teams_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            teams = list(reader)

        if not teams:
            print("Error: No teams found in teams_file.")
            return []

        # Check for required columns
        if "team_id" not in teams[0] or "team_slug" not in teams[0]:
            print("Error: CSV must have 'team_id' and 'team_slug' columns.")
            return []

        # Loop through each team, then loop through each map
        for team in teams:
            team_id = team["team_id"]
            team_slug = team["team_slug"]

            if not team_id or not team_slug:
                print(f"Warning: Skipping team with missing ID or slug: {team}")
                continue

            for map_id in MAP_ID_TO_NAME.keys():
                # Format: https://www.hltv.org/stats/teams/map/47/9565/vitality?startDate=...&endDate=...
                url = (
                    f"https://www.hltv.org/stats/teams/map/{map_id}/{team_id}/{team_slug}"
                    f"?startDate={start_date}&endDate={end_date}"
                )
                urls_to_scrape.append(url)

    except IOError as e:
        print(f"Error reading {teams_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during URL generation: {e}")
        sys.exit(1)

    print(
        f"Generated {len(urls_to_scrape)} URLs for {len(teams)} teams across {len(MAP_ID_TO_NAME)} maps."
    )
    return urls_to_scrape


def run_map_stats_scraper(urls_to_scrape):
    """
    Main scraping loop.
    1. Groups URLs by team.
    2. Scrapes all URLs for a team and saves to a team-specific CSV.
    (This function is UNCHANGED from the previous version)
    """
    print(f"Found {len(urls_to_scrape)} total URLs to scrape.")

    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved in '{OUTPUT_FOLDER}' directory.")

    # 2. Initialize the scraper
    scraper = cloudscraper.create_scraper()

    # 3. Define CSV fieldnames
    fieldnames = [
        "team_name",
        "team_id",
        "map_name",
        "map_id",
        "match_date",
        "opponent_name",
        "opponent_id",
        "score_us",
        "score_them",
        "result",
        "event_name",
        "event_id",
    ]

    # 4. Pre-scan and group URLs by team
    print("Grouping URLs by team...")
    grouped_urls = {}
    for url in urls_to_scrape:
        base_info = parse_url_for_info(url)
        if not base_info:
            print(f"  Skipping invalid URL: {url}")
            continue

        team_id = base_info["team_id"]
        team_name = base_info["team_name"]

        if team_id not in grouped_urls:
            grouped_urls[team_id] = {"team_name": team_name, "urls": []}
        grouped_urls[team_id]["urls"].append(url)

    print(f"Found {len(grouped_urls)} unique team(s) to process.")

    # 5. Loop through each TEAM and process all their URLs
    total_new_rows_all_teams = 0

    for team_id, data in grouped_urls.items():
        team_name = data["team_name"]
        team_urls = data["urls"]

        print(f"\n--- Processing Team: {team_name} (ID: {team_id}) ---")

        # 1. Set up the CSV file for *this specific team*
        output_csv_file = os.path.join(OUTPUT_FOLDER, f"{team_name}.csv")
        file_exists = os.path.isfile(output_csv_file)

        # 2. Load existing rows *for this team's file*
        existing_rows = load_existing_map_rows(output_csv_file)
        print(
            f"Loaded {len(existing_rows)} existing match rows from {output_csv_file}."
        )

        total_new_rows_this_team = 0

        # 3. Open this team's CSV in append mode
        try:
            with open(output_csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write header only if the file is brand new
                if not file_exists or os.path.getsize(output_csv_file) == 0:
                    writer.writeheader()
                    print(f"Created new file and wrote header to {output_csv_file}")

                # 4. Now, loop over all URLs *for this team*
                for i, url in enumerate(team_urls):
                    print("\n" + "=" * 50)
                    print(
                        f"Scraping URL {i+1} of {len(team_urls)} for {team_name}: {url}"
                    )
                    print("=" * 50)

                    # We already parsed the URL, but scrape_map_stats_page needs the info
                    base_info = parse_url_for_info(url)

                    page_data = scrape_map_stats_page(scraper, url, base_info)

                    if not page_data:
                        print("Could not get data for this page. Skipping.")
                        continue

                    # 5. Filter duplicates and write to this team's file
                    new_rows_for_this_page = 0
                    for match_data in page_data:
                        # Create a unique key for this match
                        row_key = (
                            match_data["team_name"],
                            match_data["opponent_name"],
                            match_data["map_name"],
                            match_data["match_date"],
                        )

                        if row_key not in existing_rows:
                            writer.writerow(match_data)
                            existing_rows.add(
                                row_key
                            )  # Add to set to prevent duplicates *in this session*
                            new_rows_for_this_page += 1

                    total_new_rows_this_team += new_rows_for_this_page
                    print(f"Found {len(page_data)} matches on this page.")
                    print(
                        f"Appended {new_rows_for_this_page} new rows to {output_csv_file}."
                    )

                    # 6. Be polite! Sleep
                    if i < len(team_urls) - 1:
                        delay = random.randint(2, 13)
                        print(f"Sleeping for {delay} seconds...")
                        time.sleep(delay)

        except IOError as e:
            print(f"Error: Could not write to file {output_csv_file}: {e}")
            continue  # Skip to the next team
        except Exception as e:
            print(f"An unexpected error occurred for team {team_name}: {e}")
            continue

        print(f"\nFinished processing for {team_name}.")
        print(f"Added {total_new_rows_this_team} new rows to {output_csv_file}.")
        total_new_rows_all_teams += total_new_rows_this_team

    print("\n--- Map Stats Scraping Complete ---")
    print(
        f"Added {total_new_rows_all_teams} new rows in total across {len(grouped_urls)} file(s)."
    )


def main():
    """
    Main function to run the scraper.
    """
    # 1. Generate the URLs first
    urls_to_scrape = generate_all_team_map_urls(
        TEAMS_INPUT_FILE, START_DATE_STR, END_DATE_STR
    )

    if not urls_to_scrape:
        print("No URLs were generated. Exiting.")
        sys.exit(0)

    # 2. Run the scraper with the generated list
    run_map_stats_scraper(urls_to_scrape)


if __name__ == "__main__":
    main()
