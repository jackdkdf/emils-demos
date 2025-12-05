import cloudscraper
import csv
import sys
import time
import os
import random
import argparse
from datetime import datetime, timedelta

# --- Import helper functions from our utils file ---
try:
    from ranking_utils import (
        generate_ranking_urls,
        scrape_ranking_page,
        load_existing_rows,
        create_top_x_teams_csv,
    )
except ImportError:
    print("Error: Could not import from 'scraper_utils.py'.")
    print("Please make sure 'scraper_utils.py' is in the same directory.")
    sys.exit(1)

# --- Main Configuration ---

# 1. Output CSV file for team rankings
RANKINGS_CSV_FILE = "data/hltv_team_rankings.csv"
PEAK_TEAMS_CSV_FILE = "data/teams_peak_36.csv"

# 2. Date generation parameters (for -gen mode)
# Note: HLTV ranking dates are always Mondays
START_DATE = datetime(year=2025, month=11, day=3)
END_DATE = datetime(year=2024, month=11, day=4)

# 3. Manual URL list (for -urls mode)
#    You can populate this list yourself if you prefer.
#    e.g., "https://www.hltv.org/ranking/teams/2025/november/3"
MANUAL_URLS_TO_SCRAPE = [
    "https://www.hltv.org/ranking/teams/2025/december/1",
    "https://www.hltv.org/ranking/teams/2025/november/24",
    "https://www.hltv.org/ranking/teams/2025/november/17",
    "https://www.hltv.org/ranking/teams/2025/november/10",
    "https://www.hltv.org/ranking/teams/2025/november/3",
    "https://www.hltv.org/ranking/teams/2025/october/27",
    "https://www.hltv.org/ranking/teams/2025/october/20",
    "https://www.hltv.org/ranking/teams/2025/october/13",
    "https://www.hltv.org/ranking/teams/2025/october/6",
    "https://www.hltv.org/ranking/teams/2025/september/29",
    "https://www.hltv.org/ranking/teams/2025/september/22",
    "https://www.hltv.org/ranking/teams/2025/september/15",
    "https://www.hltv.org/ranking/teams/2025/september/8",
    "https://www.hltv.org/ranking/teams/2025/september/2",
    "https://www.hltv.org/ranking/teams/2025/august/25",
    "https://www.hltv.org/ranking/teams/2025/august/18",
    "https://www.hltv.org/ranking/teams/2025/august/11",
    "https://www.hltv.org/ranking/teams/2025/august/4",
    "https://www.hltv.org/ranking/teams/2025/july/28",
    "https://www.hltv.org/ranking/teams/2025/july/21",
    "https://www.hltv.org/ranking/teams/2025/july/14",
    "https://www.hltv.org/ranking/teams/2025/july/7",
    "https://www.hltv.org/ranking/teams/2025/june/30",
    "https://www.hltv.org/ranking/teams/2025/june/23",
    "https://www.hltv.org/ranking/teams/2025/june/16",
    "https://www.hltv.org/ranking/teams/2025/june/9",
    "https://www.hltv.org/ranking/teams/2025/june/2",
    "https://www.hltv.org/ranking/teams/2025/may/26",
    "https://www.hltv.org/ranking/teams/2025/may/19",
    "https://www.hltv.org/ranking/teams/2025/may/12",
    "https://www.hltv.org/ranking/teams/2025/may/5",
    "https://www.hltv.org/ranking/teams/2025/april/28",
    "https://www.hltv.org/ranking/teams/2025/april/21",
    "https://www.hltv.org/ranking/teams/2025/april/14",
    "https://www.hltv.org/ranking/teams/2025/april/7",
    "https://www.hltv.org/ranking/teams/2025/march/31",
    "https://www.hltv.org/ranking/teams/2025/march/24",
    "https://www.hltv.org/ranking/teams/2025/march/17",
    "https://www.hltv.org/ranking/teams/2025/march/10",
    "https://www.hltv.org/ranking/teams/2025/march/3",
    "https://www.hltv.org/ranking/teams/2025/february/24",
    "https://www.hltv.org/ranking/teams/2025/february/17",
    "https://www.hltv.org/ranking/teams/2025/february/10",
    "https://www.hltv.org/ranking/teams/2025/february/3",
    "https://www.hltv.org/ranking/teams/2025/january/27",
    "https://www.hltv.org/ranking/teams/2025/january/20",
    "https://www.hltv.org/ranking/teams/2025/january/13",
    "https://www.hltv.org/ranking/teams/2025/january/6",
    "https://www.hltv.org/ranking/teams/2024/december/30",
    "https://www.hltv.org/ranking/teams/2024/december/23",
    "https://www.hltv.org/ranking/teams/2024/december/16",
    "https://www.hltv.org/ranking/teams/2024/december/9",
    "https://www.hltv.org/ranking/teams/2024/december/2",
    "https://www.hltv.org/ranking/teams/2024/november/25",
    "https://www.hltv.org/ranking/teams/2024/november/18",
    "https://www.hltv.org/ranking/teams/2024/november/11",
    "https://www.hltv.org/ranking/teams/2024/november/4",
]

# 4. Top X teams to filter (for -top mode)
TOP_X_TEAMS = 36

# --- End Configuration ---


def run_ranking_scraper(urls_to_scrape):
    """
    Main scraping loop to fetch rankings from a list of URLs.
    """
    print(f"Found {len(urls_to_scrape)} URLs to scrape.")

    # 1. Load all existing rows from the CSV to prevent duplicates
    #    This set will store tuples of (name, date)
    existing_rows = load_existing_rows(RANKINGS_CSV_FILE)
    print(
        f"Loaded {len(existing_rows)} existing team/date rows from {RANKINGS_CSV_FILE}."
    )

    # 2. Initialize the scraper
    scraper = cloudscraper.create_scraper()

    # 3. Define CSV fieldnames
    #    This MUST match the keys in the dict returned by scrape_ranking_page
    fieldnames = ["rank", "name", "points", "date", "team_id", "team_slug"]

    # 4. Open the CSV file in append mode
    total_new_rows = 0
    try:
        with open(RANKINGS_CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header if file is new/empty
            if not existing_rows and os.path.getsize(RANKINGS_CSV_FILE) == 0:
                writer.writeheader()
                print(f"Created new file and wrote header to {RANKINGS_CSV_FILE}")

            # 5. Start scraping loop
            for i, url in enumerate(urls_to_scrape):
                print("\n" + "=" * 50)
                print(f"Scraping URL {i+1} of {len(urls_to_scrape)}: {url}")
                print("=" * 50)

                # Scrape the page
                page_data, ranking_date = scrape_ranking_page(scraper, url)

                if not page_data or not ranking_date:
                    print("Could not get data for this page. Skipping.")
                    continue

                print(f"Scraping rankings for: {ranking_date}")

                # 6. Filter out duplicates
                new_rows_for_this_page = 0
                for team_data in page_data:
                    # Create a unique key for this team and date
                    row_key = (team_data["name"], team_data["date"])

                    if row_key not in existing_rows:
                        writer.writerow(team_data)
                        existing_rows.add(row_key)  # Add to set to prevent duplicates
                        new_rows_for_this_page += 1

                total_new_rows += new_rows_for_this_page
                print(f"Found {len(page_data)} unique teams on this page.")
                print(
                    f"Successfully appended {new_rows_for_this_page} new rows to CSV."
                )

                # 7. Be polite! Sleep for a random time if not the last URL
                if i < len(urls_to_scrape) - 1:
                    delay = random.randint(2, 9)
                    print(f"Sleeping for {delay} seconds...")
                    time.sleep(delay)

    except IOError as e:
        print(f"Error: Could not write to file {RANKINGS_CSV_FILE}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during the scrape: {e}")

    print("\n--- Ranking Scraping Complete ---")
    print(f"Added {total_new_rows} new rows in this session.")
    print(f"Total unique rows in {RANKINGS_CSV_FILE}: {len(existing_rows)}")


def main():
    """
    Main function to parse arguments and decide which action to take.
    """
    parser = argparse.ArgumentParser(
        description="HLTV Ranking Scraper",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Define the mutually exclusive group
    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "-gen",
        action="store_true",
        help="Generate weekly URLs from START_DATE to END_DATE and scrape them.",
    )
    mode_group.add_argument(
        "-urls",
        action="store_true",
        help="Scrape only the URLs manually defined in MANUAL_URLS_TO_SCRAPE.",
    )
    mode_group.add_argument(
        "-top",
        action="store_true",
        help=f"Read {RANKINGS_CSV_FILE} and generate {PEAK_TEAMS_CSV_FILE} "
        f"with all unique teams from the Top {TOP_X_TEAMS}.",
    )

    args = parser.parse_args()

    if args.gen:
        print("Mode: Generate URLs")
        # Generate the list of URLs to scrape
        urls_to_scrape = generate_ranking_urls(START_DATE, END_DATE)
        if not urls_to_scrape:
            print("No URLs were generated. Exiting.")
            sys.exit(0)
        run_ranking_scraper(urls_to_scrape)

    elif args.urls:
        print("Mode: Manual URL List")
        if not MANUAL_URLS_TO_SCRAPE:
            print(
                "Error: -urls mode selected, but MANUAL_URLS_TO_SCRAPE list is empty."
            )
            print("Please edit the script and add URLs to the list.")
            sys.exit(1)

        # Deduplicate the manual URL list
        urls_to_scrape = set(MANUAL_URLS_TO_SCRAPE)
        if len(urls_to_scrape) < len(MANUAL_URLS_TO_SCRAPE):
            print("Duplicate URLs in manual list, remove and run again.")
            sys.exit(0)

        run_ranking_scraper(urls_to_scrape)

    elif args.top:
        print(f"Mode: Generate Top {TOP_X_TEAMS} Teams File")
        create_top_x_teams_csv(RANKINGS_CSV_FILE, PEAK_TEAMS_CSV_FILE, TOP_X_TEAMS)


if __name__ == "__main__":
    main()
