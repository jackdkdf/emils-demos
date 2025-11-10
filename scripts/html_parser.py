import sys
import os
import csv
from datetime import datetime

# Import from our utility files
from results_utils import (
    get_peak_teams,
    load_existing_matches,
    write_rows_to_csv,
    PEAK_TEAMS_CSV,
    RESULTS_FOLDER,
    CSV_HEADER,
)
from html_utils import parse_results_from_html


# --- Configuration ---
# Date range for results
END_DATE = "2025-11-03"
START_DATE = "2024-11-04"

# Folder to store the HTML files you download
HTML_FOLDER = "team_html_pages"
# --- End Configuration ---


def main():
    """
    Main function to guide the user through the manual HTML parsing process.
    """
    # --- 1. Create results and HTML folders ---
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print(f"Created directory: {RESULTS_FOLDER}")
    if not os.path.exists(HTML_FOLDER):
        os.makedirs(HTML_FOLDER)
        print(f"Created directory: {HTML_FOLDER}")
        print(f"Please save your HTML files in this folder.")

    # --- 2. Get list of teams to process ---
    print(f"Loading teams from {PEAK_TEAMS_CSV}...")
    try:
        # Pass the folder path to get_peak_teams
        teams_to_process = get_peak_teams(PEAK_TEAMS_CSV)
    except FileNotFoundError:
        print(f"Error: {PEAK_TEAMS_CSV} not found.")
        print("Please run 'python hltv_scraper.py -top' first to create it.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {PEAK_TEAMS_CSV}: {e}")
        sys.exit(1)

    if not teams_to_process:
        print(f"No teams found in {PEAK_TEAMS_CSV}. Nothing to process.")
        return

    print(f"Found {len(teams_to_process)} unique teams to process.")

    # --- 3. Loop through teams ---
    total_teams = len(teams_to_process)
    for i, team in enumerate(teams_to_process):
        team_id = team["team_id"]
        team_name = team["name"]
        team_slug = team["team_slug"]

        # Construct the CSV file path here
        csv_file_name = f"{team_slug}_results.csv"
        csv_file = os.path.join(RESULTS_FOLDER, csv_file_name)

        print("\n" + "=" * 50)
        print(f"Processing Team {i+1}/{total_teams}: {team_name} (ID: {team_id})")

        # --- 4. Generate the URL for the user ---
        url = f"https://www.hltv.org/results?team={team_id}&startDate={START_DATE}&endDate={END_DATE}"

        print("\n--- ACTION REQUIRED ---")
        print("1. Open this URL in your browser:")
        print(f"   {url}")
        print(
            "\n2. Scroll down to the bottom of the page until ALL matches are loaded."
        )
        print(
            "3. Save the page (Ctrl+S or Cmd+S) as 'Webpage, Complete' or 'HTML Only'."
        )
        print(f"4. Save the file inside the '{HTML_FOLDER}' directory.")
        print("   (Example: save it as '{HTML_FOLDER}/{team_slug}.html')")

        html_file_path = ""
        while True:
            try:
                filename = input(
                    "\nType the name of the file you saved (or 'skip' to pass): "
                )
                if filename.lower() == "skip":
                    print(f"Skipping team {team_name}.")
                    break

                # Build the full path
                full_path = os.path.join(HTML_FOLDER, filename)

                if os.path.isfile(full_path):
                    html_file_path = full_path
                    break
                else:
                    print(f"Error: File not found at '{full_path}'.")
                    print(
                        "Please make sure the file is saved in the correct folder and the name is correct."
                    )
            except KeyboardInterrupt:
                print("\nScript interrupted by user. Exiting.")
                sys.exit(0)

        if not html_file_path:
            continue  # Go to the next team

        # --- 5. Process the HTML file ---
        print(f"Processing file: {html_file_path}")
        try:
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            print(f"Error reading HTML file: {e}")
            continue

        # Load existing matches from the CSV to prevent duplicates
        existing_match_ids = load_existing_matches(csv_file)
        print(f"Found {len(existing_match_ids)} existing matches to skip.")

        # Parse the HTML content
        all_matches = parse_results_from_html(html_content, url)
        if not all_matches:
            print(f"No matches found in {html_file_path}. Check the file.")
            continue

        print(f"Found {len(all_matches)} total matches in HTML.")

        # Filter out matches we already have
        new_rows = []
        for match in all_matches:
            if match["match_id"] not in existing_match_ids:
                new_rows.append(match)
                existing_match_ids.add(
                    match["match_id"]
                )  # Add to set to avoid intra-file duplicates

        print(f"Found {len(new_rows)} new matches to add.")

        # --- 6. Write new rows to CSV ---
        if new_rows:
            write_rows_to_csv(csv_file, new_rows, CSV_HEADER)
            print(f"Successfully appended {len(new_rows)} new rows to {csv_file}")
        else:
            print("No new matches to append (all data already exists).")

    print("\n" + "=" * 50)
    print("All teams processed. Manual parsing complete.")


if __name__ == "__main__":
    main()
