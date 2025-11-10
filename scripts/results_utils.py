import csv
import os
import re
import time
import random
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import Page, TimeoutError

# --- Configuration ---
PEAK_TEAMS_CSV = "teams_peak_36.csv"
RESULTS_FOLDER = "team_results"
CSV_HEADER = ["match_id", "date", "team_1", "team_2", "score_1", "score_2", "event"]
# --- End Configuration ---


def get_peak_teams(peak_csv_file):
    """
    Reads the peak_teams.csv file and returns a list of team dicts.
    Now correctly processes all teams in the file without a 'peak_rank' check.
    """
    teams = []
    if not os.path.isfile(peak_csv_file):
        raise FileNotFoundError(f"{peak_csv_file} not found.")

    with open(peak_csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # We just need the ID, name, and slug.
                team_id = row["team_id"]
                team_slug = row["team_slug"]

                # Create a clean CSV filename from the team slug
                csv_file_name = f"{team_slug}_results.csv"
                csv_file_path = os.path.join(RESULTS_FOLDER, csv_file_name)

                teams.append(
                    {
                        "team_id": team_id,
                        "name": row["name"],
                        "team_slug": team_slug,
                        "csv_file": csv_file_path,
                    }
                )
            except KeyError:
                print(f"Skipping row, missing 'team_id' or 'team_slug': {row}")

    return teams


def load_existing_matches(csv_file_path):
    """
    Reads an existing team results CSV and returns a set of match_ids
    to prevent writing duplicate rows.
    """
    existing_ids = set()
    if not os.path.isfile(csv_file_path):
        return existing_ids

    try:
        with open(csv_file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "match_id" in row:
                    existing_ids.add(row["match_id"])
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}. Starting with an empty set.")
    return existing_ids


def write_rows_to_csv(csv_file_path, data_rows, header):
    """
    Appends a list of data rows to a CSV file.
    Creates the file and writes the header if it doesn't exist.
    """
    file_exists = os.path.isfile(csv_file_path)
    try:
        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists or os.path.getsize(csv_file_path) == 0:
                writer.writeheader()
            writer.writerows(data_rows)
    except Exception as e:
        print(f"Error writing to {csv_file_path}: {e}")


def parse_results_from_html(html_content, url):
    """
    Parses the HTML content of a results page and extracts match data.
    This version is built to handle the manual HTML file structure.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    all_matches = []

    # Find all date headlines, e.g., "Results for November 3rd 2025"
    date_headlines = soup.find_all("span", class_="standard-headline")

    if not date_headlines:
        # Fallback for pages with no date headlines but results (should be rare)
        # Try to find all result containers directly
        results = soup.find_all("div", class_="result-con")
        if not results:
            print("No date headlines or 'result-con' divs found in HTML.")
            return []

        # Try to parse date from URL as a last resort
        match_date = parse_date_from_url(url)
        print(
            f"Warning: No date headlines. Using single date {match_date} for {len(results)} matches."
        )
        for res_con in results:
            match = parse_match_container(res_con, match_date)
            if match:
                all_matches.append(match)
        return all_matches

    # Main logic: Loop through each date headline and get matches below it
    for headline in date_headlines:
        try:
            # Extract date from headline text
            date_str = headline.text.replace("Results for ", "").strip()
            # Clean the date string (e.g., "November 9th, 2025" -> "November 9, 2025")
            cleaned_date_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str)
            date_obj = datetime.strptime(cleaned_date_str, "%B %d, %Y")
            match_date = date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            print(
                f"Could not parse date from headline '{headline.text}': {e}. Skipping section."
            )
            continue

        # Find the parent 'results-sublist' and then all matches inside it
        sublist = headline.find_parent("div", class_="results-sublist")
        if not sublist:
            continue

        match_containers = sublist.find_all("div", class_="result-con")
        for res_con in match_containers:
            match = parse_match_container(res_con, match_date)
            if match:
                all_matches.append(match)

    return all_matches


def parse_match_container(res_con, match_date):
    """
    Parses a single 'result-con' div and extracts match data.
    """
    try:
        link_tag = res_con.find("a", class_="a-reset")
        if not link_tag or not link_tag.get("href"):
            return None  # Not a valid match container

        match_id = link_tag["href"].split("/")[2]

        team_tags = link_tag.find_all("div", class_="team")
        if len(team_tags) != 2:
            return None  # Not a standard 2-team match

        team_1 = team_tags[0].text.strip()
        team_2 = team_tags[1].text.strip()

        score_span = link_tag.find("span", class_="result-score")
        if not score_span:
            return None

        scores = [s.strip() for s in score_span.text.split("-")]
        if len(scores) != 2:
            return None  # Not a standard score

        score_1 = scores[0]
        score_2 = scores[1]

        event = (
            link_tag.find("span", class_="event-name")
            or link_tag.find("div", class_="event-name")
        ).text.strip()

        return {
            "match_id": match_id,
            "date": match_date,
            "team_1": team_1,
            "team_2": team_2,
            "score_1": score_1,
            "score_2": score_2,
            "event": event,
        }
    except Exception as e:
        print(f"Error parsing match container: {e}")
        return None


def parse_date_from_url(url_string):
    """Helper to guess date from a URL if all else fails."""
    try:
        # Try to find a date like '2025-11-03'
        match = re.search(r"(\d{4}-\d{2}-\d{2})", url_string)
        if match:
            return match.group(1)
    except:
        pass  # Fallback to today
    return datetime.now().strftime("%Y-%m-%d")


# --- Playwright-specific functions ---


def scrape_team_results(
    page: Page, team_id, team_slug, start_date, end_date, csv_file_path
):
    """
    Uses Playwright to scrape all results pages for a single team.
    """
    existing_match_ids = load_existing_matches(csv_file_path)
    if existing_match_ids:
        print(f"Loaded {len(existing_match_ids)} existing matches to skip.")
    else:
        print(f"Creating new results file: {csv_file_path}")

    new_matches_found = 0
    offset = 0

    while True:
        url = f"https://www.hltv.org/results?team={team_id}&startDate={start_date}&endDate={end_date}&offset={offset}"
        print(f"  Fetching results page: offset={offset}...")

        try:
            # 1. Go to the page, wait for HTML
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # 2. Wait for the network to be idle
            page.wait_for_load_state("networkidle", timeout=60000)

        except TimeoutError as e:
            print(f"  Page load or network idle timed out: {e}. Stopping team.")
            break
        except Exception as e:
            print(
                f"  An unexpected error occurred during page load: {e}. Stopping team."
            )
            break

        # Check if the "No results found" message is present
        no_results_el = page.locator("div.no-results-content")
        if no_results_el.is_visible():
            if offset == 0:
                print("  No results found for this team in the date range.")
            else:
                print("  No more results found. Finished team.")
            break

        # Check for results container
        try:
            page.wait_for_selector("div.results-holder", timeout=10000)
        except TimeoutError:
            print("  No results found on page (timeout). Finished team.")
            break

        # Get the HTML of all result containers
        result_divs = page.locator("div.result-con")

        try:
            # Use inner_html() on each element, not all()
            all_html_snippets = [div.inner_html() for div in result_divs.all()]
        except AttributeError:
            print(
                "  Error: 'Locator' object has no attribute 'all'. This might be a Playwright version issue."
            )
            break  # Stop this team
        except Exception as e:
            print(f"  Error getting HTML from locators: {e}")
            break

        if not all_html_snippets:
            if offset == 0:
                print(
                    "  No matches found on the first page, even though 'no results' wasn't visible."
                )
            else:
                print("  Found no more 'result-con' divs. Finished team.")
            break

        # Get the date from the headline. This is less reliable on paginated
        # content, so we will parse the date from each sublist.
        html_content = page.content()
        all_matches_on_page = parse_results_from_html(html_content, url)

        new_rows = []
        for match in all_matches_on_page:
            if match["match_id"] not in existing_match_ids:
                new_rows.append(match)
                existing_match_ids.add(match["match_id"])

        if new_rows:
            write_rows_to_csv(csv_file_path, new_rows, CSV_HEADER)
            new_matches_found += len(new_rows)
            print(f"  Found and appended {len(new_rows)} new matches.")
        else:
            print("  No new matches found on this page (all are duplicates).")

        # HLTV pagination is 100 results per page, but we'll check just in case
        if len(all_matches_on_page) < 50:
            print("  Fewer than 50 results on page. Assuming this is the last page.")
            break

        offset += 100  # HLTV uses offset, not page number
        time.sleep(random.randint(2, 5))  # Small delay between pages
