# player_results_utils.py

import datetime
from bs4 import BeautifulSoup

# Import or Redefine the map dictionary.
# If you want to keep it centralized, you can import from map_stats_utils,
# but for standalone safety, I have included it here.
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
    """
    Generates a list of (start_date, end_date) strings, each 7 days apart.
    """
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
        # If the next week goes past the end date, stop (or clamp to end_date)
        if next_week > end_date:
            break

        s_str = current.strftime("%Y-%m-%d")
        e_str = next_week.strftime("%Y-%m-%d")
        intervals.append((s_str, e_str))
        current = next_week

    return intervals


def format_map_for_url(map_name):
    """
    Converts 'Ancient' -> 'de_ancient' for HLTV URL parameters.
    """
    clean = map_name.lower().strip()
    if not clean.startswith("de_"):
        return f"de_{clean}"
    return clean


def scrape_column_average(scraper, url, possible_headers):
    """
    Fetches a URL, finds the stats table, looks for a specific column header,
    and calculates the average value of that column for all rows (players).

    :param scraper: The cloudscraper instance
    :param url: The full URL to scrape
    :param possible_headers: List of strings to match header (e.g. ['Rating 2.0', 'Rating 1.0'])
    :return: Float (average) or None if failed/no data.
    """
    try:
        resp = scraper.get(url)
        if resp.status_code != 200:
            # 404s are common if a team didn't play that map that week
            return None

        soup = BeautifulSoup(resp.content, "html.parser")

        # Player stats are usually in '.stats-table'
        table = soup.find("table", class_="stats-table")
        if not table:
            return None

        # 1. Find Header Index
        thead = table.find("thead")
        if not thead:
            return None

        headers = thead.find_all("th")
        target_idx = -1

        for i, th in enumerate(headers):
            text = th.get_text(strip=True)
            if text in possible_headers:
                target_idx = i
                break

        if target_idx == -1:
            return None  # Column not found

        # 2. Sum values
        rows = table.find("tbody").find_all("tr")
        total = 0.0
        count = 0

        for row in rows:
            cols = row.find_all("td")
            if len(cols) <= target_idx:
                continue

            val_str = cols[target_idx].get_text(strip=True)

            # Cleaning data: remove '%' (for Flash Success) and handle '-'
            val_str = val_str.replace("%", "")
            if val_str == "-" or val_str == "":
                continue

            try:
                total += float(val_str)
                count += 1
            except ValueError:
                continue

        if count == 0:
            return 0.0

        return round(total / count, 2)

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
