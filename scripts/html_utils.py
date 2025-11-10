import re
import sys
from datetime import datetime
from bs4 import BeautifulSoup


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


def parse_match_container(div, match_date):
    """
    Parses a single <div class='result-con'> block and extracts:
    - match_id
    - team1, team2
    - score
    - event
    - date (from parent sublist)
    """
    link_tag = div.find("a", href=True)
    if not link_tag:
        print("[Debug] Skipping result without link.")
        return None

    href = link_tag["href"]
    match = re.search(r"/matches/(\d+)/", href)
    match_id = match.group(1) if match else None

    team_cells = div.find_all("div", class_="team")
    teams = [t.get_text(strip=True) for t in team_cells if t.get_text(strip=True)]

    score_span = div.find("td", class_="result-score")
    score = score_span.get_text(strip=True) if score_span else None

    event_name = div.find("span", class_="event-name")
    event = event_name.get_text(strip=True) if event_name else "Unknown"

    if len(teams) != 2:
        print(
            f"[Debug] Warning: Unexpected team count ({len(teams)}) for match {match_id}"
        )
        print(div.prettify()[:300])

    result = {
        "match_id": match_id,
        "date": match_date,
        "team_1": teams[0] if len(teams) > 0 else None,
        "team_2": teams[1] if len(teams) > 1 else None,
        "score_1": score.split("-")[0] if score else None,
        "score_2": score.split("-")[1] if score else None,
        "event": event,
    }

    print(
        f"[Parsed] {match_date}: {result['team_1']} vs {result['team_2']} "
        f"({result['score_1']}-{result['score_2']}) - {result['event']} [ID: {match_id}]"
    )

    return result


def parse_results_from_html(html_content, url=None):
    """
    Parses an HLTV results page (saved HTML) and extracts structured match data.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    all_matches = []

    results_all_divs = soup.find_all("div", class_="results-all")
    print(f"[Debug] Found {len(results_all_divs)} <div class='results-all'> containers")

    for i, results_div in enumerate(results_all_divs):
        print(f"[Debug] Parsing .results-all container #{i + 1}")

        sublists = results_div.find_all("div", class_="results-sublist")
        print(
            f"[Debug] Found {len(sublists)} .results-sublist elements in container #{i + 1}"
        )

        for sublist_idx, sublist in enumerate(sublists, start=1):
            # ---- Extract date headline ----
            headline = sublist.find("span", class_="standard-headline") or sublist.find(
                "div", class_="standard-headline"
            )

            match_date = None
            if headline:
                date_str = headline.get_text(strip=True)
                cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str)
                cleaned = cleaned.replace("Results for", "").strip()
                try:
                    date_obj = datetime.strptime(cleaned, "%B %d %Y")
                    match_date = date_obj.strftime("%Y-%m-%d")
                except Exception:
                    match_date = cleaned
                print(f"[Debug] Sublist #{sublist_idx} date: {match_date}")
            else:
                print(f"[Debug] Sublist #{sublist_idx} has no date headline.")
                match_date = "unknown"

            # ---- Extract all result-con entries ----
            match_divs = sublist.find_all("div", class_="result-con")
            print(
                f"[Debug] Found {len(match_divs)} .result-con entries in sublist #{sublist_idx}"
            )

            for match_div in match_divs:
                match_data = parse_match_container(match_div, match_date)
                if match_data:
                    all_matches.append(match_data)

    print(
        f"[Debug] Parsed total of {len(all_matches)} matches across all .results-all containers."
    )
    return all_matches
