import requests
import json
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ================= CONFIG ================= #

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (SHL-Recommendation-System)"
}

PAGE_SIZE = 12
MAX_START = 500

TIMEOUT = 30
RETRIES = 4
SLEEP_BETWEEN_REQUESTS = 1.0

OUTPUT_PATH = "data/raw_catalog.json"

# ================= HTTP ================= #

def get_html(url):
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text

        except requests.exceptions.ReadTimeout as e:
            last_error = e
            wait = attempt * 3
            print(f"⏳ Timeout ({attempt}/{RETRIES}) → retry in {wait}s")
            time.sleep(wait)

        except Exception as e:
            last_error = e
            wait = attempt * 2
            print(f"⚠️ Error ({attempt}/{RETRIES}): {e}")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {RETRIES} retries: {url}") from last_error

# ================= DISCOVERY ================= #

def extract_individual_test_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    headers = soup.find_all("th", class_="custom__table-heading__title")

    for th in headers:
        if th.get_text(strip=True) != "Individual Test Solutions":
            continue

        table = th.find_parent("table")
        if not table:
            continue

        for a in table.select("a[href]"):
            href = a["href"]
            if "/products/product-catalog/view/" in href:
                links.add(urljoin(BASE_URL, href))

    return links


def discover_all_assessments():
    all_links = set()

    for start in range(0, MAX_START, PAGE_SIZE):
        url = f"{CATALOG_URL}?start={start}&type=1"
        print(f"Discovering page: start={start}")

        html = get_html(url)
        page_links = extract_individual_test_links(html)

        if not page_links:
            print("No more Individual Test Solutions found.")
            break

        before = len(all_links)
        all_links.update(page_links)
        after = len(all_links)

        print(f"  Found {len(page_links)} (total {after})")

        if after == before:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return sorted(all_links)

# ================= DETAIL PARSER ================= #

def scrape_assessment(url):
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    def get_row(title):
        for row in soup.select("div.product-catalogue-training-calendar__row"):
            h4 = row.find("h4")
            if h4 and h4.get_text(strip=True).lower() == title.lower():
                return row
        return None

    def get_row_text(title):
        row = get_row(title)
        if not row:
            return ""
        p = row.find("p")
        return p.get_text(" ", strip=True) if p else ""

    # Name
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else ""

    # Description
    description = get_row_text("Description")

    # Job levels (list)
    job_levels_text = get_row_text("Job levels")
    job_levels = [
        j.strip()
        for j in job_levels_text.split(",")
        if j.strip()
    ]

    # Languages (list)
    languages_text = get_row_text("Languages")
    languages = [
        l.strip()
        for l in languages_text.split(",")
        if l.strip()
    ]

    # Assessment length (minutes)
    assessment_length_minutes = None
    length_row = get_row("Assessment length")
    if length_row:
        p = length_row.find("p")
        if p:
            text = p.get_text(strip=True)
            for token in text.split():
                if token.isdigit():
                    assessment_length_minutes = int(token)
                    break

    # Test Type (unique keys)
    test_type = []
    seen = set()
    for span in soup.select("span.product-catalogue__key"):
        val = span.get_text(strip=True)
        if val and val not in seen:
            seen.add(val)
            test_type.append(val)

    # Remote Testing
    remote_testing = (
        "Yes" if soup.select_one("span.catalogue__circle.-yes") else "No"
    )

    # Downloads
    downloads = []
    downloads_row = get_row("Downloads")
    if downloads_row:
        for li in downloads_row.select("li.product-catalogue__download"):
            a = li.find("a", href=True)
            lang = li.find("p", class_="product-catalogue__download-language")

            downloads.append({
                "title": a.get_text(strip=True) if a else "",
                "url": a["href"] if a else "",
                "language": lang.get_text(strip=True) if lang else "",
            })

    return {
        "name": name,
        "url": url,
        "description": description,
        "job_levels": job_levels,
        "languages": languages,
        "assessment_length_minutes": assessment_length_minutes,
        "test_type": test_type,
        "remote_testing": remote_testing,
        "downloads": downloads,
    }

# ================= PIPELINE ================= #

def build_catalog():
    links = discover_all_assessments()

    print(f"\n✅ Discovered {len(links)} Individual Test Solutions")

    if len(links) != 377:
        raise RuntimeError(f"Expected 377, got {len(links)}")

    data = []

    for i, link in enumerate(links, 1):
        print(f"[{i}/377] Scraping")
        data.append(scrape_assessment(link))
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if len(data) != 377:
        raise RuntimeError("Scrape count mismatch")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n🎉 SUCCESS: raw_catalog.json created with 377 items")

# ================= ENTRY ================= #

if __name__ == "__main__":
    build_catalog()
