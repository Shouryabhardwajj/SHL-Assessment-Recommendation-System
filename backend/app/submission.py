import csv
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/recommend"

TEST_CSV = Path("data/Gen_AI Dataset(Test-Set).csv")
OUT_CSV = Path("data/submission.csv")

TOP_K = 10

SHL_BASE = "https://www.shl.com/products/product-catalog/view/"


def normalize_url(u: str) -> str:
    """Normalize catalog slug or partial URL to full SHL product URL."""
    u = u.strip()

    # If already a full URL, keep as-is
    if u.startswith("http"):
        return u

    # handle cases like "python-new" or "/python-new/"
    u = u.strip("/")

    return f"{SHL_BASE}{u}/"


def submission() -> None:
    rows: list[dict[str, str]] = []

    with TEST_CSV.open(encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)

        if "Query" not in reader.fieldnames:
            raise ValueError(f"Expected 'Query' column, found {reader.fieldnames}")

        for row in reader:
            query = row["Query"].strip()

            resp = requests.post(
                API_URL,
                json={"query": query, "top_k": TOP_K},
                timeout=60,
            )
            resp.raise_for_status()

            data = resp.json()["recommended_assessments"]

            if not (5 <= len(data) <= 10):
                raise ValueError(f"Query '{query}' returned {len(data)} results")

            for item in data:
                rows.append(
                    {
                        "Query": query,
                        "Assessment_url": normalize_url(item["url"]),
                    }
                )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Query", "Assessment_url"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"submission.csv created with {len(rows)} rows")


if __name__ == "__main__":
    submission()
