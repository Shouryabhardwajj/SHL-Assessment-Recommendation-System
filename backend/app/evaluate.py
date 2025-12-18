import csv
import requests
import time
from collections import defaultdict

API_URL = "http://127.0.0.1:8000/recommend"
TRAIN_CSV = "data/Gen_AI Dataset(Train-Set).csv"


def extract_slug(url: str) -> str:
    if not url:
        return ""
    return url.rstrip("/").split("/")[-1]

def call_api(query):
    for attempt in range(3):
        try:
            return requests.post(
                API_URL,
                json={"query": query, "top_k": 10},
                timeout=90,
            )
        except requests.exceptions.ReadTimeout:
            print("Timeout, retrying...")
            time.sleep(2)
    raise RuntimeError("API failed after retries")



def evaluate():
    grouped = defaultdict(set)

    with open(TRAIN_CSV, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["Query"].strip()
            slug = extract_slug(row["Assessment_url"])
            grouped[query].add(slug)

    total_recall = 0.0

    for query, relevant_slugs in grouped.items():
        resp = call_api(query)
        preds = resp.json()["recommended_assessments"]
        predicted_slugs = {
            extract_slug(p["url"]) for p in preds
        }

        hits = len(relevant_slugs & predicted_slugs)
        recall = hits / len(relevant_slugs)

        total_recall += recall

        print(f"Query: {query}")
        print(f"Relevant slugs: {relevant_slugs}")
        print(f"Predicted slugs: {predicted_slugs}")
        print(f"Hits: {hits}")
        print(f"Recall@10: {recall:.3f}\n")

    mean_recall = total_recall / len(grouped)
    print("Mean Recall@10:", round(mean_recall, 4))


if __name__ == "__main__":
    evaluate()
