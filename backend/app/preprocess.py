import json

RAW_PATH = "data/raw_catalog.json"
OUT_PATH = "data/processed_catalog.json"


def normalize_text(v):
    if isinstance(v, list):
        return ", ".join(x.strip() for x in v if isinstance(x, str))
    if isinstance(v, str):
        return v.strip()
    return ""


def normalize_list(v):
    if isinstance(v, list):
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def preprocess():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    processed = []

    for item in raw:
        name = normalize_text(item.get("name"))
        description = normalize_text(item.get("description"))
        job_levels = normalize_text(item.get("job_levels"))

        test_type = normalize_list(item.get("test_type"))

        text = " ".join([
            name,
            description,

            # explicit skill signals
            "technical skills" if any(t in test_type for t in ["K", "A", "C", "S"]) else "",
            "personality behavior soft skills" if any(t in test_type for t in ["P", "B", "D", "E"]) else "",

            # role hints
            "software developer engineer programmer" if "developer" in name.lower() else "",
            "java python coding programming" if "java" in name.lower() else "",
            "communication teamwork collaboration" if any(
                k in description.lower()
                for k in ["communicat", "team", "collaborat"]
            ) else "",
        ])



        processed.append({
            "name": name,
            "url": item.get("url", ""),
            "description": description,
            "assessment_length_minutes": item.get("assessment_length_minutes"),
            "test_type": test_type,
            "remote_testing": item.get("remote_testing"),
            # text used ONLY for embeddings
            "text": text.strip()
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(processed)} items")


if __name__ == "__main__":
    preprocess()
