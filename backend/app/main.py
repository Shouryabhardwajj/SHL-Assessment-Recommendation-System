from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Literal

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------
# Constants & Configuration
# --------------------------------------------------------------------

TEST_TYPE_MAP: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}

FLAGSHIP_KEYWORDS: dict[str, list[str]] = {
    "soft": [
        "opq",
        "interpersonal",
        "english",
        "verbal",
        "personality",
    ],
    "tech": [
        "verify",
        "numerical",
        "deductive",
        "inductive",
        "ability",
    ],
}

DATA_PATH = Path("data/processed_catalog.json")
INDEX_PATH = Path("data/faiss.index")
ID_MAP_PATH = Path("data/id_map.json")

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 20
MIN_TOP_K = 5
MAX_TOP_K = 10


# --------------------------------------------------------------------
# App Initialization (lazy resources)
# --------------------------------------------------------------------

app = FastAPI(title="SHL Assessment Recommendation API")

model: SentenceTransformer | None = None
index: faiss.Index | None = None
catalog: list[dict] | None = None
id_map: list[int] | None = None


def load_resources() -> None:
    """Lazy-load model, FAISS index, and catalog/id_map once per worker."""
    global model, index, catalog, id_map

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    if index is None:
        index = faiss.read_index(str(INDEX_PATH))

    if catalog is None:
        with DATA_PATH.open("r", encoding="utf-8") as f:
            catalog = json.load(f)

    if id_map is None:
        with ID_MAP_PATH.open("r", encoding="utf-8") as f:
            # id_map is a list mapping FAISS index -> catalog index
            id_map = json.load(f)


# --------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------


class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10


class RecommendedAssessment(BaseModel):
    name: str
    url: str
    description: str
    duration: int | None
    test_type: List[str]
    adaptive_support: Literal["Yes", "No"]
    remote_support: Literal["Yes", "No"]


class RecommendResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]


# --------------------------------------------------------------------
# Health & Warmup Routes
# --------------------------------------------------------------------


@app.get("/warmup")
def warmup() -> dict[str, str]:
    load_resources()
    return {"status": "warmed"}


@app.get("/health")
def health() -> dict[str, int | str]:
    count = len(catalog) if catalog is not None else 0
    return {"status": "healthy", "items_loaded": count}


# --------------------------------------------------------------------
# Query Expansion & Intent
# --------------------------------------------------------------------

Intent = Literal["tech", "soft", "both", "general"]


def expand_query(query: str) -> str:
    q = query.lower()
    expansions: list[str] = []

    # --- TECH ROLES ---
    if any(k in q for k in ["java", "spring", "backend", "developer"]):
        expansions.append(
            "java programming backend development spring sql api microservices"
        )

    if any(k in q for k in ["python", "data", "ml", "ai"]):
        expansions.append(
            "python data analysis machine learning statistics algorithms"
        )

    if any(k in q for k in ["frontend", "ui", "ux"]):
        expansions.append(
            "frontend javascript html css react user interface"
        )

    # --- SOFT SKILLS ---
    if any(k in q for k in ["collaboration", "team", "stakeholder"]):
        expansions.append(
            "communication teamwork interpersonal skills collaboration"
        )

    if any(k in q for k in ["leadership", "manager", "manage"]):
        expansions.append(
            "leadership people management coaching decision making"
        )

    # --- GENERIC ROLE SIGNALS ---
    if "developer" in q:
        expansions.append("technical skills coding software engineering")

    if "customer" in q or "service" in q:
        expansions.append("customer service empathy communication behavior")

    if not expansions:
        return q

    return f"{q} {' '.join(expansions)}"


def detect_intent(query: str) -> Intent:
    q = query.lower()

    tech_keywords: tuple[str, ...] = (
        # programming languages
        "java", "python", "c++", "c#", "javascript", "typescript", "sql",
        "r", "go", "ruby", "php", "scala", "kotlin",
        # frameworks / tools
        "spring", "react", "angular", "node", "django", "flask",
        "aws", "azure", "gcp", "docker", "kubernetes",
        "hadoop", "spark", "tableau", "power bi",
        # roles
        "developer", "engineer", "programmer", "architect",
        "backend", "frontend", "full stack", "fullstack",
        "data scientist", "data engineer", "ml engineer",
        # technical skills
        "coding", "programming", "software", "debugging",
        "database", "api", "microservices", "cloud",
        "automation", "testing", "devops", "ci cd",
        # IT / systems
        "linux", "unix", "networking", "security",
        "cyber", "infrastructure",
    )

    soft_keywords: tuple[str, ...] = (
        # communication
        "communication", "communicate", "presentation", "listening",
        "verbal", "written", "email", "documentation",
        # teamwork / behavior
        "collaboration", "team", "teamwork", "interpersonal",
        "relationship", "stakeholder",
        # leadership / management
        "leadership", "manager", "management", "supervisor",
        "people management", "coaching", "mentoring",
        # personality traits
        "personality", "attitude", "behavior", "behaviour",
        "emotional intelligence", "motivation", "work style",
        # workplace skills
        "time management", "problem solving", "decision making",
        "critical thinking", "adaptability", "flexibility",
        "conflict", "stress", "resilience", "ethics",
    )

    has_tech = any(t in q for t in tech_keywords)
    has_soft = any(t in q for t in soft_keywords)

    if has_tech and has_soft:
        return "both"
    if has_tech:
        return "tech"
    if has_soft:
        return "soft"
    return "general"


# --------------------------------------------------------------------
# Scoring Helpers
# --------------------------------------------------------------------


def family_boost(item: dict, query: str) -> float:
    q = query.lower()
    name = item.get("name", "").lower()

    boost = 0.0

    # Programming language alignment
    for lang in ("java", "python", "sql", "c++", "c#", "javascript", "react", "angular"):
        if lang in q and lang in name:
            boost += 1.5

    # Role-based families
    if "developer" in q or "engineer" in q:
        if any(k in name for k in ("programming", "development", "coding", "software")):
            boost += 1.2

    if "manager" in q or "leadership" in q:
        if any(k in name for k in ("opq", "motivation", "leadership", "management", "scenario")):
            boost += 1.2

    if "graduate" in q or "entry" in q:
        if any(k in name for k in ("ability", "verify", "aptitude", "cognitive")):
            boost += 1.2

    return boost


# --------------------------------------------------------------------
# Core Utilities
# --------------------------------------------------------------------


def search_index(query: str, k: int = DEFAULT_K) -> np.ndarray:
    assert model is not None and index is not None
    vec = model.encode([query], normalize_embeddings=True)
    _, indices = index.search(vec, k)
    return indices[0]


def map_test_types(codes: Iterable[str]) -> list[str]:
    return [TEST_TYPE_MAP[c] for c in codes if c in TEST_TYPE_MAP]


def has_adaptive_support(item: dict) -> bool:
    if "A" in item.get("test_type", []):
        return True
    text = f"{item.get('name', '')}{item.get('description', '')}".lower()
    return "adaptive" in text


def parse_duration(val) -> int | None:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        m = re.search(r"\d+", val)
        if m:
            return int(m.group())
    return None


def extract_slug(url: str) -> str:
    if not url:
        return ""
    return url.rstrip("/").split("/")[-1]


def build_response(items: list[dict]) -> list[RecommendedAssessment]:
    results: list[RecommendedAssessment] = []

    for item in items:
        results.append(
            RecommendedAssessment(
                name=item.get("name", "").strip(),
                url=extract_slug(item.get("url", "")),
                description=item.get("description", "").strip(),
                duration=parse_duration(item.get("assessment_length_minutes")),
                test_type=map_test_types(item.get("test_type", [])),
                adaptive_support="Yes" if has_adaptive_support(item) else "No",
                remote_support="Yes" if item.get("remote_testing") == "Yes" else "No",
            )
        )

    return results


# --------------------------------------------------------------------
# Recommendation Route
# --------------------------------------------------------------------


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    load_resources()
    assert catalog is not None and id_map is not None

    top_k = max(MIN_TOP_K, min(req.top_k, MAX_TOP_K))

    expanded_query = expand_query(req.query)
    indices = search_index(expanded_query, k=DEFAULT_K)

    intent = detect_intent(req.query)

    scored: list[tuple[float, dict]] = []

    for rank, idx in enumerate(indices):
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= len(id_map):
            continue

        catalog_idx = id_map[idx_int]
        if catalog_idx < 0 or catalog_idx >= len(catalog):
            continue

        item = catalog[catalog_idx]
        test_types = item.get("test_type", [])

        score = 1.0 / (rank + 1)

        if intent == "tech" and any(t in test_types for t in ("K", "A", "C", "S")):
            score += 0.6

        if intent == "soft" and any(t in test_types for t in ("P", "B", "D", "E")):
            score += 0.6

        if intent == "both":
            score += 0.2

        score += family_boost(item, req.query)

        name = item.get("name", "").lower()
        if intent in FLAGSHIP_KEYWORDS:
            for kw in FLAGSHIP_KEYWORDS[intent]:
                if kw in name:
                    score += 0.8

        scored.append((score, item))

    seen: dict[str, tuple[float, dict]] = {}
    for score, item in scored:
        url = item.get("url", "")
        if not url:
            continue
        if url not in seen or score > seen[url][0]:
            seen[url] = (score, item)

    final_items = sorted(
        seen.values(),
        key=lambda x: x[0],
        reverse=True,
    )

    selected = [item for _, item in final_items[:top_k]]

    return RecommendResponse(
        recommended_assessments=build_response(selected)
    )
