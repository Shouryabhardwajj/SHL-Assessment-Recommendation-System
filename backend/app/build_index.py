import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "data/processed_catalog.json"
INDEX_PATH = "data/faiss.index"
ID_MAP_PATH = "data/id_map.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def build_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    texts = [x["text"] for x in catalog]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(ID_MAP_PATH, "w") as f:
        json.dump(list(range(len(catalog))), f)

    print(f"FAISS index built with {index.ntotal} vectors")


if __name__ == "__main__":
    build_index()
