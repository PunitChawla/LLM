from __future__ import annotations

import os
import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import paths, ensure_directories
from app.data_utils import load_dataset, records_with_context


def build_index() -> str:
    ensure_directories()

    model = SentenceTransformer(paths.model_dir)
    df = load_dataset(paths.data_path)
    records: List[Dict] = records_with_context(df)

    texts = [r["context_text"] for r in records]
    embeddings = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    os.makedirs(paths.index_dir, exist_ok=True)
    vec_path = os.path.join(paths.index_dir, "embeddings.npy")
    meta_path = os.path.join(paths.index_dir, "metadata.jsonl")

    np.save(vec_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return paths.index_dir


if __name__ == "__main__":
    out = build_index()
    print(f"Index built at: {out}")

