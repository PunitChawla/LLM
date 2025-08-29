from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from app.config import paths, index_cfg


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(paths.model_dir)
        self.embeddings = np.load(os.path.join(paths.index_dir, "embeddings.npy"))
        self.metadata: List[Dict] = []
        with open(os.path.join(paths.index_dir, "metadata.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[float, Dict]]:
        if top_k is None:
            top_k = index_cfg.top_k
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        results: List[Tuple[float, Dict]] = []
        for idx in top_indices:
            results.append((float(scores[idx]), self.metadata[int(idx)]))
        return results

