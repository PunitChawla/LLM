from __future__ import annotations

import os
import math
import random
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from app.config import paths, train_cfg, ensure_directories
from app.data_utils import load_dataset, expand_training_pairs


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_training_data(max_pairs: int | None) -> List[InputExample]:
    df = load_dataset(paths.data_path)
    pairs: List[Tuple[str, str]] = expand_training_pairs(df, max_pairs=max_pairs)
    examples: List[InputExample] = []
    for q, c in pairs:
        examples.append(InputExample(texts=[q, c]))
    return examples


def train_model() -> str:
    ensure_directories()
    set_seed(train_cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_name = train_cfg.base_embedding_model
    model = SentenceTransformer(base_model_name, device=device)

    train_examples = prepare_training_data(train_cfg.max_train_pairs)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_cfg.train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    num_steps_per_epoch = math.ceil(len(train_examples) / train_cfg.train_batch_size)
    warmup_steps = max(1, int(num_steps_per_epoch * train_cfg.train_epochs * train_cfg.warmup_ratio))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=train_cfg.train_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        use_amp=True,
        optimizer_params={"lr": train_cfg.learning_rate},
        output_path=paths.model_dir,
        save_best_model=True,
    )

    return paths.model_dir


if __name__ == "__main__":
    out = train_model()
    print(f"Model trained and saved to: {out}")

