import os
from dataclasses import dataclass
from typing import List


@dataclass
class Paths:
    data_path: str = os.path.join(os.getcwd(), "placement_qa_dataset_large.csv")
    model_dir: str = os.path.join(os.getcwd(), "models", "bi_encoder")
    index_dir: str = os.path.join(os.getcwd(), "indexes")


@dataclass
class TrainingConfig:
    base_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    train_epochs: int = 1
    train_batch_size: int = 32
    max_train_pairs: int | None = 30000
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    seed: int = 42


@dataclass
class IndexConfig:
    top_k: int = 5


paths = Paths()
train_cfg = TrainingConfig()
index_cfg = IndexConfig()


def ensure_directories() -> None:
    os.makedirs(os.path.dirname(paths.model_dir), exist_ok=True)
    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.index_dir, exist_ok=True)


# Voice configuration
@dataclass
class VoiceConfig:
    wake_phrases: List[str] = ("arya", "arya chat bot", "hello")
    sample_rate: int = 16000
    vosk_model_url: str = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    vosk_model_dir: str = os.path.join(os.getcwd(), "voice_models", "vosk-model-small-en-us-0.15")


voice_cfg = VoiceConfig()

