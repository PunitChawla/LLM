import os
from dataclasses import dataclass
from typing import List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading .env file


@dataclass
class Paths:
    data_path: str = os.path.join(os.getcwd(), "placment_qa_data_set_2.csv")
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
    sample_rate: int = 16000
    # Google Cloud Speech-to-Text configuration
    google_cloud_api_key: str = os.getenv("GOOGLE_CLOUD_API_KEY", "")
    google_cloud_project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")


voice_cfg = VoiceConfig()

