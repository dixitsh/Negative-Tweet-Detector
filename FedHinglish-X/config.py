import os
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = os.getenv("DATA_PATH", "data/hinglish_sentiment.csv")
    model_name: str = os.getenv("MODEL_NAME", "google/muril-base-cased")
    output_dir: str = os.getenv("OUTPUT_DIR", "artifacts")
    num_labels: int = 3
    max_length: int = int(os.getenv("MAX_LENGTH", "128"))
    num_clients: int = int(os.getenv("NUM_CLIENTS", "8"))
    rounds: int = int(os.getenv("ROUNDS", "10"))
    local_epochs: int = int(os.getenv("LOCAL_EPOCHS", "2"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))
    lr: float = float(os.getenv("LR", "2e-5"))
    lora_r: int = int(os.getenv("LORA_R", "8"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "16"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.1"))
    clip_norm: float = float(os.getenv("CLIP_NORM", "1.0"))
    dp_sigma: float = float(os.getenv("DP_SIGMA", "0.0"))
    dirichlet_alpha: float = float(os.getenv("DIRICHLET_ALPHA", "0.5"))
    seed: int = int(os.getenv("SEED", "42"))
    max_train_samples: int = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
    device: str = os.getenv("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")

CFG = Config()
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
