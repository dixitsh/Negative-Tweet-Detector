import os, re, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from config import CFG, LABEL2ID


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " USER ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataframe(path=None):
    path = path or CFG.data_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md")
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("tweet") or cols.get("sentence")
    label_col = cols.get("label") or cols.get("sentiment") or cols.get("polarity")
    if not text_col or not label_col:
        raise ValueError("CSV must contain text/tweet/sentence and label/sentiment/polarity columns")
    df = df[[text_col, label_col]].rename(columns={text_col:"text", label_col:"label"}).dropna()
    df["text"] = df["text"].map(normalize_text)
    def map_label(x):
        s = str(x).strip().lower()
        if s in LABEL2ID: return LABEL2ID[s]
        if s in {"0","negative","neg"}: return 0
        if s in {"1","neutral","neu"}: return 1
        if s in {"2","positive","pos"}: return 2
        raise ValueError(f"Unknown label: {x}")
    df["label"] = df["label"].map(map_label)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if CFG.max_train_samples and len(df) > CFG.max_train_samples:
        df, _ = train_test_split(df, train_size=CFG.max_train_samples, stratify=df.label, random_state=CFG.seed)
    return df.reset_index(drop=True)


def split_dataframe(df):
    train, temp = train_test_split(df, test_size=0.2, stratify=df.label, random_state=CFG.seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp.label, random_state=CFG.seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.texts = frame.text.tolist(); self.labels = frame.label.astype(int).tolist(); self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=CFG.max_length, return_tensors="pt")
        return {k:v.squeeze(0) for k,v in enc.items()} | {"labels": torch.tensor(self.labels[idx], dtype=torch.long)}


def dirichlet_partition(frame, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labels = frame.label.to_numpy()
    clients = [[] for _ in range(num_clients)]
    for label in sorted(np.unique(labels)):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for client, part in zip(clients, np.split(idx, cuts)): client.extend(part.tolist())
    return [frame.iloc[sorted(ids)].reset_index(drop=True) for ids in clients]
