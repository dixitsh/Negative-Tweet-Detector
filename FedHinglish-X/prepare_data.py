"""Prepare a generic Hinglish sentiment CSV and apply fuzzy normalization."""
import argparse
import os
import pandas as pd
from fuzzy_normalizer import fuzzy_normalize

LABEL_MAP = {"positive":"positive", "pos":"positive", "2":"positive",
             "neutral":"neutral", "neu":"neutral", "1":"neutral",
             "negative":"negative", "neg":"negative", "0":"negative"}


def prepare(input_path: str, output_path: str, text_col: str = "text", label_col: str = "label", threshold: float = .86):
    df = pd.read_csv(input_path)
    if text_col not in df or label_col not in df:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns. Found: {list(df.columns)}")
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    df[label_col] = df[label_col].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna().drop_duplicates(subset=[text_col]).reset_index(drop=True)
    df["normalized_text"] = df[text_col].map(lambda x: fuzzy_normalize(x, threshold))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    print(df[label_col].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default="data/hinglish_sentiment_prepared.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--threshold", type=float, default=.86)
    a = p.parse_args()
    prepare(a.input, a.output, a.text_col, a.label_col, a.threshold)
