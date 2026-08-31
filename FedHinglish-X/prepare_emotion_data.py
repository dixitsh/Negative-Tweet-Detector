"""Prepare fine-grained emotion data while preserving original labels.

Input CSV must contain text + emotion. This script does not invent emotion labels.
It normalizes label names and applies the optional fuzzy text normalizer.
"""
import argparse
import os
import pandas as pd
from fuzzy_normalizer import fuzzy_normalize
from emotion_schema import EMOTION_LABELS, normalize_emotion_label


def prepare(input_path: str, output_path: str, text_col: str, emotion_col: str,
            threshold: float = .86):
    df = pd.read_csv(input_path)
    missing = [c for c in (text_col, emotion_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")
    df = df[[text_col, emotion_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[emotion_col] = df[emotion_col].map(normalize_emotion_label)
    df = df[df[emotion_col].isin(EMOTION_LABELS)]
    df = df[df[text_col].str.len() > 0].drop_duplicates(subset=[text_col])
    df["normalized_text"] = df[text_col].map(lambda x: fuzzy_normalize(x, threshold))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    print(df[emotion_col].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default="data/hinglish_emotion_prepared.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--emotion-col", default="emotion")
    p.add_argument("--threshold", type=float, default=.86)
    a = p.parse_args()
    prepare(a.input, a.output, a.text_col, a.emotion_col, a.threshold)
