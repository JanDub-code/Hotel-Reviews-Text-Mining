from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_POSITIVE_PLACEHOLDERS = {
    "no positive",
    "nothing",
    "n a",
    "na",
    "none",
}

DEFAULT_NEGATIVE_PLACEHOLDERS = {
    "no negative",
    "nothing",
    "n a",
    "na",
    "none",
}


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def basic_clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = normalize_whitespace(text)
    return text


def replace_placeholders(series: pd.Series, placeholders: Iterable[str]) -> pd.Series:
    normalized = {p.lower().strip() for p in placeholders}

    def _clean_value(value: object) -> str:
        if not isinstance(value, str):
            return ""
        stripped = value.strip()
        if stripped.lower() in normalized:
            return ""
        return stripped

    return series.fillna("").map(_clean_value)


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "Positive_Review" not in data.columns or "Negative_Review" not in data.columns:
        raise ValueError("Expected columns 'Positive_Review' and 'Negative_Review' in the dataset.")

    data["Positive_Review"] = replace_placeholders(
        data["Positive_Review"], DEFAULT_POSITIVE_PLACEHOLDERS
    )
    data["Negative_Review"] = replace_placeholders(
        data["Negative_Review"], DEFAULT_NEGATIVE_PLACEHOLDERS
    )

    data["review_text"] = (
        data["Positive_Review"].fillna("") + " " + data["Negative_Review"].fillna("")
    ).map(normalize_whitespace)
    data["negative_text"] = data["Negative_Review"].fillna("").map(normalize_whitespace)
    data["positive_text"] = data["Positive_Review"].fillna("").map(normalize_whitespace)
    return data


def load_dataset(csv_path: str | Path, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Place Hotel_Reviews.csv into data/raw/."
        )

    df = pd.read_csv(csv_path)
    df = add_text_columns(df)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    return df


def create_binary_sentiment_labels(
    df: pd.DataFrame,
    score_col: str = "Reviewer_Score",
    negative_threshold: float = 5.0,
    positive_threshold: float = 8.0,
) -> pd.DataFrame:
    if score_col not in df.columns:
        raise ValueError(f"Expected column '{score_col}' in the dataset.")

    data = df.copy()
    score = pd.to_numeric(data[score_col], errors="coerce")
    mask = score.le(negative_threshold) | score.ge(positive_threshold)
    data = data.loc[mask].copy()
    data["label"] = np.where(score.loc[mask] <= negative_threshold, "negative", "positive")
    return data.reset_index(drop=True)


def create_three_class_labels(
    df: pd.DataFrame,
    score_col: str = "Reviewer_Score",
    negative_threshold: float = 5.0,
    positive_threshold: float = 8.0,
) -> pd.DataFrame:
    if score_col not in df.columns:
        raise ValueError(f"Expected column '{score_col}' in the dataset.")

    data = df.copy()
    score = pd.to_numeric(data[score_col], errors="coerce")

    def _map_label(value: float) -> str:
        if value <= negative_threshold:
            return "negative"
        if value >= positive_threshold:
            return "positive"
        return "neutral"

    data["label"] = score.map(_map_label)
    return data.reset_index(drop=True)
