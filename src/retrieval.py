from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import basic_clean_text, ensure_output_dir, load_dataset


def search_by_query(texts: pd.Series, query: str, top_k: int, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([basic_clean_text(query)])
    scores = cosine_similarity(query_vec, matrix).ravel()
    top_indices = np.argsort(scores)[::-1][:top_k]
    result = pd.DataFrame({
        "rank": np.arange(1, len(top_indices) + 1),
        "score": scores[top_indices],
        "text": texts.iloc[top_indices].values,
    })
    return result


def search_by_row(texts: pd.Series, query_index: int, top_k: int, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    matrix = vectorizer.fit_transform(texts)
    query_vec = matrix[query_index]
    scores = cosine_similarity(query_vec, matrix).ravel()
    scores[query_index] = -1.0
    top_indices = np.argsort(scores)[::-1][:top_k]
    result = pd.DataFrame({
        "rank": np.arange(1, len(top_indices) + 1),
        "score": scores[top_indices],
        "text": texts.iloc[top_indices].values,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple TF-IDF similarity search for hotel reviews.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--text-column", default="review_text", choices=["review_text", "negative_text", "positive_text"])
    parser.add_argument("--query", default=None, help="Free-text query")
    parser.add_argument("--query-index", type=int, default=None, help="Use one document as query by index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if args.query is None and args.query_index is None:
        raise ValueError("Provide either --query or --query-index.")

    output_dir = ensure_output_dir(args.output_dir)

    df = load_dataset(args.data, sample_size=args.sample_size, random_state=args.random_state)
    df = df[df[args.text_column].fillna("").str.len() > 0].copy()
    df["clean_text"] = df[args.text_column].map(basic_clean_text)
    df = df[df["clean_text"].str.len() > 0].copy().reset_index(drop=True)

    ngram_range = (1, 2) if args.use_bigrams else (1, 1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )

    if args.query is not None:
        results = search_by_query(df["clean_text"], args.query, args.top_k, vectorizer)
        output_name = "retrieval_results_query.csv"
    else:
        if args.query_index < 0 or args.query_index >= len(df):
            raise IndexError(f"query-index must be between 0 and {len(df) - 1}")
        results = search_by_row(df["clean_text"], args.query_index, args.top_k, vectorizer)
        output_name = "retrieval_results_index.csv"

    output_path = Path(output_dir) / output_name
    results.to_csv(output_path, index=False)

    print("Retrieval finished.")
    print(results.to_string(index=False))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
