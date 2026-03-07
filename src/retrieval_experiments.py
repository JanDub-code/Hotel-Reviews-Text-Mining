from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import basic_clean_text, ensure_output_dir, load_dataset


DEFAULT_QUERIES = [
    "dirty room and noisy street",
    "friendly staff and great location",
    "small bathroom and old furniture",
    "excellent breakfast and comfortable bed",
    "slow wifi and poor service",
    "good hotel",
]


def load_queries(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_QUERIES
    query_path = Path(path)
    queries = [
        line.strip()
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise ValueError("The query file is empty.")
    return queries


def shorten(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TF-IDF retrieval index and run multiple hotel-review queries.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--text-column", default="review_text", choices=["review_text", "positive_text", "negative_text"])
    parser.add_argument("--queries-file", default=None, help="Optional text file with one query per line")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/retrieval")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    queries = load_queries(args.queries_file)

    df = load_dataset(args.data, sample_size=args.sample_size, random_state=args.random_state)
    rows_loaded = int(len(df))
    df = df[df[args.text_column].fillna("").str.len() > 0].copy()
    rows_after_text_filter = int(len(df))
    df["clean_text"] = df[args.text_column].map(basic_clean_text)
    df = df[df["clean_text"].str.len() > 0].copy().reset_index(drop=True)
    rows_after_cleaning = int(len(df))

    ngram_range = (1, 2) if args.use_bigrams else (1, 1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    matrix = vectorizer.fit_transform(df["clean_text"])

    save_npz(output_dir / "tfidf_index.npz", matrix)
    documents_df = pd.DataFrame(
        {
            "doc_id": np.arange(len(df)),
            "hotel_name": df["Hotel_Name"].fillna(""),
            "reviewer_score": df["Reviewer_Score"],
            "text_excerpt": df[args.text_column].fillna("").map(shorten),
        }
    )
    documents_path = output_dir / "retrieval_documents.csv"
    documents_df.to_csv(documents_path, index=False)

    results_rows = []
    query_summary_rows = []
    for query_id, query in enumerate(queries, start=1):
        query_vec = vectorizer.transform([basic_clean_text(query)])
        scores = cosine_similarity(query_vec, matrix).ravel()
        top_indices = np.argsort(scores)[::-1][: args.top_k]
        top_scores = scores[top_indices]

        query_summary_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "top_score": float(top_scores[0]) if len(top_scores) else 0.0,
                "mean_top_k_score": float(top_scores.mean()) if len(top_scores) else 0.0,
            }
        )

        for rank, doc_index in enumerate(top_indices, start=1):
            results_rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "rank": rank,
                    "score": float(scores[doc_index]),
                    "doc_id": int(doc_index),
                    "hotel_name": df.iloc[doc_index]["Hotel_Name"],
                    "reviewer_score": float(df.iloc[doc_index]["Reviewer_Score"]),
                    "text_excerpt": shorten(str(df.iloc[doc_index][args.text_column])),
                }
            )

    results_df = pd.DataFrame(results_rows)
    results_path = output_dir / "retrieval_examples.csv"
    results_df.to_csv(results_path, index=False)

    query_summary_df = pd.DataFrame(query_summary_rows).sort_values(
        ["top_score", "mean_top_k_score"], ascending=False
    )
    query_summary_path = output_dir / "retrieval_query_summary.csv"
    query_summary_df.to_csv(query_summary_path, index=False)

    metadata = {
        "rows_loaded": rows_loaded,
        "rows_after_text_filter": rows_after_text_filter,
        "rows_after_cleaning": rows_after_cleaning,
        "text_column": args.text_column,
        "top_k": args.top_k,
        "use_bigrams": args.use_bigrams,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "max_features": args.max_features,
        "matrix_shape": list(matrix.shape),
        "n_queries": len(queries),
    }
    metadata_path = output_dir / "retrieval_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_lines = [
        "# Phase 7 summary",
        "",
        "## Retrieval index",
        f"- text column: `{args.text_column}`",
        f"- matrix shape: `{matrix.shape[0]} x {matrix.shape[1]}`",
        f"- TF-IDF bigrams enabled: `{args.use_bigrams}`",
        f"- number of queries: `{len(queries)}`",
        "",
        "## Output files",
        f"- sparse TF-IDF index: `{output_dir / 'tfidf_index.npz'}`",
        f"- document metadata: `{documents_path}`",
        f"- retrieval examples: `{results_path}`",
        f"- query summary: `{query_summary_path}`",
    ]
    summary_md_path = output_dir / "phase7_summary.md"
    summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Retrieval experiments finished.")
    print(query_summary_df.to_string(index=False))
    print(f"Index saved to: {output_dir / 'tfidf_index.npz'}")
    print(f"Examples saved to: {results_path}")
    print(f"Query summary saved to: {query_summary_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
