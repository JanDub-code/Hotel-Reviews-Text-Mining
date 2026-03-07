from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from common import basic_clean_text, ensure_output_dir, load_dataset


def top_terms_per_cluster(model: KMeans, feature_names: np.ndarray, top_n: int = 12) -> dict[str, list[str]]:
    result = {}
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    for cluster_id in range(model.n_clusters):
        term_indices = order_centroids[cluster_id, :top_n]
        result[f"cluster_{cluster_id}"] = feature_names[term_indices].tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster negative hotel reviews with K-Means.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    df = load_dataset(args.data, sample_size=args.sample_size, random_state=args.random_state)
    rows_loaded = int(len(df))
    df = df[df["negative_text"].fillna("").str.len() > 0].copy()
    rows_after_text_filter = int(len(df))
    df["clean_text"] = df["negative_text"].map(basic_clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    rows_after_cleaning = int(len(df))

    ngram_range = (1, 2) if args.use_bigrams else (1, 1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    X = vectorizer.fit_transform(df["clean_text"])

    model = KMeans(n_clusters=args.k, n_init=10, random_state=args.random_state)
    labels = model.fit_predict(X)

    silhouette = None
    if X.shape[0] >= args.k * 3:
        sample_size = min(5000, X.shape[0])
        silhouette = silhouette_score(X, labels, sample_size=sample_size, random_state=args.random_state)

    feature_names = vectorizer.get_feature_names_out()
    top_terms = top_terms_per_cluster(model, feature_names)
    cluster_sizes = {
        f"cluster_{cluster_id}": int(np.sum(labels == cluster_id))
        for cluster_id in range(args.k)
    }

    representative_reviews = {}
    distances = model.transform(X)
    for cluster_id in range(args.k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            representative_reviews[f"cluster_{cluster_id}"] = []
            continue
        cluster_distances = distances[cluster_indices, cluster_id]
        top_local = cluster_indices[np.argsort(cluster_distances)[:3]]
        representative_reviews[f"cluster_{cluster_id}"] = df.iloc[top_local]["negative_text"].tolist()

    results = {
        "config": vars(args),
        "dataset": {
            "rows_loaded": rows_loaded,
            "rows_after_text_filter": rows_after_text_filter,
            "rows_after_cleaning": rows_after_cleaning,
            "n_documents": int(X.shape[0]),
        },
        "preprocessing": {
            "text_column": "negative_text",
            "cleaning_steps": [
                "placeholder replacement for Negative_Review",
                "drop empty negative_text",
                "lowercasing",
                "remove non-letter characters",
                "normalize whitespace",
                "drop empty cleaned texts",
            ],
            "vectorizer": {
                "method": "tfidf",
                "stop_words": "english",
                "ngram_range": list(ngram_range),
                "min_df": args.min_df,
                "max_df": args.max_df,
                "max_features": args.max_features,
            },
        },
        "silhouette_score": silhouette,
        "cluster_sizes": cluster_sizes,
        "top_terms": top_terms,
        "representative_reviews": representative_reviews,
    }

    output_path = Path(output_dir) / "clustering_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_rows = []
    for cluster_id in range(args.k):
        cluster_name = f"cluster_{cluster_id}"
        summary_rows.append(
            {
                "cluster_id": cluster_name,
                "size": cluster_sizes[cluster_name],
                "top_terms": ", ".join(top_terms[cluster_name]),
                "representative_review_1": representative_reviews[cluster_name][0]
                if len(representative_reviews[cluster_name]) > 0
                else "",
                "representative_review_2": representative_reviews[cluster_name][1]
                if len(representative_reviews[cluster_name]) > 1
                else "",
                "representative_review_3": representative_reviews[cluster_name][2]
                if len(representative_reviews[cluster_name]) > 2
                else "",
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")
    summary_path = Path(output_dir) / "clustering_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Clustering finished.")
    print(f"Documents clustered: {X.shape[0]}")
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.4f}")
    for cluster_name, terms in top_terms.items():
        print(f"{cluster_name}: {', '.join(terms)}")
    print(f"Detailed results saved to: {output_path}")
    print(f"Cluster summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
