from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_output_dir


def build_score_distribution(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(data_path, usecols=["Reviewer_Score"])
    distribution = (
        df["Reviewer_Score"]
        .value_counts()
        .sort_index()
        .rename_axis("reviewer_score")
        .reset_index(name="count")
    )
    csv_path = output_dir / "reviewer_score_distribution.csv"
    distribution.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(distribution["reviewer_score"].astype(str), distribution["count"], color="#2c5f77")
    plt.title("Reviewer Score Distribution")
    plt.xlabel("Reviewer Score")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    png_path = output_dir / "reviewer_score_distribution.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    return csv_path, png_path


def build_model_results_table(baseline_dir: Path, feature_selection_dir: Path, output_dir: Path) -> Path:
    baseline_df = pd.read_csv(baseline_dir / "classification_summary.csv").copy()
    baseline_df.insert(0, "scenario", "baseline_bigram")

    fs_summary = pd.read_csv(feature_selection_dir / "feature_selection_summary.csv")
    best_row = fs_summary.sort_values("macro_f1", ascending=False).iloc[0]
    best_run_dir = Path(str(best_row["output_dir"]))
    best_label = f"feature_selection_k_{int(best_row['feature_selection_k'])}"
    best_df = pd.read_csv(best_run_dir / "classification_summary.csv").copy()
    best_df.insert(0, "scenario", best_label)

    model_results_df = pd.concat([baseline_df, best_df], ignore_index=True)
    output_path = output_dir / "model_results_table.csv"
    model_results_df.to_csv(output_path, index=False)
    return output_path


def build_top_features_table(top_features_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(top_features_path)
    positive = (
        df[df["sentiment"] == "positive"][["rank", "term"]]
        .rename(columns={"term": "positive_term"})
        .sort_values("rank")
    )
    negative = (
        df[df["sentiment"] == "negative"][["rank", "term"]]
        .rename(columns={"term": "negative_term"})
        .sort_values("rank")
    )
    merged = positive.merge(negative, on="rank", how="outer").sort_values("rank")
    output_path = output_dir / "top_features_table.csv"
    merged.to_csv(output_path, index=False)
    return output_path


def build_cluster_summary_table(cluster_summary_path: Path, cluster_labels_path: Path, output_dir: Path) -> Path:
    summary_df = pd.read_csv(cluster_summary_path)
    labels_df = pd.read_csv(cluster_labels_path)
    merged = summary_df.merge(labels_df, on="cluster_id", how="left")
    output_path = output_dir / "cluster_summary_table.csv"
    merged.to_csv(output_path, index=False)
    return output_path


def build_retrieval_examples_table(
    retrieval_examples_path: Path, query_assessment_path: Path, output_dir: Path
) -> Path:
    examples_df = pd.read_csv(retrieval_examples_path)
    assessment_df = pd.read_csv(query_assessment_path)
    top_examples_df = examples_df[examples_df["rank"] <= 3].copy()
    merged = top_examples_df.merge(assessment_df, on="query_id", how="left").sort_values(
        ["query_id", "rank"]
    )
    output_path = output_dir / "retrieval_examples_table.csv"
    merged.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 8 visuals and summary tables.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--baseline-classification-dir", default="outputs/classification_bigram")
    parser.add_argument("--feature-selection-dir", default="outputs/feature_selection")
    parser.add_argument("--clustering-summary", default="outputs/clustering/k_7/clustering_summary.csv")
    parser.add_argument("--cluster-labels", default="outputs/clustering/k_7/cluster_labels.csv")
    parser.add_argument("--retrieval-examples", default="outputs/retrieval/retrieval_examples.csv")
    parser.add_argument("--retrieval-assessment", default="outputs/retrieval/query_assessment.csv")
    parser.add_argument("--top-features", default="outputs/feature_selection/feature_selection_top_features.csv")
    parser.add_argument("--output-dir", default="outputs/visuals")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    score_csv, score_png = build_score_distribution(Path(args.data), output_dir)
    model_results_path = build_model_results_table(
        Path(args.baseline_classification_dir),
        Path(args.feature_selection_dir),
        output_dir,
    )
    top_features_table_path = build_top_features_table(Path(args.top_features), output_dir)
    cluster_summary_table_path = build_cluster_summary_table(
        Path(args.clustering_summary),
        Path(args.cluster_labels),
        output_dir,
    )
    retrieval_examples_table_path = build_retrieval_examples_table(
        Path(args.retrieval_examples),
        Path(args.retrieval_assessment),
        output_dir,
    )

    summary = {
        "reviewer_score_distribution_csv": str(score_csv),
        "reviewer_score_distribution_png": str(score_png),
        "model_results_table": str(model_results_path),
        "top_features_table": str(top_features_table_path),
        "cluster_summary_table": str(cluster_summary_table_path),
        "retrieval_examples_table": str(retrieval_examples_table_path),
    }
    summary_path = output_dir / "phase8_outputs.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Phase 8 outputs generated.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
