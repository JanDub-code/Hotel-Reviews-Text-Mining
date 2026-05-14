from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification import build_models
from common import (
    basic_clean_text,
    create_binary_sentiment_labels,
    ensure_output_dir,
    load_dataset,
)


MODEL_LABELS = {
    "naive_bayes": "Multinomial NB",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
}


def model_score_for_label(pipeline: Pipeline, texts: pd.Series, target_label: str) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        return probabilities[:, classes.index(target_label)]

    decision = pipeline.decision_function(texts)
    if np.ndim(decision) == 1:
        # In the binary case scikit-learn returns scores for classes_[1].
        return decision if classes[1] == target_label else -decision

    return decision[:, classes.index(target_label)]


def build_classification_curves(
    data_path: Path,
    output_dir: Path,
    feature_selection_k: int,
    random_state: int,
    save_curve_points: bool = False,
) -> Path:
    df = load_dataset(data_path, sample_size=None, random_state=random_state)
    df = create_binary_sentiment_labels(df)
    df = df[df["review_text"].fillna("").str.len() > 0].copy()
    df["clean_text"] = df["review_text"].map(basic_clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=random_state,
    )
    y_negative = (y_test == "negative").astype(int).to_numpy()
    negative_prevalence = float(y_negative.mean())

    metrics_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for model_name, classifier in build_models(random_state=random_state).items():
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            max_features=50000,
        )
        pipeline = Pipeline(
            [
                ("vectorizer", vectorizer),
                ("feature_selection", SelectKBest(score_func=chi2, k=feature_selection_k)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        scores = model_score_for_label(pipeline, x_test, "negative")

        precision, recall, pr_thresholds = precision_recall_curve(y_negative, scores)
        fpr, tpr, roc_thresholds = roc_curve(y_negative, scores)
        average_precision = average_precision_score(y_negative, scores)
        roc_auc = roc_auc_score(y_negative, scores)

        pr_curves[model_name] = (recall, precision, average_precision)
        roc_curves[model_name] = (fpr, tpr, roc_auc)
        metrics_rows.append(
            {
                "model": model_name,
                "average_precision_negative": average_precision,
                "roc_auc_negative": roc_auc,
                "negative_prevalence": negative_prevalence,
            }
        )

        if save_curve_points:
            for idx, (recall_value, precision_value) in enumerate(zip(recall, precision)):
                curve_rows.append(
                    {
                        "curve": "precision_recall",
                        "model": model_name,
                        "point": idx,
                        "x": recall_value,
                        "y": precision_value,
                        "threshold": pr_thresholds[idx] if idx < len(pr_thresholds) else np.nan,
                    }
                )
            for idx, (fpr_value, tpr_value) in enumerate(zip(fpr, tpr)):
                curve_rows.append(
                    {
                        "curve": "roc",
                        "model": model_name,
                        "point": idx,
                        "x": fpr_value,
                        "y": tpr_value,
                        "threshold": roc_thresholds[idx] if idx < len(roc_thresholds) else np.nan,
                    }
                )

    metrics_path = output_dir / "classification_curve_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    if save_curve_points:
        pd.DataFrame(curve_rows).to_csv(output_dir / "classification_curve_points.csv", index=False)

    plt.figure(figsize=(8.2, 5.4))
    for model_name, (recall, precision, average_precision) in pr_curves.items():
        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} (AP={average_precision:.3f})",
        )
    plt.axhline(
        negative_prevalence,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Negative prevalence={negative_prevalence:.3f}",
    )
    plt.xlabel("Recall for negative reviews")
    plt.ylabel("Precision for negative reviews")
    plt.title("Precision-Recall curves for negative-review detection")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "classification_pr_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.2, 5.4))
    for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} (AUC={roc_auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for negative-review detection")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "classification_roc_curve.png", dpi=180)
    plt.close()

    return metrics_path


def build_feature_selection_plot(feature_selection_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(feature_selection_path).sort_values("feature_selection_k")
    labels = ["baseline" if value == 0 else f"{int(value / 1000)}k" for value in df["feature_selection_k"]]
    x = np.arange(len(df))

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(x, df["macro_f1"], marker="o", linewidth=2, label="Macro F1")
    plt.plot(x, df["accuracy"], marker="o", linewidth=2, label="Accuracy")
    plt.xticks(x, labels)
    plt.xlabel("Number of selected features (k)")
    plt.ylabel("Score")
    plt.title("Feature selection effect for Linear SVM")
    y_min = max(0.0, float(min(df["macro_f1"].min(), df["accuracy"].min())) - 0.03)
    y_max = min(1.0, float(max(df["macro_f1"].max(), df["accuracy"].max())) + 0.02)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = output_dir / "feature_selection_linear_svm.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def build_clustering_plot(clustering_comparison_path: Path, output_dir: Path, selected_k: int) -> Path:
    df = pd.read_csv(clustering_comparison_path).sort_values("k")

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(df["k"], df["silhouette_score"], marker="o", linewidth=2, color="#2c5f77")
    best_row = df.sort_values("silhouette_score", ascending=False).iloc[0]
    selected_row = df[df["k"] == selected_k]
    plt.scatter([best_row["k"]], [best_row["silhouette_score"]], color="#b23b3b", zorder=3)
    plt.annotate(
        f"highest silhouette: k={int(best_row['k'])}",
        xy=(best_row["k"], best_row["silhouette_score"]),
        xytext=(0, -24),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    if not selected_row.empty:
        row = selected_row.iloc[0]
        plt.scatter([row["k"]], [row["silhouette_score"]], color="#2d8f55", zorder=4)
        plt.annotate(
            f"selected k={selected_k}",
            xy=(row["k"], row["silhouette_score"]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("K-Means k comparison")
    plt.xticks(df["k"].astype(int))
    y_min = float(df["silhouette_score"].min())
    y_max = float(df["silhouette_score"].max())
    margin = max((y_max - y_min) * 0.18, 0.0005)
    plt.ylim(y_min - margin, y_max + margin)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    output_path = output_dir / "clustering_k_comparison.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def build_retrieval_score_plot(retrieval_summary_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(retrieval_summary_path).sort_values("mean_top_k_score", ascending=True)

    plt.figure(figsize=(8.2, 5.2))
    y = np.arange(len(df))
    plt.barh(y - 0.18, df["mean_top_k_score"], height=0.34, label="Mean top-5 cosine")
    plt.barh(y + 0.18, df["top_score"], height=0.34, label="Top cosine")
    plt.yticks(y, df["query"])
    plt.xlabel("Cosine similarity")
    plt.title("Retrieval query score comparison")
    plt.xlim(0, 1.05)
    plt.grid(True, axis="x", alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = output_dir / "retrieval_query_scores.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from experiment outputs.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--output-dir", default="outputs/visuals")
    parser.add_argument("--feature-selection", default="outputs/feature_selection/feature_selection_linear_svm.csv")
    parser.add_argument("--clustering-comparison", default="outputs/clustering/clustering_k_comparison.csv")
    parser.add_argument("--retrieval-summary", default="outputs/retrieval/retrieval_query_summary.csv")
    parser.add_argument("--feature-selection-k", type=int, default=20000)
    parser.add_argument("--selected-clustering-k", type=int, default=7)
    parser.add_argument("--save-curve-points", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    metrics_path = build_classification_curves(
        Path(args.data),
        output_dir,
        feature_selection_k=args.feature_selection_k,
        random_state=args.random_state,
        save_curve_points=args.save_curve_points,
    )
    feature_plot_path = build_feature_selection_plot(Path(args.feature_selection), output_dir)
    clustering_plot_path = build_clustering_plot(
        Path(args.clustering_comparison),
        output_dir,
        selected_k=args.selected_clustering_k,
    )
    retrieval_plot_path = build_retrieval_score_plot(Path(args.retrieval_summary), output_dir)

    print("Report figures generated.")
    print(f"Classification curve metrics: {metrics_path}")
    print(f"Precision-Recall curve: {output_dir / 'classification_pr_curve.png'}")
    print(f"ROC curve: {output_dir / 'classification_roc_curve.png'}")
    print(f"Feature-selection plot: {feature_plot_path}")
    print(f"Clustering comparison plot: {clustering_plot_path}")
    print(f"Retrieval score plot: {retrieval_plot_path}")


if __name__ == "__main__":
    main()
