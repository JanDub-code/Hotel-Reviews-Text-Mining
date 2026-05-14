from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import (
    basic_clean_text,
    create_binary_sentiment_labels,
    create_three_class_labels,
    ensure_output_dir,
    load_dataset,
)


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "linear_svm": LinearSVC(random_state=random_state),
    }


def get_top_features(model_name: str, trained_model: object, feature_names: np.ndarray, top_n: int = 20) -> dict:
    model = trained_model
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        model = model.named_steps["classifier"]

    if not hasattr(model, "coef_"):
        return {}

    coef = model.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    result: dict[str, dict[str, list[str]]] = {}
    if coef.shape[0] == 1:
        weights = coef[0]
        top_positive_idx = np.argsort(weights)[-top_n:][::-1]
        top_negative_idx = np.argsort(weights)[:top_n]
        result[model_name] = {
            "top_positive_terms": feature_names[top_positive_idx].tolist(),
            "top_negative_terms": feature_names[top_negative_idx].tolist(),
        }
    else:
        classes = getattr(model, "classes_", [f"class_{i}" for i in range(coef.shape[0])])
        per_class = {}
        for idx, class_name in enumerate(classes):
            top_idx = np.argsort(coef[idx])[-top_n:][::-1]
            per_class[str(class_name)] = feature_names[top_idx].tolist()
        result[model_name] = per_class
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline sentiment classification for hotel reviews.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=50000, help="Optional sample size")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Use all available rows and ignore --sample-size.",
    )
    parser.add_argument("--label-scheme", choices=["binary", "three_class"], default="binary")
    parser.add_argument("--text-column", default="review_text", choices=["review_text", "positive_text", "negative_text"])
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--feature-selection-k", type=int, default=0, help="0 disables SelectKBest(chi2)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    effective_sample_size = None if args.full_dataset else args.sample_size
    df = load_dataset(args.data, sample_size=effective_sample_size, random_state=args.random_state)
    rows_loaded = int(len(df))
    if args.label_scheme == "binary":
        df = create_binary_sentiment_labels(df)
    else:
        df = create_three_class_labels(df)
    rows_after_labeling = int(len(df))

    df = df[df[args.text_column].fillna("").str.len() > 0].copy()
    rows_after_text_filter = int(len(df))
    df["clean_text"] = df[args.text_column].map(basic_clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    rows_after_cleaning = int(len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.random_state,
    )

    ngram_range = (1, 2) if args.use_bigrams else (1, 1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    labels = sorted(df["label"].unique().tolist())

    models = build_models(random_state=args.random_state)
    config = vars(args).copy()
    if args.full_dataset:
        config["sample_size"] = None
    config["effective_sample_size"] = effective_sample_size

    all_results: dict[str, dict] = {
        "config": config,
        "dataset": {
            "rows_loaded": rows_loaded,
            "rows_after_labeling": rows_after_labeling,
            "rows_after_text_filter": rows_after_text_filter,
            "rows_after_cleaning": rows_after_cleaning,
            "label_distribution": df["label"].value_counts().to_dict(),
        },
        "preprocessing": {
            "text_column": args.text_column,
            "label_scheme": args.label_scheme,
            "cleaning_steps": [
                "placeholder replacement for Positive_Review and Negative_Review",
                "drop empty selected text column",
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
            "rows_dropped": {
                "by_label_filter": rows_loaded - rows_after_labeling,
                "by_empty_selected_text": rows_after_labeling - rows_after_text_filter,
                "by_empty_clean_text": rows_after_text_filter - rows_after_cleaning,
            },
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "metrics": {},
        "top_features": {},
    }

    for model_name, classifier in models.items():
        steps = [("vectorizer", vectorizer)]
        if args.feature_selection_k and args.feature_selection_k > 0:
            steps.append(("feature_selection", SelectKBest(score_func=chi2, k=args.feature_selection_k)))
        steps.append(("classifier", classifier))
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        model_result = {
            "labels": labels,
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        all_results["metrics"][model_name] = model_result

        fitted_vectorizer = pipeline.named_steps["vectorizer"]
        feature_names = fitted_vectorizer.get_feature_names_out()
        if "feature_selection" in pipeline.named_steps:
            selector = pipeline.named_steps["feature_selection"]
            feature_names = feature_names[selector.get_support()]
        all_results["top_features"].update(
            get_top_features(model_name, pipeline, feature_names, top_n=20)
        )

    results_path = Path(output_dir) / "classification_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    preprocessing_path = Path(output_dir) / "preprocessing_summary.json"
    with preprocessing_path.open("w", encoding="utf-8") as f:
        json.dump(all_results["preprocessing"], f, indent=2, ensure_ascii=False)

    summary_rows = []
    for model_name, metrics in all_results["metrics"].items():
        summary_rows.append(
            {
                "model": model_name,
                "macro_precision": metrics["classification_report"]["macro avg"]["precision"],
                "macro_recall": metrics["classification_report"]["macro avg"]["recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["classification_report"]["accuracy"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_path = Path(output_dir) / "classification_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    confusion_rows = []
    for model_name, metrics in all_results["metrics"].items():
        metric_labels = metrics["labels"]
        for true_idx, true_label in enumerate(metric_labels):
            for pred_idx, pred_label in enumerate(metric_labels):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": metrics["confusion_matrix"][true_idx][pred_idx],
                    }
                )
    confusion_df = pd.DataFrame(confusion_rows)
    confusion_path = Path(output_dir) / "confusion_matrices.csv"
    confusion_df.to_csv(confusion_path, index=False)

    print("Classification finished.")
    print(summary_df.to_string(index=False))
    print(f"Detailed results saved to: {results_path}")
    print(f"Preprocessing summary saved to: {preprocessing_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Confusion matrices saved to: {confusion_path}")


if __name__ == "__main__":
    main()
