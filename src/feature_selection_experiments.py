from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import ensure_output_dir


ROOT = Path(__file__).resolve().parents[1]


def parse_k_values(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("Provide at least one k value.")
    return values


def output_dir_name(k_value: int) -> str:
    return "baseline" if k_value == 0 else f"k_{k_value}"


def run_experiment(args: argparse.Namespace, k_value: int, output_root: Path) -> Path:
    run_dir = ensure_output_dir(output_root / output_dir_name(k_value))
    command = [
        sys.executable,
        str(ROOT / "src" / "classification.py"),
        "--data",
        args.data,
        "--sample-size",
        str(args.sample_size),
        "--label-scheme",
        args.label_scheme,
        "--text-column",
        args.text_column,
        "--min-df",
        str(args.min_df),
        "--max-df",
        str(args.max_df),
        "--max-features",
        str(args.max_features),
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(args.random_state),
        "--output-dir",
        str(run_dir),
    ]
    if args.use_bigrams:
        command.append("--use-bigrams")
    if k_value > 0:
        command.extend(["--feature-selection-k", str(k_value)])

    subprocess.run(command, check=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SelectKBest(chi2) experiments for hotel review classification.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--label-scheme", choices=["binary", "three_class"], default="binary")
    parser.add_argument("--text-column", default="review_text", choices=["review_text", "positive_text", "negative_text"])
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--k-values", default="0,5000,10000,20000,30000")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/feature_selection")
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    output_root = ensure_output_dir(args.output_dir)

    summary_rows: list[dict[str, object]] = []
    run_results: dict[int, dict] = {}
    run_dirs: dict[int, Path] = {}

    for k_value in k_values:
        run_dir = run_experiment(args, k_value, output_root)
        run_dirs[k_value] = run_dir

        summary_df = pd.read_csv(run_dir / "classification_summary.csv")
        results = json.loads((run_dir / "classification_results.json").read_text(encoding="utf-8"))
        run_results[k_value] = results

        for row in summary_df.to_dict(orient="records"):
            summary_rows.append(
                {
                    "feature_selection_k": k_value,
                    "model": row["model"],
                    "macro_precision": row["macro_precision"],
                    "macro_recall": row["macro_recall"],
                    "macro_f1": row["macro_f1"],
                    "weighted_f1": row["weighted_f1"],
                    "accuracy": row["accuracy"],
                    "output_dir": str(run_dir),
                }
            )

    all_summary_df = pd.DataFrame(summary_rows).sort_values(
        ["model", "macro_f1", "feature_selection_k"],
        ascending=[True, False, True],
    )
    summary_path = output_root / "feature_selection_summary.csv"
    all_summary_df.to_csv(summary_path, index=False)

    linear_svm_df = all_summary_df[all_summary_df["model"] == "linear_svm"].sort_values(
        "feature_selection_k"
    )
    linear_svm_path = output_root / "feature_selection_linear_svm.csv"
    linear_svm_df.to_csv(linear_svm_path, index=False)

    best_row = all_summary_df.sort_values("macro_f1", ascending=False).iloc[0]
    best_k = int(best_row["feature_selection_k"])
    best_model = str(best_row["model"])
    best_results = run_results[best_k]
    best_features = best_results["top_features"].get(best_model, {})

    top_feature_rows = []
    if "top_positive_terms" in best_features:
        for rank, term in enumerate(best_features["top_positive_terms"], start=1):
            top_feature_rows.append(
                {"sentiment": "positive", "rank": rank, "term": term, "model": best_model, "feature_selection_k": best_k}
            )
        for rank, term in enumerate(best_features["top_negative_terms"], start=1):
            top_feature_rows.append(
                {"sentiment": "negative", "rank": rank, "term": term, "model": best_model, "feature_selection_k": best_k}
            )
    else:
        for class_name, terms in best_features.items():
            for rank, term in enumerate(terms, start=1):
                top_feature_rows.append(
                    {"sentiment": class_name, "rank": rank, "term": term, "model": best_model, "feature_selection_k": best_k}
                )

    top_features_df = pd.DataFrame(top_feature_rows)
    top_features_path = output_root / "feature_selection_top_features.csv"
    top_features_df.to_csv(top_features_path, index=False)

    baseline_linear_svm = linear_svm_df[linear_svm_df["feature_selection_k"] == 0].iloc[0]
    best_linear_svm = linear_svm_df.sort_values("macro_f1", ascending=False).iloc[0]
    delta_macro_f1 = float(best_linear_svm["macro_f1"] - baseline_linear_svm["macro_f1"])
    delta_accuracy = float(best_linear_svm["accuracy"] - baseline_linear_svm["accuracy"])

    summary_lines = [
        "# Phase 5 summary",
        "",
        "## Setup",
        f"- text column: `{args.text_column}`",
        f"- label scheme: `{args.label_scheme}`",
        f"- TF-IDF bigrams enabled: `{args.use_bigrams}`",
        f"- tested k values: `{', '.join(str(value) for value in k_values)}`",
        "",
        "## Best overall configuration",
        f"- best model: `{best_model}`",
        f"- best feature-selection k: `{best_k}`",
        f"- macro F1: `{best_row['macro_f1']:.4f}`",
        f"- accuracy: `{best_row['accuracy']:.4f}`",
        "",
        "## Linear SVM comparison",
        f"- baseline macro F1 without feature selection: `{baseline_linear_svm['macro_f1']:.4f}`",
        f"- best macro F1 with feature selection: `{best_linear_svm['macro_f1']:.4f}` at `k={int(best_linear_svm['feature_selection_k'])}`",
        f"- delta macro F1: `{delta_macro_f1:+.4f}`",
        f"- delta accuracy: `{delta_accuracy:+.4f}`",
        "",
        "## Output files",
        f"- summary table: `{summary_path}`",
        f"- linear SVM comparison: `{linear_svm_path}`",
        f"- top features for the best configuration: `{top_features_path}`",
    ]
    summary_md_path = output_root / "phase5_summary.md"
    summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Feature selection experiments finished.")
    print(all_summary_df.to_string(index=False))
    print(f"Summary saved to: {summary_path}")
    print(f"Linear SVM summary saved to: {linear_svm_path}")
    print(f"Top features saved to: {top_features_path}")
    print(f"Markdown summary saved to: {summary_md_path}")


if __name__ == "__main__":
    main()
