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
        raise ValueError("Provide at least one clustering k value.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-Means clustering experiments for negative hotel reviews.")
    parser.add_argument("--data", required=True, help="Path to Hotel_Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--k-values", default="5,6,7,8,9")
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/clustering")
    args = parser.parse_args()

    output_root = ensure_output_dir(args.output_dir)
    k_values = parse_k_values(args.k_values)

    comparison_rows: list[dict[str, object]] = []

    for k_value in k_values:
        run_dir = ensure_output_dir(output_root / f"k_{k_value}")
        command = [
            sys.executable,
            str(ROOT / "src" / "clustering.py"),
            "--data",
            args.data,
            "--sample-size",
            str(args.sample_size),
            "--k",
            str(k_value),
            "--min-df",
            str(args.min_df),
            "--max-df",
            str(args.max_df),
            "--max-features",
            str(args.max_features),
            "--random-state",
            str(args.random_state),
            "--output-dir",
            str(run_dir),
        ]
        if args.use_bigrams:
            command.append("--use-bigrams")
        subprocess.run(command, check=True)

        results = json.loads((run_dir / "clustering_results.json").read_text(encoding="utf-8"))
        cluster_sizes = list(results["cluster_sizes"].values())
        comparison_rows.append(
            {
                "k": k_value,
                "silhouette_score": results["silhouette_score"],
                "n_documents": results["dataset"]["n_documents"],
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
                "output_dir": str(run_dir),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("k")
    comparison_path = output_root / "clustering_k_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_row = comparison_df.sort_values("silhouette_score", ascending=False).iloc[0]
    summary_lines = [
        "# Phase 6 experiment summary",
        "",
        "## Setup",
        "- text column: `negative_text`",
        f"- TF-IDF bigrams enabled: `{args.use_bigrams}`",
        f"- tested k values: `{', '.join(str(value) for value in k_values)}`",
        "",
        "## Best k by silhouette score",
        f"- best k: `{int(best_row['k'])}`",
        f"- silhouette score: `{best_row['silhouette_score']:.4f}`",
        f"- output directory: `{best_row['output_dir']}`",
        "",
        f"- comparison table: `{comparison_path}`",
    ]
    summary_md_path = output_root / "phase6_summary.md"
    summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Clustering experiments finished.")
    print(comparison_df.to_string(index=False))
    print(f"Comparison saved to: {comparison_path}")
    print(f"Markdown summary saved to: {summary_md_path}")


if __name__ == "__main__":
    main()
