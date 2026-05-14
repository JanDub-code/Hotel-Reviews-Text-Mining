from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Hotel_Reviews.csv"
OUTPUT_DIR = ROOT / "outputs"


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download Hotel_Reviews.csv first."
        )

    run_command(
        [
            "python",
            str(ROOT / "src" / "classification.py"),
            "--data",
            str(DATA_PATH),
            "--full-dataset",
            "--use-bigrams",
            "--feature-selection-k",
            "20000",
            "--output-dir",
            str(OUTPUT_DIR),
        ]
    )

    run_command(
        [
            "python",
            str(ROOT / "src" / "clustering.py"),
            "--data",
            str(DATA_PATH),
            "--sample-size",
            "50000",
            "--k",
            "7",
            "--use-bigrams",
            "--output-dir",
            str(OUTPUT_DIR),
        ]
    )

    run_command(
        [
            "python",
            str(ROOT / "src" / "retrieval.py"),
            "--data",
            str(DATA_PATH),
            "--full-dataset",
            "--query",
            "dirty room and noisy street",
            "--top-k",
            "5",
            "--output-dir",
            str(OUTPUT_DIR),
        ]
    )

    print("All modules finished.")


if __name__ == "__main__":
    main()
