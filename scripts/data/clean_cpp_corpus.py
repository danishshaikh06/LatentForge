"""Remove oversized license blocks and excess whitespace from a C++ corpus."""

import argparse
import re
from pathlib import Path

from latent_forge.core.paths import PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=PROCESSED_DATA_DIR / "cpp_dataset.txt")
    parser.add_argument(
        "--output-file", type=Path, default=PROCESSED_DATA_DIR / "cpp_dataset_clean.txt"
    )
    return parser.parse_args()


def clean_corpus(text: str) -> str:
    text = re.sub(r"/\*[\s\S]{200,}?\*/", "", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text)


def main() -> None:
    args = parse_args()
    text = args.input_file.read_text(encoding="utf-8", errors="ignore")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(clean_corpus(text), encoding="utf-8")
    print(f"Saved cleaned corpus to {args.output_file}")


if __name__ == "__main__":
    main()
