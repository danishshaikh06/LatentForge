"""Build a raw C++ corpus from the vendored source repositories."""

import argparse
from pathlib import Path

from latent_forge.core.paths import PROCESSED_DATA_DIR, THIRD_PARTY_DIR


CPP_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=THIRD_PARTY_DIR)
    parser.add_argument("--output-file", type=Path, default=PROCESSED_DATA_DIR / "cpp_dataset.txt")
    return parser.parse_args()


def build_corpus(source_dir: Path, output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with output_file.open("w", encoding="utf-8") as corpus_file:
        for source_file in source_dir.rglob("*"):
            if not source_file.is_file() or source_file.suffix.lower() not in CPP_EXTENSIONS:
                continue
            try:
                code = source_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            corpus_file.write(code)
            corpus_file.write("\n\n// FILE_SEPARATOR\n\n")
            file_count += 1
    return file_count


def main() -> None:
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {args.source_dir}")
    file_count = build_corpus(args.source_dir, args.output_file)
    print(f"Collected {file_count} C/C++ files into {args.output_file}")


if __name__ == "__main__":
    main()
