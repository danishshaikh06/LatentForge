from pathlib import Path

from latent_forge.core.paths import PROCESSED_DATA_DIR


INPUT_FILES = [
    PROCESSED_DATA_DIR / "cpp_instruction_dataset.txt",
    PROCESSED_DATA_DIR / "cpp_instruction_dataset_v2.txt",
]
OUTPUT_FILE = PROCESSED_DATA_DIR / "fine_tuning_dataset.txt"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    merged_parts = []

    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        merged_parts.append(read_text(path))

    merged_text = "\n\n".join(part for part in merged_parts if part) + "\n"

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(merged_text)

    print(f"Merged {len(INPUT_FILES)} files.")
    print(f"Saved merged dataset to: {OUTPUT_FILE}")
    print(f"Output size: {OUTPUT_FILE.stat().st_size} bytes")


if __name__ == "__main__":
    main()
