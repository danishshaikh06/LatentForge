import json
from pathlib import Path

from latent_forge.core.paths import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "cleaned_c++_data.json"
OUTPUT_FILE = PROCESSED_DATA_DIR / "cpp_instruction_dataset.txt"


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_example(row: dict) -> str:
    user = row.get("user", "").strip()
    assistant = row.get("assistant", "").strip()

    return (
        f"User: {user}\n"
        f"Assistant:\n{assistant}\n"
        f"\n\n"
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_FILE}")

    rows = load_rows(INPUT_FILE)
    formatted_examples = [format_example(row) for row in rows if row.get("user") and row.get("assistant")]

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.writelines(formatted_examples)

    print(f"Input rows: {len(rows)}")
    print(f"Formatted examples written: {len(formatted_examples)}")
    print(f"Saved formatted dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
