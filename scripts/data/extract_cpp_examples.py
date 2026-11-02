import argparse
import json
import re
from pathlib import Path

from latent_forge.core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


DEFAULT_INPUT_FILE = RAW_DATA_DIR / "code_alpaca_20k.json"
DEFAULT_OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_c++_data.json"


CPP_INSTRUCTION_HINTS = [
    "c++",
    "cpp",
    "std::",
    "iostream",
    "vector",
]

CPP_CODE_HINTS = [
    "#include",
    "std::",
    "using namespace std",
    "int main(",
    "cout",
    "cin",
    "vector<",
    "string ",
    "class ",
    "struct ",
    "::",
    "public:",
    "private:",
    "protected:",
    "return 0;",
]

NON_CPP_CODE_HINTS = [
    "def ",
    "import ",
    "print(",
    "if __name__ == \"__main__\"",
    "if __name__ == '__main__'",
    "self",
    "lambda ",
    "console.log",
    "function(",
    "function ",
    "select ",
    "<?php",
    "system.out.println",
    "system.out.print",
    "public static void main",
    "import java.",
    "public class",
    "private class",
    "protected class",
    "arraylist<",
    "hashmap<",
    "hashset<",
    "linkedlist<",
    "scanner",
    "bufferedreader",
    "inputstreamreader",
    "stringbuilder",
    "throws exception",
    "@override",
    "new scanner",
    "new arraylist",
    "new hashmap",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract C++ instruction/code examples from a dataset.")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to the source dataset file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save the cleaned C++ dataset.",
    )
    return parser.parse_args()


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        first_non_ws = ""
        while True:
            char = f.read(1)
            if not char:
                break
            if not char.isspace():
                first_non_ws = char
                break

        f.seek(0)

        if first_non_ws == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array dataset.")
            return data

        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def looks_like_cpp_instruction(instruction: str, input_text: str) -> bool:
    combined = normalize_text(f"{instruction} {input_text}")
    return any(hint in combined for hint in CPP_INSTRUCTION_HINTS)


def looks_like_cpp_code(output: str) -> bool:
    lowered = output.lower()

    if any(hint in lowered for hint in NON_CPP_CODE_HINTS):
        return False

    score = 0
    for hint in CPP_CODE_HINTS:
        if hint.lower() in lowered:
            score += 1

    return score >= 2


def extract_fields(row: dict) -> dict:
    instruction = (
        row.get("instruction")
        or row.get("query")
        or row.get("prompt")
        or ""
    ).strip()

    input_text = (
        row.get("input")
        or row.get("context")
        or ""
    ).strip()

    output = (
        row.get("output")
        or row.get("answer")
        or row.get("response")
        or ""
    ).strip()

    lang = str(row.get("lang", "")).strip().lower()

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "lang": lang,
    }


def build_record(fields: dict) -> dict:
    instruction = fields["instruction"]
    input_text = fields["input"]
    output = fields["output"]

    user_parts = [instruction]
    if input_text and input_text != "< noinput >":
        user_parts.append(f"Input:\n{input_text}")

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "user": "\n\n".join(part for part in user_parts if part),
        "assistant": output,
    }


def should_keep(fields: dict) -> bool:
    if not fields["instruction"] or not fields["output"]:
        return False

    if fields["lang"] in {"cpp", "c++"}:
        return True

    return looks_like_cpp_instruction(fields["instruction"], fields["input"]) or looks_like_cpp_code(fields["output"])


def main():
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    data = load_dataset(input_file)

    cleaned_rows = []
    skipped_rows = 0

    for row in data:
        fields = extract_fields(row)
        if should_keep(fields):
            cleaned_rows.append(build_record(fields))
        else:
            skipped_rows += 1

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(cleaned_rows, f, indent=2, ensure_ascii=False)

    print(f"Input rows: {len(data)}")
    print(f"C++ rows kept: {len(cleaned_rows)}")
    print(f"Rows skipped: {skipped_rows}")
    print(f"Saved cleaned data to: {output_file}")


if __name__ == "__main__":
    main()
