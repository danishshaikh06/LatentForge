
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


DATASET_ID = "roneneldan/TinyStories"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_story(text: str) -> str:
    # Keep the contents intact except for line endings
    # and surrounding whitespace.
    return text.replace("\r\n", "\n").strip()


def collect_stories(
    *,
    split: str,
    count: int,
    output_path: Path,
    revision: str,
    seed: int,
    shuffle_buffer: int,
    exclude_hashes: set[str] | None = None,
) -> set[str]:
    """
    Stream a deterministic subset of unique stories.

    The validation set can exclude exact duplicates
    that appeared in our selected training set.
    """
    dataset = load_dataset(
        DATASET_ID,
        split=split,
        revision=revision,
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer,
    )

    selected_hashes = set()
    excluded = exclude_hashes or set()

    with output_path.open(
        "w", encoding="utf-8"
    ) as f:
        for example in dataset:
            text = normalize_story(example["text"])

            if not text:
                continue

            fingerprint = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if fingerprint in selected_hashes:
                continue

            if fingerprint in excluded:
                continue

            f.write(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                ) + "\n"
            )

            selected_hashes.add(fingerprint)

            if len(selected_hashes) % 5000 == 0:
                print(
                    f"{split}: "
                    f"{len(selected_hashes):,} stories"
                )

            if len(selected_hashes) == count:
                break

    if len(selected_hashes) != count:
        raise RuntimeError(
            f"Requested {count} stories for {split}, "
            f"found only {len(selected_hashes)}."
        )

    return selected_hashes


def iter_stories(path: Path, limit: int | None = None):
    """
    Read locally selected stories without loading
    the entire text dataset into RAM.
    """
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if limit is not None and index >= limit:
                break

            yield json.loads(line)["text"]


def train_bpe_tokenizer(
    train_path: Path,
    tokenizer_path: Path,
    vocab_size: int,
    tokenizer_stories: int,
) -> Tokenizer:

    tokenizer = Tokenizer(
        models.BPE(unk_token=None)
    )

    # Encode arbitrary UTF-8 text at the byte level.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # Fit on training stories only.
    tokenizer.train_from_iterator(
        iter_stories(train_path, tokenizer_stories),
        trainer=trainer,
        length=tokenizer_stories,
    )

    tokenizer.save(str(tokenizer_path))

    # Verify special IDs are present.
    for name in SPECIAL_TOKENS:
        if tokenizer.token_to_id(name) is None:
            raise RuntimeError(f"Missing special token: {name}")

    # Verify lossless byte-level round trip.
    sample = "Once upon a time, Lily said: Hello! 🌟"

    ids = tokenizer.encode(
        sample,
        add_special_tokens=False,
    ).ids

    recovered = tokenizer.decode(
        ids,
        skip_special_tokens=False,
    )

    if recovered != sample:
        raise RuntimeError(
            "Tokenizer failed UTF-8 round-trip test"
        )

    return tokenizer


def batches(iterator, batch_size=128):
    batch = []

    for value in iterator:
        batch.append(value)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def encode_split(
    *,
    input_path: Path,
    output_path: Path,
    offsets_path: Path,
    tokenizer: Tokenizer,
) -> dict:
    """
    Pack stories into a contiguous uint16 token file.

    Every story:
        BOS + tokenized story + EOS

    Offsets identify where each story begins.
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    token_count = 0
    story_count = 0
    offsets = []

    with output_path.open("wb") as out:
        for text_batch in batches(iter_stories(input_path)):

            encodings = tokenizer.encode_batch(
                text_batch,
                add_special_tokens=False,
            )

            for encoding in encodings:
                offsets.append(token_count)

                ids = (
                    [bos_id]
                    + encoding.ids
                    + [eos_id]
                )

                array = np.asarray(
                    ids,
                    dtype=np.uint16,
                )

                array.tofile(out)

                token_count += len(ids)
                story_count += 1

    np.save(
        offsets_path,
        np.asarray(offsets, dtype=np.int64),
    )

    if story_count == 0:
        raise RuntimeError(
            f"No stories were encoded from {input_path}"
        )

    return {
        "stories": story_count,
        "tokens": token_count,
        "average_tokens_per_story": (
            token_count / story_count
        ),
        "sha256": sha256_file(output_path),
    }


def prepare(args):
    if args.vocab_size < 512 or args.vocab_size > 65535:
        raise ValueError(
            "vocab_size must be between 512 and 65535 "
            "when storing token IDs as uint16"
        )

    if min(
        args.train_stories,
        args.val_stories,
        args.tokenizer_stories,
        args.shuffle_buffer,
    ) <= 0:
        raise ValueError("All counts must be positive")

    if args.tokenizer_stories > args.train_stories:
        raise ValueError(
            "tokenizer_stories cannot exceed train_stories"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pin the dataset revision for reproducibility.
    revision = HfApi().dataset_info(
        DATASET_ID,
        revision=args.revision,
    ).sha

    train_text = output_dir / "train.jsonl"
    val_text = output_dir / "validation.jsonl"
    tokenizer_path = output_dir / "tokenizer.json"

    train_tokens = output_dir / "train.bin"
    val_tokens = output_dir / "validation.bin"

    train_offsets = output_dir / "train_offsets.npy"
    val_offsets = output_dir / "validation_offsets.npy"

    print("Dataset revision:", revision)

    train_hashes = collect_stories(
        split="train",
        count=args.train_stories,
        output_path=train_text,
        revision=revision,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )

    collect_stories(
        split="validation",
        count=args.val_stories,
        output_path=val_text,
        revision=revision,
        seed=args.seed + 1,
        shuffle_buffer=args.shuffle_buffer,
        exclude_hashes=train_hashes,
    )

    tokenizer = train_bpe_tokenizer(
        train_path=train_text,
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
        tokenizer_stories=args.tokenizer_stories,
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    print("Actual vocabulary size:", actual_vocab_size)

    train_stats = encode_split(
        input_path=train_text,
        output_path=train_tokens,
        offsets_path=train_offsets,
        tokenizer=tokenizer,
    )

    val_stats = encode_split(
        input_path=val_text,
        output_path=val_tokens,
        offsets_path=val_offsets,
        tokenizer=tokenizer,
    )

    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": revision,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "tokenizer_training_stories": args.tokenizer_stories,
        "vocab_size": actual_vocab_size,
        "dtype": "uint16",
        "pad_id": tokenizer.token_to_id("<pad>"),
        "bos_id": tokenizer.token_to_id("<bos>"),
        "eos_id": tokenizer.token_to_id("<eos>"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_text_sha256": sha256_file(train_text),
        "validation_text_sha256": sha256_file(val_text),
        "train": train_stats,
        "validation": val_stats,
    }

    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print("Preparation complete:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/tinystories_v1",
    )
    parser.add_argument(
        "--train-stories",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--val-stories",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--tokenizer-stories",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--revision",
        default="main",
    )

    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
