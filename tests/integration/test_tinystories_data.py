
import hashlib
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from latent_forge.data.tinystories import (
    load_tinystories_data,
)


DATA_ROOT = Path("data/tinystories_smoke")


@pytest.fixture(scope="module")
def prepared():
    if not (DATA_ROOT / "manifest.json").exists():
        pytest.skip("Run TinyStories preparation first")

    return load_tinystories_data(
        DATA_ROOT,
        seq_len=64,
    )


def test_tokenizer_round_trip(prepared):
    _, _, tokenizer, _ = prepared

    samples = [
        "Once upon a time, there was a little girl.",
        "The dog was happy!",
        "Hello 🌟 — café",
    ]

    for text in samples:
        token_ids = tokenizer.encode(text)
        reconstructed = tokenizer.decode(token_ids)

        assert reconstructed == text


def test_dataset_shapes(prepared):
    train_ds, val_ds, _, _ = prepared

    for dataset in (train_ds, val_ds):
        x, y = dataset[0]

        assert x.shape == (64,)
        assert y.shape == (64,)

        assert x.dtype == torch.long
        assert y.dtype == torch.long


def test_next_token_shift(prepared):
    train_ds, _, _, _ = prepared

    x, y = train_ds[0]

    assert torch.equal(
        x[1:],
        y[:-1],
    )


def test_token_ids_within_vocabulary(prepared):
    train_ds, val_ds, tokenizer, _ = prepared

    for dataset in (train_ds, val_ds):
        for idx in range(min(10, len(dataset))):
            x, y = dataset[idx]

            assert x.min() >= 0
            assert y.min() >= 0

            assert x.max() < tokenizer.vocab_size
            assert y.max() < tokenizer.vocab_size


def test_special_token_ids(prepared):
    _, _, _, manifest = prepared

    tokenizer = Tokenizer.from_file(
        str(DATA_ROOT / "tokenizer.json")
    )

    for name, key in [
        ("<pad>", "pad_id"),
        ("<bos>", "bos_id"),
        ("<eos>", "eos_id"),
    ]:
        assert tokenizer.token_to_id(name) == manifest[key]


def test_no_exact_train_validation_duplicates(prepared):
    train_path = DATA_ROOT / "train.jsonl"
    val_path = DATA_ROOT / "validation.jsonl"

    def hashes(path):
        with path.open(encoding="utf-8") as f:
            return {
                hashlib.sha256(
                    json.loads(line)["text"].encode("utf-8")
                ).hexdigest()
                for line in f
            }

    assert hashes(train_path).isdisjoint(
        hashes(val_path)
    )
